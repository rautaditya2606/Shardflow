#!/usr/bin/env python3
"""
ShardFlow v2 Kaggle Remote Node 1 Runner (Kaggle Instance B).

Runs Node 1 (Layers 24..48 + LM Head on Qwen2.5-14B, or Layers 14..28 on Qwen2.5-7B)
connected directly to the AWS EC2 Rust TCP Relay without tunnels.

Startup Sequence:
1. Start Node 1 first on Kaggle Instance B.
2. Node 1 connects to relay (3.23.174.207:9500) and waits for Node 0.
3. When Node 0 starts, relay pairs them and they execute READY handshake.
4. Node 1 enters pure compute decode loop (recv activation -> forward -> send token).
"""

import os
import sys
import time
import socket
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

if os.path.exists("/kaggle"):
    os.environ["HF_HOME"] = "/kaggle/working/hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/hf_home"
    os.environ["HF_HUB_CACHE"] = "/kaggle/working/hf_home"

import torch
from transformers import AutoConfig

from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.node import PipelineNode
from shardflow.node.draft_model import rewind_kv_cache
from shardflow.orchestrator.sampler import sample_next_token
from shardflow.transport.relay import (
    RELAY_HOST,
    RELAY_PORT,
    AUTH_BYTE,
    connect_to_relay,
    handshake,
    send_tensor,
    recv_tensor,
    send_token,
    recv_token,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("node1")


def main():
    parser = argparse.ArgumentParser(description="ShardFlow v2 Node 1 (Kaggle B TCP Relay Runner)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="Model path or HF ID (default: Qwen/Qwen2.5-14B-Instruct)")
    parser.add_argument("--layer-start", type=int, default=None, help="Starting layer index (default: half of total layers, e.g. 24 for 14B)")
    parser.add_argument("--layer-end", type=int, default=None, help="Ending layer index (default: total layers, e.g. 48 for 14B)")
    parser.add_argument("--4bit", action="store_true", help="Enable 4-bit NF4 loading")
    parser.add_argument("--relay-host", default=RELAY_HOST, help=f"EC2 Relay IP (default: {RELAY_HOST})")
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT, help=f"EC2 Relay Port (default: {RELAY_PORT})")
    parser.add_argument("--device", default="cuda", help="Target device (default: cuda)")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16", help="Precision (default: float16 for T4 Tensor Cores)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Sampling top_k")
    parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top_p")
    parser.add_argument("--eos-token-id", type=int, default=151643, help="EOS token ID (default: 151643 for Qwen2.5)")
    parser.add_argument("--no-cuda-graphs", action="store_true", default=True, help="Disable CUDA Graphs in eager mode")
    args = parser.parse_args()

    model_path = args.model if os.path.exists(args.model) else args.model
    config = AutoConfig.from_pretrained(model_path)
    total_layers = getattr(config, "num_hidden_layers", 48)

    layer_start = args.layer_start if args.layer_start is not None else (total_layers // 2)
    layer_end = args.layer_end if args.layer_end is not None else total_layers

    # Ensure FP16 is used on T4 (compute capability 7.5)
    target_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    if target_dtype == torch.bfloat16 and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.warning("GPU does not support native BF16 (T4 is CC 7.5). Overriding to torch.float16 for full Tensor Core speed.")
            target_dtype = torch.float16

    print("=" * 70, flush=True)
    print("🚀 SHARDFLOW v2 REMOTE NODE 1 (KAGGLE INSTANCE B)", flush=True)
    print(f"Base Model:    {model_path}")
    print(f"Layer Range:   [{layer_start}..{layer_end}) -> Indices {layer_start}..{layer_end-1} ({layer_end - layer_start}/{total_layers} layers + Final Norm + LM Head)")
    print(f"Relay Target:  {args.relay_host}:{args.relay_port}")
    print(f"Precision:     {target_dtype}")
    print(f"Device:        {args.device}")
    print("=" * 70, flush=True)

    # 1. Load Model Slice (Layers layer_start..layer_end + Final Norm + LM Head)
    logger.info("Loading model shard (layers %d..%d + LM Head)...", layer_start, layer_end)
    t0 = time.perf_counter()
    model_slice = load_layer_slice(
        model_path=model_path,
        layer_start=layer_start,
        layer_end=layer_end,
        include_norm=(layer_end == total_layers),
        include_lm_head=(layer_end == total_layers),
        device=args.device,
        dtype=target_dtype,
        load_in_4bit=getattr(args, "4bit", False),
    )
    logger.info("✅ Model slice loaded in %.2f s", time.perf_counter() - t0)

    # 2. Initialize Pipeline Node
    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=False,
        is_last_node=True,
        enable_cuda_graphs=not args.no_cuda_graphs,
    )
    if model_slice.config is not None:
        node.kv_store.initialize_static_pool(
            config=model_slice.config,
            device=model_slice.device,
            dtype=target_dtype,
        )

    # 3. Connect to Relay and enter compute loop with automatic reconnection
    while True:
        try:
            logger.info("Connecting to TCP relay at %s:%d ...", args.relay_host, args.relay_port)
            sock = connect_to_relay(host=args.relay_host, port=args.relay_port, auth_byte=AUTH_BYTE)
            logger.info("✅ Connected to relay. Waiting for Node 0 to connect...")

            handshake(sock)
            logger.info("🌟 HANDSHAKE COMPLETE! Entering pure compute decode loop...")

            session_id = "relay_session"
            step = 0

            while True:
                # Receive activation tensor from Node 0
                try:
                    tensor, drafts = recv_tensor(sock)
                except (ConnectionError, EOFError):
                    logger.warning("Relay connection closed by peer. Waiting to reconnect...")
                    break

                # If this is a prefill sequence (length > 1), reset KV cache for clean generation
                if tensor.shape[1] > 1:
                    node.kv_store.evict(session_id)
                    step = 0

                step += 1
                t_fwd_start = time.perf_counter()

                # Move tensor to GPU
                tensor_gpu = tensor.to(node.model_slice.device, non_blocking=False)

                # Execute forward pass through layers + norm + LM head
                output = node._forward(
                    tensor_gpu,
                    session_id=session_id,
                    compute_head=True,
                )

                if drafts:
                    # Speculative candidate verification
                    # Input length = len(drafts) + 1
                    # output[0, i, :] predicts token following candidate i -> compare with drafts[i]
                    accepted_tokens = []
                    next_token = None
                    for i in range(len(drafts)):
                        cand = sample_next_token(
                            output[0, i, :],
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                        )
                        if cand == drafts[i]:
                            accepted_tokens.append(drafts[i])
                        else:
                            next_token = cand
                            break

                    if next_token is None:
                        # All drafts accepted; bonus sample from final candidate position
                        next_token = sample_next_token(
                            output[0, -1, :],
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                        )

                    accepted_count = len(accepted_tokens) + 1

                    # Rewind KV cache to exact accepted sequence length
                    cache = node.kv_store.get(session_id)
                    if cache is not None:
                        past_seq = node._get_cache_seq_len(cache)
                        past_seq_before = int(past_seq or 0) - (len(drafts) + 1)
                        rewind_kv_cache(cache, past_seq_before + accepted_count)

                    is_eos = (next_token == args.eos_token_id)
                    send_token(sock, next_token, accepted_count=accepted_count, is_eos=is_eos)

                else:
                    # Standard 1-token decode or prefill
                    logits = output[0, -1, :]
                    token_id = sample_next_token(
                        logits,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                    )
                    is_eos = (token_id == args.eos_token_id)
                    send_token(sock, token_id, accepted_count=1, is_eos=is_eos)

                t_fwd = (time.perf_counter() - t_fwd_start) * 1000.0
                if step % 20 == 0 or step == 1:
                    logger.info("Step %4d | GPU Forward Compute: %.2f ms | Token ID: %d", step, t_fwd, token_id if not drafts else next_token)

        except (ConnectionError, socket.error) as e:
            logger.warning("Relay connection error (%s). Reconnecting in 3s...", e)
            time.sleep(3.0)
        except KeyboardInterrupt:
            logger.info("Shutting down Node 1...")
            break
        except Exception as e:
            logger.exception("Unexpected error in Node 1 compute loop: %s", e)
            time.sleep(2.0)


if __name__ == "__main__":
    main()
