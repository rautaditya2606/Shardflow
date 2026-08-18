#!/usr/bin/env python3
"""
ShardFlow Google Colab Node 1 Runner (Terminal Slice & Speculative Verifier).

Runs Node 1 (Layers 14..28 + LM Head on Qwen2.5-7B) on a Google Colab T4 GPU
connected directly to the AWS EC2 TCP Relay.

Usage on Colab (Notebook 1):
  !python scripts/colab_node1.py --model Qwen/Qwen2.5-7B-Instruct --relay-host <YOUR_RELAY_IP> --relay-port 9500
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

if os.path.exists("/content"):
    os.environ["HF_HOME"] = "/content/hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/content/hf_home"
    os.environ["HF_HUB_CACHE"] = "/content/hf_home"

import torch
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available in this Colab session! "
        "Please enable GPU acceleration via: Runtime -> Change runtime type -> T4 GPU."
    )

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
    recv_tensor_timed,
    send_token_timed,
)
from scripts.kaggle_node1 import Node1Profiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("colab_node1")


def main():
    parser = argparse.ArgumentParser(description="ShardFlow Google Colab Node 1 (Terminal Slice)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model ID or path")
    parser.add_argument("--layer-start", type=int, default=14, help="Starting layer index for Node 1 slice")
    parser.add_argument("--layer-end", type=int, default=None, help="Ending layer index (default: last layer)")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device to use (default: cuda:0)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--relay-host", type=str, default=RELAY_HOST, help="AWS EC2 relay IP/hostname")
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT, help="AWS EC2 relay port")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Top-K sampling (0=disabled)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling (1.0=disabled)")
    parser.add_argument("--eos-token-id", type=int, default=151643, help="EOS token ID")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load weights in 4-bit NF4")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model)
    total_layers = getattr(config, "num_hidden_layers", 28)
    layer_start = args.layer_start
    layer_end = args.layer_end if args.layer_end is not None else total_layers

    target_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    if target_dtype == torch.bfloat16 and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.warning("Device does not natively support bfloat16 (capability < 8.0). Falling back to float16.")
            target_dtype = torch.float16

    print("=" * 70)
    print(" SHARDFLOW GOOGLE COLAB NODE 1 (TERMINAL SLICE & VERIFIER)")
    print(f"Base Model:   {args.model}")
    print(f"Layer Split:  [{layer_start}, {layer_end}) of {total_layers} layers")
    print(f"Device:       {args.device} ({torch.cuda.get_device_name(0)})")
    print(f"Relay Target: {args.relay_host}:{args.relay_port}")
    print("=" * 70)

    # 1. Load Model Slice (Layers layer_start..layer_end + Final Norm + LM Head)
    t0 = time.perf_counter()
    model_slice = load_layer_slice(
        model_path=args.model,
        layer_start=layer_start,
        layer_end=layer_end,
        include_norm=(layer_end == total_layers),
        include_lm_head=(layer_end == total_layers),
        device=args.device,
        dtype=target_dtype,
        load_in_4bit=args.load_in_4bit,
    )
    t_load = time.perf_counter() - t0
    logger.info("Terminal model slice loaded in %0.2fs", t_load)

    # 2. Initialize Pipeline Node
    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=False,
        is_last_node=True,
    )

    profiler = Node1Profiler()

    # 3. Connection & Execution Loop
    while True:
        try:
            logger.info("Connecting to TCP relay at %s:%d ...", args.relay_host, args.relay_port)
            sock = connect_to_relay(host=args.relay_host, port=args.relay_port, auth_byte=AUTH_BYTE)
            logger.info("Connected to relay. Waiting for Node 0 to connect...")

            # Colab Node 1 is listener in the 2-way handshake
            handshake(sock, is_initiator=False)
            logger.info("HANDSHAKE COMPLETE! Entering pure compute decode loop...")

            session_id = "relay_session"
            step = 0
            last_verified_round_id = 0

            while True:
                # 1. Receive activation tensor from Node 0 timed
                try:
                    tensor, drafts, recv_stats = recv_tensor_timed(sock)
                except (ConnectionError, EOFError):
                    logger.warning("Relay connection closed by peer. Waiting to reconnect...")
                    break

                round_id = recv_stats.get("round_id", 0)
                parent_round_id = recv_stats.get("parent_round_id", 0)

                # Reset KV cache on prefill (length > 1 and no drafts)
                if tensor.shape[1] > 1 and not drafts:
                    if step > 0:
                        profiler.print_breakdown()
                    node.kv_store.evict(session_id)
                    step = 0
                    last_verified_round_id = 0

                # Check if this in-flight child was predicated on a mismatched parent round
                if parent_round_id > 0 and parent_round_id != last_verified_round_id:
                    send_token_timed(
                        sock,
                        token_id=0,
                        accepted_count=0,
                        is_eos=False,
                        compute_ms=0.0,
                        round_id=round_id,
                        is_stale_discard=True,
                    )
                    continue

                t_step_0 = time.perf_counter()

                # 2. Move tensor to GPU
                t_c2g_0 = time.perf_counter()
                tensor_gpu = tensor.to(node.model_slice.device, non_blocking=False)
                t_c2g_1 = time.perf_counter()

                if drafts:
                    # Speculative verification of K candidate tokens
                    t_fwd_0 = time.perf_counter()
                    output = node._forward(
                        tensor_gpu,
                        session_id=session_id,
                        compute_head=True,
                    )
                    if output.is_cuda:
                        torch.cuda.synchronize(output.device)
                    t_fwd_1 = time.perf_counter()

                    t_head_0 = time.perf_counter()
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

                    if accepted_count == len(drafts) + 1:
                        last_verified_round_id = round_id
                    else:
                        last_verified_round_id = 0

                    is_eos = (next_token == args.eos_token_id)
                    t_head_1 = time.perf_counter()
                    node1_compute_ms = (t_head_1 - t_c2g_0) * 1000.0

                    send_stats = send_token_timed(
                        sock,
                        next_token,
                        accepted_count=accepted_count,
                        is_eos=is_eos,
                        compute_ms=node1_compute_ms,
                        round_id=round_id,
                        is_stale_discard=False,
                    )
                    t_step_1 = time.perf_counter()
                    step += accepted_count

                    profiler.record(
                        recv_ms=recv_stats["tcp_recv_ms"],
                        deser_ms=recv_stats["deserialize_ms"],
                        c2g_ms=(t_c2g_1 - t_c2g_0) * 1000.0,
                        fwd_ms=(t_fwd_1 - t_fwd_0) * 1000.0,
                        head_ms=0.0,
                        smpl_ms=(t_head_1 - t_head_0) * 1000.0,
                        send_ms=send_stats["tcp_send_ms"],
                        total_ms=(t_step_1 - t_step_0) * 1000.0,
                    )

                else:
                    # Standard 1-token decode
                    step += 1
                    t_fwd_0 = time.perf_counter()
                    hidden_out = node._forward(
                        tensor_gpu,
                        session_id=session_id,
                        compute_head=False,
                    )
                    if hidden_out.is_cuda:
                        torch.cuda.synchronize(hidden_out.device)
                    t_fwd_1 = time.perf_counter()

                    t_head_0 = time.perf_counter()
                    if node.model_slice.norm is not None:
                        hidden_out = node.model_slice.norm(hidden_out)
                    if node.model_slice.lm_head is not None:
                        logits = node.model_slice.lm_head(hidden_out[:, -1:, :])
                    else:
                        logits = hidden_out[:, -1:, :]

                    next_token = sample_next_token(
                        logits[0, -1, :],
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                    )
                    is_eos = (next_token == args.eos_token_id)
                    t_head_1 = time.perf_counter()
                    node1_compute_ms = (t_head_1 - t_c2g_0) * 1000.0

                    send_stats = send_token_timed(
                        sock,
                        next_token,
                        accepted_count=1,
                        is_eos=is_eos,
                        compute_ms=node1_compute_ms,
                        round_id=round_id,
                        is_stale_discard=False,
                    )
                    t_step_1 = time.perf_counter()

                    profiler.record(
                        recv_ms=recv_stats["tcp_recv_ms"],
                        deser_ms=recv_stats["deserialize_ms"],
                        c2g_ms=(t_c2g_1 - t_c2g_0) * 1000.0,
                        fwd_ms=(t_fwd_1 - t_fwd_0) * 1000.0,
                        head_ms=(t_head_1 - t_head_0) * 1000.0,
                        smpl_ms=0.0,
                        send_ms=send_stats["tcp_send_ms"],
                        total_ms=(t_step_1 - t_step_0) * 1000.0,
                    )

        except (ConnectionError, TimeoutError, EOFError, socket.error) as e:
            logger.warning("Session ended or disconnected (%s). Re-entering connection loop in 2s...", e)
            time.sleep(2.0)
        except Exception as e:
            logger.error("Unexpected error in Colab Node 1: %s", e, exc_info=True)
            time.sleep(2.0)


if __name__ == "__main__":
    main()
