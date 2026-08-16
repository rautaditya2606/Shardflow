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
    recv_tensor,
    recv_tensor_timed,
    send_token,
    send_token_timed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("node1")


class Node1Profiler:
    """Microsecond-accurate latency profiler for Node 1 decode steps."""

    def __init__(self):
        self.tcp_recv_times = []
        self.deserialize_times = []
        self.cpu_to_gpu_times = []
        self.node1_fwd_times = []
        self.head_sample_times = []
        self.cuda_sync_times = []
        self.tcp_send_times = []
        self.total_step_times = []

    def record(self, recv_ms, deser_ms, c2g_ms, fwd_ms, head_ms, sync_ms, send_ms, total_ms):
        self.tcp_recv_times.append(recv_ms)
        self.deserialize_times.append(deser_ms)
        self.cpu_to_gpu_times.append(c2g_ms)
        self.node1_fwd_times.append(fwd_ms)
        self.head_sample_times.append(head_ms)
        self.cuda_sync_times.append(sync_ms)
        self.tcp_send_times.append(send_ms)
        self.total_step_times.append(total_ms)

    def print_breakdown(self):
        if not self.total_step_times:
            return
        n = len(self.total_step_times)
        avg = lambda lst: (sum(lst) / len(lst)) if lst else 0.0
        p95 = lambda lst: sorted(lst)[int(len(lst) * 0.95)] if lst else 0.0

        avg_total = avg(self.total_step_times)
        print("\n" + "=" * 70, flush=True)
        print(f"⏱️ NODE 1 PER-TOKEN LATENCY PROFILER BREAKDOWN ({n} decode steps)", flush=True)
        print("=" * 70, flush=True)
        print(f"  1. TCP Recv Wait (from Relay):{avg(self.tcp_recv_times):6.2f} ms  (p95: {p95(self.tcp_recv_times):6.2f} ms)")
        print(f"  2. Tensor Deserialization:    {avg(self.deserialize_times):6.2f} ms  (p95: {p95(self.deserialize_times):6.2f} ms)")
        print(f"  3. CPU -> GPU Transfer:       {avg(self.cpu_to_gpu_times):6.2f} ms  (p95: {p95(self.cpu_to_gpu_times):6.2f} ms)")
        print(f"  4. Node 1 GPU Forward:        {avg(self.node1_fwd_times):6.2f} ms  (p95: {p95(self.node1_fwd_times):6.2f} ms)")
        print(f"  5. Final Norm + LM Head + Smpl:{avg(self.head_sample_times):6.2f} ms  (p95: {p95(self.head_sample_times):6.2f} ms)")
        print(f"  6. CUDA Synchronize:          {avg(self.cuda_sync_times):6.2f} ms  (p95: {p95(self.cuda_sync_times):6.2f} ms)")
        print(f"  7. TCP Send Token (to EC2):   {avg(self.tcp_send_times):6.2f} ms  (p95: {p95(self.tcp_send_times):6.2f} ms)")
        print("  " + "-" * 66, flush=True)
        print(f"  TOTAL NODE 1 LATENCY:         {avg_total:6.2f} ms", flush=True)
        print("=" * 70, flush=True)


def main():
    parser = argparse.ArgumentParser(description="ShardFlow v2 Node 1 (Kaggle B TCP Relay Runner)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="Model path or HF ID (default: Qwen/Qwen2.5-14B-Instruct)")
    parser.add_argument("--layer-start", type=int, default=None, help="Starting layer index (default: half of total layers, e.g. 24 for 14B)")
    parser.add_argument("--layer-end", type=int, default=None, help="Ending layer index (default: total layers, e.g. 48 for 14B)")
    parser.add_argument("--4bit", action="store_true", help="Enable 4-bit NF4 loading")
    parser.add_argument("--relay-host", default=RELAY_HOST, help=f"EC2 Relay IP (default: {RELAY_HOST})")
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT, help=f"EC2 Relay Port (default: {RELAY_PORT})")
    parser.add_argument("--device", default="cuda", help="Target device (default: cuda)")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16", help="Precision (default: float16)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Sampling top-k")
    parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top-p")
    parser.add_argument("--eos-token-id", type=int, default=151643, help="EOS token ID")
    parser.add_argument("--no-cuda-graphs", action="store_true", default=True, help="Disable CUDA Graphs in eager mode")
    args = parser.parse_args()

    model_path = args.model if os.path.exists(args.model) else args.model
    config = AutoConfig.from_pretrained(model_path)
    total_layers = getattr(config, "num_hidden_layers", 48)

    layer_start = args.layer_start if args.layer_start is not None else (total_layers // 2)
    layer_end = args.layer_end if args.layer_end is not None else total_layers

    target_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    if target_dtype == torch.bfloat16 and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.warning("GPU does not support native BF16 (T4 is CC 7.5). Overriding to torch.float16.")
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
    profiler = Node1Profiler()

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
                # 1. Receive activation tensor from Node 0 timed
                try:
                    tensor, drafts, recv_stats = recv_tensor_timed(sock)
                except (ConnectionError, EOFError):
                    logger.warning("Relay connection closed by peer. Waiting to reconnect...")
                    break

                # If this is a prefill sequence (length > 1), reset KV cache and print profiler summary of previous prompt
                if tensor.shape[1] > 1:
                    if step > 0:
                        profiler.print_breakdown()
                    node.kv_store.evict(session_id)
                    step = 0

                t_step_0 = time.perf_counter()

                # 2. Move tensor to GPU
                t_c2g_0 = time.perf_counter()
                tensor_gpu = tensor.to(node.model_slice.device, non_blocking=False)
                t_c2g_1 = time.perf_counter()

                if drafts:
                    # Speculative verification
                    t_fwd_0 = time.perf_counter()
                    output = node._forward(
                        tensor_gpu,
                        session_id=session_id,
                        compute_head=True,
                    )
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
                        next_token = sample_next_token(
                            output[0, -1, :],
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                        )

                    accepted_count = len(accepted_tokens) + 1
                    cache = node.kv_store.get(session_id)
                    if cache is not None:
                        past_seq = node._get_cache_seq_len(cache)
                        past_seq_before = int(past_seq or 0) - (len(drafts) + 1)
                        rewind_kv_cache(cache, past_seq_before + accepted_count)

                    is_eos = (next_token == args.eos_token_id)
                    t_head_1 = time.perf_counter()

                    t_sync_0 = time.perf_counter()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_sync_1 = time.perf_counter()

                    send_stats = send_token_timed(sock, next_token, accepted_count=accepted_count, is_eos=is_eos)
                    t_step_1 = time.perf_counter()

                else:
                    # Standard 1-token decode
                    step += 1
                    t_fwd_0 = time.perf_counter()
                    hidden_out = node._forward(
                        tensor_gpu,
                        session_id=session_id,
                        compute_head=False,
                    )
                    t_fwd_1 = time.perf_counter()

                    t_head_0 = time.perf_counter()
                    if node.model_slice.norm is not None:
                        hidden_out = node.model_slice.norm(hidden_out)
                    if node.model_slice.lm_head is not None:
                        logits = node.model_slice.lm_head(hidden_out)
                    else:
                        logits = hidden_out

                    token_id = sample_next_token(
                        logits[0, -1, :],
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                    )
                    is_eos = (token_id == args.eos_token_id)
                    t_head_1 = time.perf_counter()

                    t_sync_0 = time.perf_counter()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_sync_1 = time.perf_counter()

                    send_stats = send_token_timed(sock, token_id, accepted_count=1, is_eos=is_eos)
                    t_step_1 = time.perf_counter()

                    profiler.record(
                        recv_ms=recv_stats["tcp_recv_ms"],
                        deser_ms=recv_stats["deserialize_ms"],
                        c2g_ms=(t_c2g_1 - t_c2g_0) * 1000.0,
                        fwd_ms=(t_fwd_1 - t_fwd_0) * 1000.0,
                        head_ms=(t_head_1 - t_head_0) * 1000.0,
                        sync_ms=(t_sync_1 - t_sync_0) * 1000.0,
                        send_ms=send_stats["tcp_send_ms"],
                        total_ms=(t_step_1 - t_step_0) * 1000.0,
                    )

                if step > 0 and (step % 30 == 0):
                    logger.info("Node 1 Step %4d | GPU Fwd: %.2f ms | Norm+Head+Smpl: %.2f ms", step, (t_fwd_1 - t_fwd_0)*1000, (t_head_1 - t_head_0)*1000)

        except (ConnectionError, socket.error) as e:
            logger.warning("Relay connection error (%s). Reconnecting in 3s...", e)
            time.sleep(3.0)
        except KeyboardInterrupt:
            logger.info("Shutting down Node 1...")
            profiler.print_breakdown()
            break
        except Exception as e:
            logger.exception("Unexpected error in Node 1: %s", e)
            time.sleep(2.0)


if __name__ == "__main__":
    main()
