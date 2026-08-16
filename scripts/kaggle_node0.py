#!/usr/bin/env python3
"""
ShardFlow v2 Kaggle Remote Node 0 Runner (Kaggle Instance A).

Runs Node 0 (Layers 0..24 + Embeddings on Qwen2.5-14B, or Layers 0..14 on Qwen2.5-7B)
connected directly to the AWS EC2 Rust TCP Relay without tunnels.

Drives the distributed generation loop peer-to-peer across the relay:
Node 0 (Embed + Layers 0..24) -> Relay -> Node 1 (Layers 24..48 + LM Head) -> Relay -> Node 0
"""

import os
import sys
import time
import socket
import argparse
import logging
import statistics
from typing import Optional, Tuple, List, Dict, Union
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
from transformers import AutoConfig, AutoTokenizer

from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.node import PipelineNode
from shardflow.node.draft_model import DraftSampler, rewind_kv_cache
from shardflow.transport.relay import (
    RELAY_HOST,
    RELAY_PORT,
    AUTH_BYTE,
    connect_to_relay,
    handshake,
    send_tensor,
    send_tensor_timed,
    recv_tensor,
    recv_token,
    recv_token_timed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("node0")


class Node0Profiler:
    """Microsecond-accurate latency profiler for Node 0 decode steps."""

    def __init__(self):
        self.embed_times = []
        self.node0_gpu_times = []
        self.gpu_to_cpu_times = []
        self.serialize_times = []
        self.tcp_send_times = []
        self.tcp_recv_wait_times = []
        self.total_step_times = []

    def record(self, embed_ms, gpu_fwd_ms, g2c_ms, ser_ms, send_ms, recv_ms, total_ms):
        self.embed_times.append(embed_ms)
        self.node0_gpu_times.append(gpu_fwd_ms)
        self.gpu_to_cpu_times.append(g2c_ms)
        self.serialize_times.append(ser_ms)
        self.tcp_send_times.append(send_ms)
        self.tcp_recv_wait_times.append(recv_ms)
        self.total_step_times.append(total_ms)

    def print_breakdown(self):
        if not self.total_step_times:
            return
        n = len(self.total_step_times)
        avg = lambda lst: (sum(lst) / len(lst)) if lst else 0.0
        p95 = lambda lst: sorted(lst)[int(len(lst) * 0.95)] if lst else 0.0

        avg_total = avg(self.total_step_times)
        print("\n" + "=" * 70, flush=True)
        print(f"⏱️ NODE 0 PER-TOKEN LATENCY PROFILER BREAKDOWN ({n} decode steps)", flush=True)
        print("=" * 70, flush=True)
        print(f"  1. Token Embeddings:         {avg(self.embed_times):6.2f} ms  (p95: {p95(self.embed_times):6.2f} ms)")
        print(f"  2. Node 0 GPU Forward (Sync):{avg(self.node0_gpu_times):6.2f} ms  (p95: {p95(self.node0_gpu_times):6.2f} ms)")
        print(f"  3. GPU -> CPU Transfer:      {avg(self.gpu_to_cpu_times):6.2f} ms  (p95: {p95(self.gpu_to_cpu_times):6.2f} ms)")
        print(f"  4. Tensor Serialization:     {avg(self.serialize_times):6.2f} ms  (p95: {p95(self.serialize_times):6.2f} ms)")
        print(f"  5. TCP Send (Node 0 -> EC2): {avg(self.tcp_send_times):6.2f} ms  (p95: {p95(self.tcp_send_times):6.2f} ms)")
        print(f"  6. TCP Recv Wait (RTT+Node1):{avg(self.tcp_recv_wait_times):6.2f} ms  (p95: {p95(self.tcp_recv_wait_times):6.2f} ms)")
        print("  " + "-" * 66, flush=True)
        print(f"  TOTAL STEP LATENCY:          {avg_total:6.2f} ms  ({1000.0/avg_total:.2f} TPS)", flush=True)
        print("=" * 70, flush=True)


def generate(
    prompt: str,
    tokenizer,
    node: PipelineNode,
    sock: socket.socket,
    max_tokens: int = 100,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    spec_k: int = 0,
    eos_token_id: int = 151643,
    profiler: Optional[Node0Profiler] = None,
) -> dict:
    """Generate tokens for a prompt through the distributed relay pipeline with error handling."""
    session_id = f"relay_session_{int(time.time()*1000)}"
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)

    # Initialize draft sampler if speculative decoding is enabled
    draft_sampler = node.draft_sampler if (spec_k > 0 and node.draft_sampler is not None) else None
    if draft_sampler:
        draft_sampler.prefill(prompt_tokens)

    t_start = time.perf_counter()
    t_first_token = None
    generated_tokens = []
    total_drafted = 0
    total_accepted = 0

    print(f"\nUser Prompt: \"{prompt}\"", flush=True)
    print("Assistant: ", end="", flush=True)

    try:
        # 1. Prefill Phase
        token_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=node.model_slice.device)
        if node.model_slice.embed_tokens is not None:
            hidden_states = node.model_slice.embed_tokens(token_tensor)
        else:
            hidden_states = token_tensor

        # Forward pass on Node 0
        output = node._forward(hidden_states, session_id=session_id, compute_head=False)

        # Transmit activation to Node 1 via relay
        send_tensor(sock, output)

        # Receive prefill token from Node 1
        next_token, _, is_eos = recv_token(sock)
        t_first_token = time.perf_counter()
        generated_tokens.append(next_token)

        word = tokenizer.decode([next_token], skip_special_tokens=True)
        print(word, end="", flush=True)

        # 2. Autoregressive Decode Loop
        step = 1
        while step < max_tokens and not is_eos:
            if next_token == eos_token_id:
                break

            t_step_0 = time.perf_counter()

            if draft_sampler and spec_k > 0:
                # Speculative decoding path
                drafts = draft_sampler.generate_drafts(
                    next_token,
                    k=spec_k,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                total_drafted += len(drafts)

                candidate_tokens = [next_token] + drafts
                cand_tensor = torch.tensor([candidate_tokens], dtype=torch.long, device=node.model_slice.device)

                if node.model_slice.embed_tokens is not None:
                    cand_hidden = node.model_slice.embed_tokens(cand_tensor)
                else:
                    cand_hidden = cand_tensor

                cache = node.kv_store.get(session_id)
                past_seq_len = node._get_cache_seq_len(cache)

                cand_output = node._forward(cand_hidden, session_id=session_id, compute_head=False)
                send_stats = send_tensor_timed(sock, cand_output, draft_tokens=drafts)

                next_token, accepted_count, is_eos, recv_stats = recv_token_timed(sock)
                total_accepted += max(0, accepted_count - 1)

                if cache is not None:
                    rewind_kv_cache(cache, past_seq_len + accepted_count)
                draft_target = draft_sampler.seq_len - len(drafts) + accepted_count
                draft_sampler.rewind(draft_target)

                if drafts and accepted_count > 1:
                    for d_idx in range(accepted_count - 1):
                        tok = drafts[d_idx]
                        generated_tokens.append(tok)
                        print(tokenizer.decode([tok], skip_special_tokens=True), end="", flush=True)

                generated_tokens.append(next_token)
                print(tokenizer.decode([next_token], skip_special_tokens=True), end="", flush=True)
                step += accepted_count

            else:
                # Standard 1-token decode with precise phase timing
                t_emb_0 = time.perf_counter()
                tok_tensor = torch.tensor([[next_token]], dtype=torch.long, device=node.model_slice.device)
                if node.model_slice.embed_tokens is not None:
                    hidden = node.model_slice.embed_tokens(tok_tensor)
                else:
                    hidden = tok_tensor
                t_emb_1 = time.perf_counter()

                t_fwd_0 = time.perf_counter()
                output = node._forward(hidden, session_id=session_id, compute_head=False)
                if output.is_cuda:
                    torch.cuda.synchronize(output.device)
                t_fwd_1 = time.perf_counter()

                send_stats = send_tensor_timed(sock, output)

                next_token, _, is_eos, recv_stats = recv_token_timed(sock)
                t_step_1 = time.perf_counter()

                generated_tokens.append(next_token)
                print(tokenizer.decode([next_token], skip_special_tokens=True), end="", flush=True)
                step += 1

                if profiler is not None:
                    profiler.record(
                        embed_ms=(t_emb_1 - t_emb_0) * 1000.0,
                        gpu_fwd_ms=(t_fwd_1 - t_fwd_0) * 1000.0,
                        g2c_ms=send_stats["gpu_to_cpu_ms"],
                        ser_ms=send_stats["serialize_ms"],
                        send_ms=send_stats["tcp_send_ms"],
                        recv_ms=recv_stats["tcp_recv_ms"],
                        total_ms=(t_step_1 - t_step_0) * 1000.0,
                    )

    except TimeoutError as te:
        print(f"\n❌ [TIMEOUT ERROR]: {te}", flush=True)
        print("Kaggle Node 1 or EC2 Relay stopped responding. Please check Kaggle B status.", flush=True)
    except ConnectionError as ce:
        print(f"\n❌ [CONNECTION ERROR]: {ce}", flush=True)
    except Exception as ex:
        print(f"\n❌ [UNEXPECTED ERROR]: {ex}", flush=True)
    finally:
        node.kv_store.evict(session_id)

    t_end = time.perf_counter()
    total_time = t_end - t_start
    ttft = (t_first_token - t_start) if t_first_token else total_time
    decode_time = (t_end - t_first_token) if t_first_token else total_time
    tok_count = len(generated_tokens)
    tps = (tok_count - 1) / decode_time if (decode_time > 0 and tok_count > 1) else (tok_count / decode_time if decode_time > 0 else 0)

    print("\n" + "-" * 55, flush=True)
    stats_str = f"📊 Tokens: {tok_count} | TTFT: {ttft*1000:.1f} ms | Decode Time: {decode_time:.2f} s | Speed: {tps:.2f} TPS 🚀"
    if total_drafted > 0:
        accept_rate = (total_accepted / total_drafted) * 100.0
        stats_str += f" | Draft Accept Rate: {accept_rate:.1f}% ({total_accepted}/{total_drafted})"
    print(stats_str, flush=True)
    print("-" * 55, flush=True)

    return {
        "tokens": tok_count,
        "ttft": ttft,
        "decode_time": decode_time,
        "tps": tps,
        "draft_accept_rate": (total_accepted / total_drafted * 100.0) if total_drafted > 0 else None,
    }

    print("\n" + "-" * 55, flush=True)
    stats_str = f"📊 Tokens: {tok_count} | TTFT: {ttft*1000:.1f} ms | Decode Time: {decode_time:.2f} s | Speed: {tps:.2f} TPS 🚀"
    if total_drafted > 0:
        accept_rate = (total_accepted / total_drafted) * 100.0
        stats_str += f" | Draft Accept Rate: {accept_rate:.1f}% ({total_accepted}/{total_drafted})"
    print(stats_str, flush=True)
    print("-" * 55, flush=True)

    return {
        "tokens": tok_count,
        "ttft": ttft,
        "decode_time": decode_time,
        "tps": tps,
        "draft_accept_rate": (total_accepted / total_drafted * 100.0) if total_drafted > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description="ShardFlow v2 Node 0 (Kaggle A TCP Relay Runner)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="Model path or HF ID (default: Qwen/Qwen2.5-14B-Instruct)")
    parser.add_argument("--draft-model", default=None, help="Draft model path for speculative decoding (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--spec-k", type=int, default=0, help="Speculative candidate tokens (default: 0 to disable, 4 to test)")
    parser.add_argument("--layer-start", type=int, default=0, help="Starting layer index (default: 0)")
    parser.add_argument("--layer-end", type=int, default=None, help="Ending layer index (default: half of total layers, e.g. 24 for 14B)")
    parser.add_argument("--4bit", action="store_true", help="Enable 4-bit NF4 loading")
    parser.add_argument("--relay-host", default=RELAY_HOST, help=f"EC2 Relay IP (default: {RELAY_HOST})")
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT, help=f"EC2 Relay Port (default: {RELAY_PORT})")
    parser.add_argument("--device", default="cuda", help="Target device (default: cuda)")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16", help="Precision (default: float16)")
    parser.add_argument("--max-tokens", type=int, default=60, help="Max tokens per generation")
    parser.add_argument("--prompt", default=None, help="Single prompt to benchmark (optional)")
    parser.add_argument("--cuda-graphs", action="store_true", default=True, help="Enable CUDA Graphs and Static KV cache (default: True)")
    parser.add_argument("--no-cuda-graphs", action="store_false", dest="cuda_graphs", help="Disable CUDA Graphs and use DynamicCache fallback")
    parser.add_argument("--static-kv", action="store_true", default=True, help="Enable Static KV cache on GPU (default: True)")
    parser.add_argument("--no-static-kv", action="store_false", dest="static_kv", help="Disable Static KV cache")
    args = parser.parse_args()

    model_path = args.model if os.path.exists(args.model) else args.model
    config = AutoConfig.from_pretrained(model_path)
    total_layers = getattr(config, "num_hidden_layers", 48)

    layer_start = args.layer_start
    layer_end = args.layer_end if args.layer_end is not None else (total_layers // 2)

    target_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    if target_dtype == torch.bfloat16 and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.warning("GPU does not support native BF16 (T4 is CC 7.5). Overriding to torch.float16.")
            target_dtype = torch.float16

    use_cuda = (args.device == "cuda" or (isinstance(args.device, str) and args.device.startswith("cuda"))) and torch.cuda.is_available()
    enable_static_kv = args.static_kv and use_cuda
    enable_cuda_graphs = args.cuda_graphs and enable_static_kv

    print("=" * 70, flush=True)
    print("🚀 SHARDFLOW v2 REMOTE NODE 0 (KAGGLE INSTANCE A)", flush=True)
    print(f"Base Model:    {model_path}")
    print(f"Layer Range:   [{layer_start}..{layer_end}) -> Indices {layer_start}..{layer_end-1} ({layer_end - layer_start}/{total_layers} layers + Embeddings)")
    print(f"Draft Model:   {args.draft_model or 'DISABLED (spec_k=0)'} (Speculative K={args.spec_k})")
    print(f"CUDA Graphs:   {'ENABLED ⚡' if enable_cuda_graphs else 'DISABLED (eager mode)'}")
    print(f"Static KV:     {'ENABLED (GPU StaticCache)' if enable_static_kv else 'DISABLED (DynamicCache)'}")
    print(f"Relay Target:  {args.relay_host}:{args.relay_port}")
    print(f"Precision:     {target_dtype}")
    print(f"Device:        {args.device}")
    print("=" * 70, flush=True)

    # 1. Load Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    eos_id = getattr(tokenizer, "eos_token_id", 151643)

    # 2. Load Model Slice (Layers 0..layer_end + Embedding Table)
    logger.info("Loading model shard (layers %d..%d + Embeddings)...", layer_start, layer_end)
    t0 = time.perf_counter()
    model_slice = load_layer_slice(
        model_path=model_path,
        layer_start=layer_start,
        layer_end=layer_end,
        device=args.device,
        dtype=target_dtype,
        load_in_4bit=getattr(args, "4bit", False),
    )
    logger.info("✅ Model slice loaded in %.2f s", time.perf_counter() - t0)

    # 3. Initialize Pipeline Node & Draft Sampler
    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=True,
        is_last_node=False,
        draft_model=args.draft_model if args.spec_k > 0 else None,
        spec_k=args.spec_k,
        enable_cuda_graphs=enable_cuda_graphs,
    )
    node.kv_store.enable_static_cache = enable_static_kv

    if model_slice.config is not None and enable_static_kv:
        node.kv_store.initialize_static_pool(
            config=model_slice.config,
            device=model_slice.device,
            dtype=target_dtype,
        )
        if node.kv_store._static_slots and enable_cuda_graphs:
            logger.info("Capturing CUDA Graphs on Node 0...")
            captured = node.graph_runner.capture(node.kv_store._static_slots[0].cache)
            if captured:
                logger.info("✅ CUDA Graphs captured & active on Node 0!")

    # 4. Connect to Relay and perform handshake with Node 1
    logger.info("Connecting to TCP relay at %s:%d ...", args.relay_host, args.relay_port)
    sock = connect_to_relay(host=args.relay_host, port=args.relay_port, auth_byte=AUTH_BYTE)
    logger.info("✅ Connected to relay. Executing READY handshake with Node 1...")
    handshake(sock)
    logger.info("🌟 HANDSHAKE COMPLETE! Cluster is paired and ready for inference.")

    # 5. Run Live Inference Prompts
    prompts = [args.prompt] if args.prompt else [
        "Explain quantum entanglement in simple terms.",
        "Write a Python function to compute Fibonacci numbers using dynamic programming.",
        "What are the key advantages of pipeline parallelism for distributed LLM inference?",
    ]

    tps_results = []
    ttft_results = []
    global_profiler = Node0Profiler()

    try:
        for idx, prompt in enumerate(prompts, 1):
            print(f"\n" + "=" * 60)
            print(f"⚡ BENCHMARK PROMPT {idx}/{len(prompts)}")
            print("=" * 60)

            prompt_profiler = Node0Profiler()
            stats = generate(
                prompt=prompt,
                tokenizer=tokenizer,
                node=node,
                sock=sock,
                max_tokens=args.max_tokens,
                temperature=0.0,
                spec_k=args.spec_k,
                eos_token_id=eos_id,
                profiler=prompt_profiler,
            )
            if stats["tokens"] > 1:
                tps_results.append(stats["tps"])
                ttft_results.append(stats["ttft"])
                for i in range(len(prompt_profiler.total_step_times)):
                    global_profiler.record(
                        prompt_profiler.embed_times[i],
                        prompt_profiler.node0_gpu_times[i],
                        prompt_profiler.gpu_to_cpu_times[i],
                        prompt_profiler.serialize_times[i],
                        prompt_profiler.tcp_send_times[i],
                        prompt_profiler.tcp_recv_wait_times[i],
                        prompt_profiler.total_step_times[i],
                    )

            prompt_profiler.print_breakdown()

        if tps_results:
            print("\n" + "=" * 70)
            print("🏆 FINAL BENCHMARK SUMMARY (ShardFlow v2 over Direct TCP Relay)")
            print(f"  Model:                 {model_path}")
            print(f"  Avg Decode Throughput: {statistics.mean(tps_results):.2f} tokens/sec 🚀")
            print(f"  Max Decode Throughput: {max(tps_results):.2f} tokens/sec")
            print(f"  Avg TTFT:              {statistics.mean(ttft_results)*1000:.1f} ms")
            print(f"  Transport:             Direct TCP Relay ({args.relay_host}:{args.relay_port})")
            print("=" * 70)
            global_profiler.print_breakdown()

    finally:
        sock.close()


if __name__ == "__main__":
    main()
