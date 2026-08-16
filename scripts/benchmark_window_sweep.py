#!/usr/bin/env python3
"""
ShardFlow v2 — In-Flight Speculative Window Sweep Benchmark (Node 0).

Sweeps in-flight speculative window depths (W=1, 2, 3) with N-gram K=4
without reloading model weights to measure WAN bubble hiding and TPS scaling.
"""

import os
import sys
import time
import argparse
import statistics
from pathlib import Path
from typing import List, Optional

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
from shardflow.node.ngram_draft import NGramDraftSampler
from shardflow.transport.relay import (
    RELAY_HOST,
    RELAY_PORT,
    AUTH_BYTE,
    connect_to_relay,
    handshake,
)
from scripts.kaggle_node0 import Node0Profiler, generate, AsyncTokenReceiver


def run_window_sweep(
    model_path: str,
    window_list: List[int],
    spec_k: int = 4,
    draft_model: Optional[str] = None,
    draft_device: Optional[str] = None,
    layer_start: int = 0,
    layer_end: Optional[int] = None,
    relay_host: str = RELAY_HOST,
    relay_port: int = RELAY_PORT,
    device: str = "cuda",
    dtype: str = "float16",
    max_tokens: int = 60,
):
    config = AutoConfig.from_pretrained(model_path)
    total_layers = getattr(config, "num_hidden_layers", 28)
    layer_end = layer_end if layer_end is not None else (total_layers // 2)

    target_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    if target_dtype == torch.bfloat16 and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            target_dtype = torch.float16

    print("=" * 75)
    print("🔬 SHARDFLOW IN-FLIGHT SPECULATIVE WINDOW SWEEP BENCHMARK")
    print(f"Base Model:        {model_path} (Layers {layer_start}..{layer_end}) on {device}")
    if draft_model:
        print(f"Draft Model:       {draft_model} (Neural Draft, K={spec_k})")
    else:
        print(f"Draft Model:       N-gram Prompt Lookup (K={spec_k})")
    print(f"Window Sweep (W):  {window_list}")
    print(f"Relay:             {relay_host}:{relay_port}")
    print("=" * 75)

    print("\n[1/3] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    eos_id = getattr(tokenizer, "eos_token_id", 151643)

    print("[2/3] Loading Base Model Slice into VRAM...")
    t0 = time.perf_counter()
    model_slice = load_layer_slice(
        model_path=model_path,
        layer_start=layer_start,
        layer_end=layer_end,
        device=device,
        dtype=target_dtype,
    )
    print(f"✅ Model slice loaded in {time.perf_counter() - t0:.2f}s")

    print(f"[3/3] Initializing Node with spec_k={spec_k}...")
    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=True,
        is_last_node=False,
        draft_model=draft_model if draft_model else None,
        draft_device=draft_device,
        spec_k=spec_k,
    )

    if draft_model and node.draft_sampler is None:
        raise RuntimeError(f"Draft model '{draft_model}' failed to load on Node 0. Please check path.")

    if node.draft_sampler is not None:
        print("\n⚡ Warming up torch.compile kernels on draft model before connecting to relay...")
        t_w0 = time.perf_counter()
        dummy_tokens = [151644, 872, 198, 100, 200, 300, 400, 500]
        node.draft_sampler.prefill(dummy_tokens)
        _ = node.draft_sampler.generate_drafts(dummy_tokens[-1], k=spec_k)
        node.draft_sampler.reset()
        print(f"✅ Draft model warmup complete in {time.perf_counter() - t_w0:.2f}s")

    prompts = [
        "Explain quantum entanglement in simple terms.",
        "Write a Python function to compute Fibonacci numbers using dynamic programming.",
        "What are the key advantages of pipeline parallelism for distributed LLM inference?",
    ]

    results_table = []

    print("\nConnecting to relay...")
    sock = connect_to_relay(host=relay_host, port=relay_port, auth_byte=AUTH_BYTE)
    receiver = AsyncTokenReceiver(sock)
    try:
        handshake(sock)
        print("🌟 Handshake successful! Beginning Window-sweep iterations...\n")

        ngram_sampler = None
        if spec_k > 0 and not draft_model:
            ngram_sampler = NGramDraftSampler(max_ngram_size=3, min_ngram_size=1, spec_k=spec_k)

        for w in window_list:
            print("=" * 75)
            print(f"▶️ RUNNING BENCHMARK WITH SPECULATIVE WINDOW W = {w} (K={spec_k})")
            print("=" * 75)

            w_profiler = Node0Profiler()
            tps_list = []
            ttft_list = []

            for p_idx, prompt in enumerate(prompts, 1):
                prompt_prof = Node0Profiler()
                stats = generate(
                    prompt=prompt,
                    tokenizer=tokenizer,
                    node=node,
                    sock=sock,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    spec_k=spec_k,
                    ngram_sampler=ngram_sampler,
                    eos_token_id=eos_id,
                    profiler=prompt_prof,
                    receiver=receiver,
                    enable_async_spec=True,
                    spec_window=w,
                )

                if stats["tokens"] > 1:
                    tps_list.append(stats["tps"])
                    ttft_list.append(stats["ttft"])
                    for i in range(len(prompt_prof.total_step_times)):
                        w_profiler.record(
                            embed_ms=prompt_prof.embed_times[i],
                            gpu_fwd_ms=prompt_prof.node0_gpu_times[i],
                            g2c_ms=prompt_prof.gpu_to_cpu_times[i],
                            ser_ms=prompt_prof.serialize_times[i],
                            send_ms=prompt_prof.tcp_send_times[i],
                            recv_ms=prompt_prof.tcp_recv_wait_times[i],
                            total_ms=prompt_prof.total_step_times[i],
                            draft_gen_ms=prompt_prof.draft_gen_times[i],
                            draft_wait_ms=prompt_prof.draft_wait_times[i] if prompt_prof.draft_wait_times else 0.0,
                            gpu0_ms=prompt_prof.gpu0_fwd_times[i] if prompt_prof.gpu0_fwd_times else 0.0,
                            pcie_ms=prompt_prof.pcie_transfer_times[i] if prompt_prof.pcie_transfer_times else 0.0,
                            gpu1_ms=prompt_prof.gpu1_fwd_times[i] if prompt_prof.gpu1_fwd_times else 0.0,
                            node1_compute_ms=prompt_prof.node1_compute_times[i] if prompt_prof.node1_compute_times else 0.0,
                            network_rtt_ms=prompt_prof.pure_network_rtt_times[i] if prompt_prof.pure_network_rtt_times else 0.0,
                            bubble_ms=prompt_prof.inter_round_bubble_times[i] if prompt_prof.inter_round_bubble_times else 0.0,
                            accepted=prompt_prof.accepted_per_round[i],
                            drafted=prompt_prof.drafted_per_round[i],
                            is_spec=prompt_prof.is_spec_step[i],
                        )

            avg_tps = statistics.mean(tps_list) if tps_list else 0.0
            avg_ttft = statistics.mean(ttft_list) * 1000.0 if ttft_list else 0.0
            avg_fwd = statistics.mean(w_profiler.node0_gpu_times) if w_profiler.node0_gpu_times else 0.0
            avg_wait = statistics.mean(w_profiler.tcp_recv_wait_times) if w_profiler.tcp_recv_wait_times else 0.0
            avg_bubble = statistics.mean(w_profiler.inter_round_bubble_times) if w_profiler.inter_round_bubble_times else 0.0
            avg_net_rtt = statistics.mean(w_profiler.pure_network_rtt_times) if w_profiler.pure_network_rtt_times else 0.0
            avg_n1_comp = statistics.mean(w_profiler.node1_compute_times) if w_profiler.node1_compute_times else 0.0

            spec_acc = [acc for acc, is_s in zip(w_profiler.accepted_per_round, w_profiler.is_spec_step) if is_s]
            spec_drf = [drf for drf, is_s in zip(w_profiler.drafted_per_round, w_profiler.is_spec_step) if is_s]

            tok_per_round = (sum(spec_acc) / len(spec_acc)) if spec_acc else 1.0
            acc_rate = (sum(max(0, a - 1) for a in spec_acc) / sum(spec_drf) * 100.0) if (spec_drf and sum(spec_drf) > 0) else 0.0
            full_hits = sum(1 for acc, drf in zip(spec_acc, spec_drf) if acc == drf + 1)
            full_hit_rate = (full_hits / len(spec_acc) * 100.0) if spec_acc else 0.0

            results_table.append({
                "window": w,
                "tps": avg_tps,
                "ttft_ms": avg_ttft,
                "tokens_per_round": tok_per_round,
                "accept_rate": acc_rate,
                "full_hit_rate": full_hit_rate,
                "bubble_ms": avg_bubble,
                "fwd_ms": avg_fwd,
                "wait_ms": avg_wait,
                "n1_comp_ms": avg_n1_comp,
                "net_rtt_ms": avg_net_rtt,
            })

        # Print Final Comparison Table
        print("\n" + "=" * 115)
        print("📊 IN-FLIGHT SPECULATIVE WINDOW EMPIRICAL RESULTS (K=4 N-gram)")
        print("=" * 115)
        header = f"{'Window':>6} | {'TPS':>6} | {'TTFT (ms)':>9} | {'Tok/Round':>9} | {'Full Hit %':>10} | {'Bubble (ms)':>11} | {'N0 Fwd (ms)':>11} | {'N1 Comp (ms)':>12} | {'Net RTT (ms)':>12}"
        print(header)
        print("-" * 115)
        for row in results_table:
            print(
                f"{row['window']:6d} | "
                f"{row['tps']:6.2f} | "
                f"{row['ttft_ms']:9.1f} | "
                f"{row['tokens_per_round']:9.2f} | "
                f"{row['full_hit_rate']:9.1f}% | "
                f"{row['bubble_ms']:11.2f} | "
                f"{row['fwd_ms']:11.2f} | "
                f"{row['n1_comp_ms']:12.2f} | "
                f"{row['net_rtt_ms']:12.2f}"
            )
        print("=" * 115)

        if results_table:
            best = max(results_table, key=lambda r: r["tps"])
            print(f"\n🏆 Optimal In-Flight Window: W={best['window']} with {best['tps']:.2f} TPS ({best['tokens_per_round']:.2f} tokens/round, {best['full_hit_rate']:.1f}% full hits)")

    finally:
        if receiver is not None:
            receiver.stop()
        sock.close()


def main():
    parser = argparse.ArgumentParser(description="ShardFlow In-Flight Speculative Window Sweep Benchmark")
    parser.add_argument("--model", default="/kaggle/working/models/Qwen2.5-7B-Instruct", help="Model path or HF ID")
    parser.add_argument("--draft-model", default=None, help="Neural draft model path or HF ID (default: None, uses N-gram)")
    parser.add_argument("--draft-device", default=None, help="Device for draft model (default: cuda:1 if available, else cuda:0)")
    parser.add_argument("--spec-k", type=int, default=4, help="Speculative K (default: 4)")
    parser.add_argument("--windows", default="1,2,3", help="Comma-separated speculative window depths to test (default: 1,2,3)")
    parser.add_argument("--relay-host", default=RELAY_HOST, help="Relay IP")
    parser.add_argument("--layer-start", type=int, default=0, help="Starting layer index (default: 0)")
    parser.add_argument("--layer-end", type=int, default=14, help="Ending layer index (default: 14)")
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT, help="Relay port")
    parser.add_argument("--device", default="cuda", help="Target device (default: cuda)")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16", help="Precision (default: float16)")
    parser.add_argument("--max-tokens", type=int, default=60, help="Max tokens per generation")
    args = parser.parse_args()

    windows = [int(w.strip()) for w in args.windows.split(",") if w.strip()]
    run_window_sweep(
        model_path=args.model,
        window_list=windows,
        spec_k=args.spec_k,
        draft_model=args.draft_model,
        draft_device=args.draft_device,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        relay_host=args.relay_host,
        relay_port=args.relay_port,
        device=args.device,
        dtype=args.dtype,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
