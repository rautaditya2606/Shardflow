#!/usr/bin/env python3
"""
ShardFlow v2 — Speculative Decoding K-Sweep Benchmark (Node 0).

Sweeps K values (e.g. K=0, 1, 2, 3, 4, 6, 8) across standard benchmark prompts
without reloading model weights to find the empirical optimal K and tokens/round.
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
from scripts.kaggle_node0 import Node0Profiler, generate


# ponytail: concise benchmark runner that reuses loaded model across K configurations
def run_k_sweep(
    model_path: str,
    k_list: List[int],
    draft_model: Optional[str] = None,
    layer_start: int = 0,
    layer_end: Optional[int] = None,
    relay_host: str = RELAY_HOST,
    relay_port: int = RELAY_PORT,
    device: str = "cuda",
    dtype: str = "float16",
    max_tokens: int = 60,
    load_in_4bit: bool = False,
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
    print("🔬 SHARDFLOW SPECULATIVE DECODING K-SWEEP BENCHMARK")
    print(f"Base Model:     {model_path} (Layers {layer_start}..{layer_end})")
    print(f"Draft Model:    {draft_model or 'N-gram prompt lookup'}")
    print(f"K Sweep Values: {k_list}")
    print(f"Relay:          {relay_host}:{relay_port}")
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
        load_in_4bit=load_in_4bit,
    )
    print(f"✅ Model slice loaded in {time.perf_counter() - t0:.2f}s")

    max_k = max(k_list)
    print(f"[3/3] Initializing Node with max_k={max_k}...")
    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=True,
        is_last_node=False,
        draft_model=draft_model if draft_model else None,
        spec_k=max_k,
    )

    if draft_model and node.draft_sampler is None:
        raise RuntimeError(f"Draft model '{draft_model}' failed to load on Node 0 GPU. Please check path or HF model ID.")

    prompts = [
        "Explain quantum entanglement in simple terms.",
        "Write a Python function to compute Fibonacci numbers using dynamic programming.",
        "What are the key advantages of pipeline parallelism for distributed LLM inference?",
    ]

    results_table = []

    print("\nConnecting to relay...")
    sock = connect_to_relay(host=relay_host, port=relay_port, auth_byte=AUTH_BYTE)
    try:
        handshake(sock)
        print("🌟 Handshake successful! Beginning K-sweep iterations...\n")

        for k in k_list:
            print("=" * 75)
            print(f"▶️ RUNNING BENCHMARK WITH K = {k}")
            print("=" * 75)

            ngram_sampler = None
            if k > 0 and not draft_model:
                ngram_sampler = NGramDraftSampler(max_ngram_size=3, min_ngram_size=1, spec_k=k)
            elif k > 0 and draft_model and node.draft_sampler:
                node.draft_sampler.spec_k = k

            k_profiler = Node0Profiler()
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
                    spec_k=k,
                    ngram_sampler=ngram_sampler,
                    eos_token_id=eos_id,
                    profiler=prompt_prof,
                )

                if stats["tokens"] > 1:
                    tps_list.append(stats["tps"])
                    ttft_list.append(stats["ttft"])
                    for i in range(len(prompt_prof.total_step_times)):
                        k_profiler.record(
                            embed_ms=prompt_prof.embed_times[i],
                            gpu_fwd_ms=prompt_prof.node0_gpu_times[i],
                            g2c_ms=prompt_prof.gpu_to_cpu_times[i],
                            ser_ms=prompt_prof.serialize_times[i],
                            send_ms=prompt_prof.tcp_send_times[i],
                            recv_ms=prompt_prof.tcp_recv_wait_times[i],
                            total_ms=prompt_prof.total_step_times[i],
                            draft_gen_ms=prompt_prof.draft_gen_times[i],
                            accepted=prompt_prof.accepted_per_round[i],
                            drafted=prompt_prof.drafted_per_round[i],
                            is_spec=prompt_prof.is_spec_step[i],
                        )

            avg_tps = statistics.mean(tps_list) if tps_list else 0.0
            avg_ttft = statistics.mean(ttft_list) * 1000.0 if ttft_list else 0.0
            avg_fwd = statistics.mean(k_profiler.node0_gpu_times) if k_profiler.node0_gpu_times else 0.0
            avg_wait = statistics.mean(k_profiler.tcp_recv_wait_times) if k_profiler.tcp_recv_wait_times else 0.0
            avg_draft = statistics.mean(k_profiler.draft_gen_times) if k_profiler.draft_gen_times else 0.0

            spec_acc = [acc for acc, is_s in zip(k_profiler.accepted_per_round, k_profiler.is_spec_step) if is_s]
            spec_drf = [drf for drf, is_s in zip(k_profiler.drafted_per_round, k_profiler.is_spec_step) if is_s]

            tok_per_round = (sum(spec_acc) / len(spec_acc)) if spec_acc else 1.0
            acc_rate = (sum(max(0, a - 1) for a in spec_acc) / sum(spec_drf) * 100.0) if (spec_drf and sum(spec_drf) > 0) else 0.0

            results_table.append({
                "k": k,
                "tps": avg_tps,
                "ttft_ms": avg_ttft,
                "tokens_per_round": tok_per_round,
                "accept_rate": acc_rate,
                "draft_ms": avg_draft,
                "fwd_ms": avg_fwd,
                "wait_ms": avg_wait,
            })

        # Print Final Comparison Table
        print("\n" + "=" * 88)
        print("📊 SPECULATIVE DECODING K-SWEEP EMPIRICAL RESULTS")
        print("=" * 88)
        header = f"{'K':>3} | {'TPS':>6} | {'TTFT (ms)':>9} | {'Tok/Round':>9} | {'Accept %':>8} | {'Draft (ms)':>10} | {'N0 Fwd (ms)':>11} | {'Wait (ms)':>9}"
        print(header)
        print("-" * 88)
        for row in results_table:
            print(
                f"{row['k']:3d} | "
                f"{row['tps']:6.2f} | "
                f"{row['ttft_ms']:9.1f} | "
                f"{row['tokens_per_round']:9.2f} | "
                f"{row['accept_rate']:7.1f}% | "
                f"{row['draft_ms']:10.2f} | "
                f"{row['fwd_ms']:11.2f} | "
                f"{row['wait_ms']:9.2f}"
            )
        print("=" * 88)

        if results_table:
            best = max(results_table, key=lambda r: r["tps"])
            print(f"\n🏆 Optimal Configuration: K={best['k']} with {best['tps']:.2f} TPS ({best['tokens_per_round']:.2f} tokens/round)")

    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(description="ShardFlow K-Sweep Speculative Decoding Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model path or HF ID")
    parser.add_argument("--draft-model", default=None, help="Draft model path for neural speculative (optional)")
    parser.add_argument("--k-values", default="0,1,2,3,4,6,8", help="Comma-separated K values to test")
    parser.add_argument("--relay-host", default=RELAY_HOST, help="Relay IP")
    parser.add_argument("--layer-start", type=int, default=0, help="Starting layer index (default: 0)")
    parser.add_argument("--layer-end", type=int, default=None, help="Ending layer index (default: total_layers // 2)")
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT, help="Relay port")
    parser.add_argument("--device", default="cuda", help="Target device")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16", help="Precision")
    parser.add_argument("--max-tokens", type=int, default=60, help="Max tokens per generation")
    parser.add_argument("--4bit", action="store_true", help="Enable 4-bit loading")
    args = parser.parse_args()

    k_list = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    run_k_sweep(
        model_path=args.model,
        k_list=k_list,
        draft_model=args.draft_model,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        relay_host=args.relay_host,
        relay_port=args.relay_port,
        device=args.device,
        dtype=args.dtype,
        max_tokens=args.max_tokens,
        load_in_4bit=getattr(args, "4bit", False),
    )


if __name__ == "__main__":
    main()
