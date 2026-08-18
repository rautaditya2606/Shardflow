#!/usr/bin/env python3
"""
ShardFlow Google Colab Node 0 Runner (Initiator Slice + Neural Drafter).

Runs Node 0 (Layers 0..14 on Qwen2.5-7B) along with the Neural Draft Model (Qwen2.5-0.5B)
on a Google Colab T4 GPU (sharing cuda:0 in ~8.6 GB VRAM) connected to the AWS EC2 TCP Relay.

Usage on Colab (Notebook 2):
  !python scripts/colab_node0.py --model Qwen/Qwen2.5-7B-Instruct --draft-model Qwen/Qwen2.5-0.5B-Instruct --spec-k 8 --relay-host <YOUR_RELAY_IP> --relay-port 9500
"""

import os
import sys
import time
import socket
import argparse
import logging
import statistics
from typing import Optional, List, Dict
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

from transformers import AutoTokenizer, AutoConfig
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("colab_node0")


def main():
    parser = argparse.ArgumentParser(description="ShardFlow Google Colab Node 0 (Initiator & Drafter)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model ID or path")
    parser.add_argument("--draft-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Draft model ID")
    parser.add_argument("--spec-k", type=int, default=8, help="Speculative candidate depth K (default: 8)")
    parser.add_argument("--windows", type=int, nargs="+", default=[1], help="In-flight speculative window depths to evaluate (default: 1)")
    parser.add_argument("--layer-start", type=int, default=0, help="Starting layer index for Node 0 slice")
    parser.add_argument("--layer-end", type=int, default=14, help="Ending layer index for Node 0 slice")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device for base model slice")
    parser.add_argument("--draft-device", type=str, default="cuda:0", help="GPU device for draft model")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--relay-host", type=str, default=RELAY_HOST, help="AWS EC2 relay IP/hostname")
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT, help="AWS EC2 relay port")
    parser.add_argument("--max-tokens", type=int, default=60, help="Maximum generated tokens per prompt")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load weights in 4-bit NF4")
    parser.add_argument("--enable-cuda-graphs", action="store_true", default=False, help="Enable CUDA Graphs for drafter (default: False for Colab shared single-GPU)")
    parser.add_argument("--no-cuda-graphs", action="store_true", help="Disable CUDA Graphs and run in high-accuracy eager draft mode")
    args = parser.parse_args()

    target_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    if target_dtype == torch.bfloat16 and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.warning("Device does not natively support bfloat16 (capability < 8.0). Falling back to float16.")
            target_dtype = torch.float16

    print("=" * 75)
    print(" SHARDFLOW GOOGLE COLAB NODE 0 (INITIATOR & SPECULATIVE BENCHMARK)")
    print(f"Base Model:    {args.model} (Layers {args.layer_start}..{args.layer_end}) on {args.device}")
    print(f"Draft Model:   {args.draft_model} (K={args.spec_k}) on {args.draft_device}")
    print(f"Relay Target:  {args.relay_host}:{args.relay_port}")
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    print("=" * 75)

    # 1. Load Tokenizer
    print("\n[1/3] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    eos_id = getattr(tokenizer, "eos_token_id", 151643)

    # 2. Load Base Model Slice
    print("\n[2/3] Loading Base Model Slice into VRAM...")
    t0 = time.perf_counter()
    model_slice = load_layer_slice(
        model_path=args.model,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        device=args.device,
        dtype=target_dtype,
        load_in_4bit=args.load_in_4bit,
    )
    print(f"[OK] Model slice loaded in {time.perf_counter() - t0:.2f}s")

    # 3. Initialize Node with Neural Drafter
    use_graphs = args.enable_cuda_graphs and not args.no_cuda_graphs
    print(f"\n[3/3] Initializing Node with spec_k={args.spec_k} (cuda_graphs={use_graphs})...")
    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=True,
        is_last_node=False,
        draft_model=args.draft_model,
        draft_device=args.draft_device,
        spec_k=args.spec_k,
        enable_cuda_graphs=use_graphs,
    )

    if args.draft_model and node.draft_sampler is None:
        raise RuntimeError(f"Draft model '{args.draft_model}' failed to initialize on Node 0.")

    # Warmup draft model
    if node.draft_sampler is not None:
        print("\nWarming up draft model before connecting to relay...")
        t_w0 = time.perf_counter()
        dummy_tokens = [151644, 872, 198, 100, 200, 300, 400, 500]
        node.draft_sampler.prefill(dummy_tokens)
        _ = node.draft_sampler.generate_drafts(dummy_tokens[-1], k=args.spec_k)
        node.draft_sampler.reset()
        print(f"[OK] Draft model warmup complete in {time.perf_counter() - t_w0:.2f}s")

    prompts = [
        "Explain quantum entanglement in simple terms.",
        "Write a Python function to compute Fibonacci numbers using dynamic programming.",
        "What are the key advantages of pipeline parallelism for distributed LLM inference?",
    ]

    results_table = []

    print("\nConnecting to relay...")
    sock = connect_to_relay(host=args.relay_host, port=args.relay_port, auth_byte=AUTH_BYTE)
    receiver = AsyncTokenReceiver(sock)

    try:
        handshake(sock, is_initiator=True)
        print("Handshake successful! Beginning benchmark iterations...\n")

        ngram_sampler = None
        if args.spec_k > 0 and not args.draft_model:
            ngram_sampler = NGramDraftSampler(max_ngram_size=3, min_ngram_size=1, spec_k=args.spec_k)

        for w in args.windows:
            print("=" * 75)
            print(f" RUNNING BENCHMARK WITH SPECULATIVE WINDOW W = {w} (K={args.spec_k})")
            print("=" * 75)

            w_profiler = Node0Profiler()
            tps_list = []
            ttft_list = []

            for p_idx, prompt_text in enumerate(prompts, 1):
                prompt_prof = Node0Profiler()
                stats = generate(
                    prompt=prompt_text,
                    tokenizer=tokenizer,
                    node=node,
                    sock=sock,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    spec_k=args.spec_k,
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
        sampler_name = f"Neural Draft {args.draft_model}" if args.draft_model else "N-gram Draft"
        print(f" SHARDFLOW COLAB SPECULATIVE RESULTS ({sampler_name}, K={args.spec_k})")
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
            print(f"\n[BEST] Optimal In-Flight Window: W={best['window']} with {best['tps']:.2f} TPS ({best['tokens_per_round']:.2f} tokens/round, {best['full_hit_rate']:.1f}% full hits)")

    finally:
        if receiver is not None:
            receiver.stop()
        sock.close()


if __name__ == "__main__":
    main()
