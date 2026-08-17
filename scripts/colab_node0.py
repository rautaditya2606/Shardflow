#!/usr/bin/env python3
"""
ShardFlow Google Colab Node 0 Runner (Initiator Slice + Neural Drafter).

Runs Node 0 (Layers 0..14 on Qwen2.5-7B) along with the Neural Draft Model (Qwen2.5-0.5B)
on a Google Colab T4 GPU (sharing cuda:0 in ~8.1 GB VRAM) connected to the AWS EC2 TCP Relay.

Usage on Colab (Notebook 2):
  !python scripts/colab_node0.py --model Qwen/Qwen2.5-7B-Instruct --draft-model Qwen/Qwen2.5-0.5B-Instruct --spec-k 8 --relay-host 3.23.174.207 --relay-port 9500
"""

import os
import sys
import time
import socket
import argparse
import logging
from typing import Optional, Tuple, List, Dict, Union
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
from shardflow.transport.relay import (
    RELAY_HOST,
    RELAY_PORT,
    AUTH_BYTE,
    connect_to_relay,
    handshake,
)
from scripts.kaggle_node0 import Node0Profiler, generate

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
    parser.add_argument("--layer-start", type=int, default=0, help="Starting layer index for Node 0 slice")
    parser.add_argument("--layer-end", type=int, default=14, help="Ending layer index for Node 0 slice")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device for base model slice")
    parser.add_argument("--draft-device", type=str, default="cuda:0", help="GPU device for draft model")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--relay-host", type=str, default=RELAY_HOST, help="AWS EC2 relay IP/hostname")
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT, help="AWS EC2 relay port")
    parser.add_argument("--max-tokens", type=int, default=60, help="Maximum generated tokens per prompt")
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
    )
    print(f"[OK] Model slice loaded in {time.perf_counter() - t0:.2f}s")

    # 3. Initialize Node with Neural Drafter
    print(f"\n[3/3] Initializing Node with spec_k={args.spec_k}...")
    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=True,
        is_last_node=False,
        draft_model=args.draft_model,
        draft_device=args.draft_device,
        spec_k=args.spec_k,
    )

    if node.draft_sampler is None:
        raise RuntimeError(f"Draft model '{args.draft_model}' failed to initialize on Node 0.")

    # Warmup draft model
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

    print("\nConnecting to relay...")
    sock = connect_to_relay(host=args.relay_host, port=args.relay_port, auth_byte=AUTH_BYTE)
    handshake(sock, is_initiator=True)
    print("Handshake successful! Beginning benchmark iterations...\n")

    profiler = Node0Profiler()
    summary_records = []

    for idx, prompt_text in enumerate(prompts):
        print(f"\nUser Prompt: \"{prompt_text}\"")
        print("Assistant: ", end="", flush=True)

        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
        session_id = f"colab_session_{int(time.time() * 1000)}"

        t_gen_0 = time.perf_counter()
        gen_tokens = generate(
            node=node,
            tokenizer=tokenizer,
            sock=sock,
            prompt_tokens=prompt_tokens,
            session_id=session_id,
            max_new_tokens=args.max_tokens,
            temperature=0.0,
            profiler=profiler,
            spec_k=args.spec_k,
            async_spec=False,
        )
        t_gen_1 = time.perf_counter()
        decode_time = t_gen_1 - t_gen_0

        stats = profiler.summary()
        tps = len(gen_tokens) / max(0.001, decode_time)

        print("\n" + "-" * 55)
        print(f"Tokens: {len(gen_tokens)} | TTFT: {stats['ttft_ms']:.1f} ms | Decode Time: {decode_time:.2f} s | Speed: {tps:.2f} TPS | Draft Accept Rate: {stats['accept_rate']:.1f}% ({stats['accepted_drafts']}/{stats['total_drafts']})")
        print("-" * 55)

        summary_records.append({
            "prompt_idx": idx + 1,
            "tps": tps,
            "ttft": stats["ttft_ms"],
            "tokens": len(gen_tokens),
            "decode_time": decode_time,
            "tok_per_round": stats.get("tok_per_round", 1.0),
            "full_hit_rate": stats.get("full_hit_rate", 0.0),
            "bubble_ms": stats.get("avg_bubble_ms", 0.0),
            "gpu_fwd_ms": stats.get("avg_gpu_fwd_ms", 0.0),
            "node1_comp_ms": stats.get("avg_node1_comp_ms", 0.0),
            "net_rtt_ms": stats.get("avg_network_rtt_ms", 0.0),
        })

    sock.close()

    print("\n" + "=" * 115)
    print(" SHARDFLOW GOOGLE COLAB DISTRIBUTED SPECULATIVE BENCHMARK RESULTS")
    print("=" * 115)
    print(f"{'Prompt #':<10} | {'TPS':>6} | {'TTFT (ms)':>9} | {'Tok/Round':>9} | {'Full Hit %':>10} | {'N0 Fwd (ms)':>11} | {'N1 Comp (ms)':>12} | {'Net RTT (ms)':>12}")
    print("-" * 115)

    for r in summary_records:
        print(f"{r['prompt_idx']:<10} | {r['tps']:>6.2f} | {r['ttft']:>9.1f} | {r['tok_per_round']:>9.2f} | {r['full_hit_rate']:>9.1f}% | {r['gpu_fwd_ms']:>11.2f} | {r['node1_comp_ms']:>12.2f} | {r['net_rtt_ms']:>12.2f}")
    print("=" * 115)

    import numpy as np
    avg_tps = float(np.mean([r["tps"] for r in summary_records]))
    avg_tok_round = float(np.mean([r["tok_per_round"] for r in summary_records]))
    print(f"\nAverage Throughput: {avg_tps:.2f} TPS across {len(summary_records)} prompts ({avg_tok_round:.2f} tokens/round)")


if __name__ == "__main__":
    main()
