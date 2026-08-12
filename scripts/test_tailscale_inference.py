"""
Test distributed inference directly across Tailscale Colab nodes.
"""

import asyncio
import time
import torch
from shardflow.orchestrator.orchestrator import Orchestrator

async def main():
    print("=" * 60)
    print("🚀 TESTING SHARDFLOW DISTRIBUTED INFERENCE ACROSS COLAB NODES")
    print("=" * 60)

    model_path = "Qwen/Qwen2.5-7B-Instruct"
    registry_url = "https://shardflow.onrender.com"

    print("\n1. Initializing Orchestrator (loading tokenizer & fetching topology)...")
    orch = Orchestrator(
        model_path=model_path,
        registry_url=registry_url,
        device="cpu",
    )
    await orch.initialize()

    nodes = await orch.fetch_topology_async(force=True)
    print(f"Active Cluster Topology: {nodes}")

    prompt = "Explain why the sky is blue in 2 short bullet points."
    print(f"\nPrompt: '{prompt}'")
    print("\n2. Generating completion (streaming)...\n")

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0

    print("Response: ", end="", flush=True)
    async for chunk in orch.generate_stream(
        prompt=prompt,
        max_tokens=40,
        temperature=0.7,
    ):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        print(chunk, end="", flush=True)
        token_count += 1
    print()

    total_time = time.perf_counter() - start_time
    ttft = (first_token_time - start_time) if first_token_time else total_time
    decode_time = total_time - ttft
    tps = token_count / decode_time if decode_time > 0 else 0

    print("\n" + "=" * 60)
    print(f"✅ GENERATION COMPLETE!")
    print(f"  Tokens Generated: {token_count}")
    print(f"  TTFT (Prefill):   {ttft:.3f}s")
    print(f"  Total Time:       {total_time:.3f}s")
    print(f"  Decode Throughput:{tps:.2f} tok/s")
    print("=" * 60)

    await orch.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
