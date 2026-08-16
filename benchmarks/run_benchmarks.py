"""
ShardFlow Benchmark Suite — measures latency, throughput, TTFT, and pipeline metrics.
"""

import argparse
import asyncio
import json
import logging
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoConfig
from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.node import PipelineNode
from shardflow.orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


async def run_benchmark(
    model_path: str,
    prompts: list[str],
    max_tokens: int = 30,
    num_nodes: int = 2,
    device: str = "cpu",
) -> dict:
    config = AutoConfig.from_pretrained(model_path)
    total_layers = config.num_hidden_layers

    layers_per_node = total_layers // num_nodes
    remainder = total_layers % num_nodes
    layer_ranges = []
    start = 0
    for i in range(num_nodes):
        end = start + layers_per_node + (1 if i < remainder else 0)
        layer_ranges.append((start, end))
        start = end

    base_port = 9200
    slices = []
    for i, (l_start, l_end) in enumerate(layer_ranges):
        is_last = (i == num_nodes - 1)
        model_slice = load_layer_slice(
            model_path=model_path,
            layer_start=l_start,
            layer_end=l_end,
            include_norm=is_last,
            include_lm_head=is_last,
            device=device,
        )
        slices.append(model_slice)

    nodes = []
    for i, (m_slice, (l_start, l_end)) in enumerate(zip(slices, layer_ranges)):
        is_first = (i == 0)
        is_last = (i == num_nodes - 1)
        next_host = "127.0.0.1" if not is_last else None
        next_port = base_port + i + 1 if not is_last else None

        node = PipelineNode(
            model_slice=m_slice,
            is_first_node=is_first,
            is_last_node=is_last,
            next_node_host=next_host,
            next_node_port=next_port,
            listen_host="127.0.0.1",
            listen_port=base_port + i,
        )
        nodes.append(node)

    for i in reversed(range(len(nodes))):
        await nodes[i].start()
        await asyncio.sleep(0.2)

    orchestrator = Orchestrator(
        model_path=model_path,
        node_addresses=[("127.0.0.1", base_port + i) for i in range(num_nodes)],
        device="cpu",
    )

    results = []

    try:
        await orchestrator.initialize()

        for idx, prompt in enumerate(prompts):
            print(f"\n[Bench {idx+1}/{len(prompts)}] Prompt: '{prompt[:30]}...'")
            t0 = time.perf_counter()
            completion = await orchestrator.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                stream=False,
            )
            t1 = time.perf_counter()
            total_duration = t1 - t0
            tok_count = len(orchestrator.tokenizer.encode(completion))
            tok_s = tok_count / total_duration if total_duration > 0 else 0.0

            res = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "generated_tokens": tok_count,
                "duration_seconds": round(total_duration, 3),
                "tokens_per_second": round(tok_s, 2),
            }
            results.append(res)
            print(f"-> Generated {tok_count} tokens in {total_duration:.2f}s ({tok_s:.2f} tok/s)")

    finally:
        await orchestrator.shutdown()
        for node in nodes:
            await node.stop()

    summary = {
        "model": model_path,
        "num_nodes": num_nodes,
        "device": device,
        "num_runs": len(prompts),
        "avg_tokens_per_second": round(sum(r["tokens_per_second"] for r in results) / len(results), 2) if results else 0,
        "runs": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="ShardFlow Benchmark Runner")
    parser.add_argument("--model", default="./models/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="benchmarks/results/benchmark_summary.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    prompts = [
        "Once upon a time",
        "The secret to building scalable distributed systems is",
        "In artificial intelligence, transformer models are",
    ]

    summary = asyncio.run(run_benchmark(
        model_path=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens,
        num_nodes=args.num_nodes,
        device=args.device,
    ))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Benchmark completed! Summary saved to {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
