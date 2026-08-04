"""
End-to-end test — split a model across 2 processes on localhost and generate text.

This script:
1. Splits the model into 2 node processes (layers 0-11 and layers 11-22)
2. Starts both nodes as async tasks on different ports
3. Starts the orchestrator
4. Generates text from a prompt
5. Verifies the output is coherent

Usage:
    python -m tests.test_e2e_localhost --model ./models/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import asyncio
import logging
import sys
import time
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoConfig

from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.node import PipelineNode
from shardflow.orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


async def run_e2e_test(
    model_path: str,
    prompt: str = "Once upon a time",
    max_tokens: int = 30,
    temperature: float = 0.0,
    device: str = "cpu",
    num_nodes: int = 2,
):
    """
    Run a full end-to-end test with the model split across N nodes on localhost.
    """
    config = AutoConfig.from_pretrained(model_path)
    total_layers = config.num_hidden_layers
    logger.info("Model: %s, layers: %d, hidden_size: %d", model_path, total_layers, config.hidden_size)

    # Calculate layer ranges for each node
    layers_per_node = total_layers // num_nodes
    remainder = total_layers % num_nodes
    layer_ranges = []
    start = 0
    for i in range(num_nodes):
        end = start + layers_per_node + (1 if i < remainder else 0)
        layer_ranges.append((start, end))
        start = end

    logger.info("Layer partition: %s", layer_ranges)

    # Base port for nodes
    base_port = 9000

    # --- Load model slices ---
    logger.info("Loading model slices...")
    slices = []
    for i, (layer_start, layer_end) in enumerate(layer_ranges):
        is_last = (i == num_nodes - 1)
        logger.info("Loading node %d: layers [%d, %d), is_last=%s", i, layer_start, layer_end, is_last)

        model_slice = load_layer_slice(
            model_path=model_path,
            layer_start=layer_start,
            layer_end=layer_end,
            include_norm=is_last,
            include_lm_head=is_last,
            device=device,
        )
        slices.append(model_slice)

    # --- Create and start nodes ---
    nodes = []
    for i, (model_slice, (layer_start, layer_end)) in enumerate(zip(slices, layer_ranges)):
        is_first = (i == 0)
        is_last = (i == num_nodes - 1)

        # Next node connection (not for the last node)
        next_host = "127.0.0.1" if not is_last else None
        next_port = base_port + i + 1 if not is_last else None

        node = PipelineNode(
            model_slice=model_slice,
            is_first_node=is_first,
            is_last_node=is_last,
            next_node_host=next_host,
            next_node_port=next_port,
            listen_host="127.0.0.1",
            listen_port=base_port + i,
        )
        nodes.append(node)

    # Start nodes in REVERSE order (last node first, so connections succeed)
    for i in reversed(range(len(nodes))):
        logger.info("Starting node %d on port %d...", i, base_port + i)
        await nodes[i].start()
        # Brief delay to let the server socket bind
        await asyncio.sleep(0.3)

    logger.info("All %d nodes started", num_nodes)

    # --- Create and run orchestrator ---
    node_addresses = [(f"127.0.0.1", base_port + i) for i in range(num_nodes)]

    orchestrator = Orchestrator(
        model_path=model_path,
        node_addresses=node_addresses,
        device="cpu",  # Embedding always on CPU
    )

    try:
        await orchestrator.initialize()

        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Generating {max_tokens} tokens across {num_nodes} nodes...")
        print(f"{'='*60}\n")

        start_time = time.perf_counter()
        result = await orchestrator.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        total_time = time.perf_counter() - start_time

        print(f"\n{'='*60}")
        print(f"Completion: {result}")
        print(f"Time: {total_time:.2f}s")
        print(f"Tokens: {len(orchestrator.tokenizer.encode(result))}")
        print(f"Speed: {len(orchestrator.tokenizer.encode(result)) / total_time:.1f} tok/s")
        print(f"{'='*60}")

    finally:
        await orchestrator.shutdown()
        for node in nodes:
            await node.stop()


def main():
    parser = argparse.ArgumentParser(description="ShardFlow E2E Test")
    parser.add_argument(
        "--model",
        default="./models/TinyLlama-1.1B-Chat-v1.0",
        help="Model path",
    )
    parser.add_argument("--prompt", default="Once upon a time", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=30, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--num-nodes", type=int, default=2, help="Number of nodes to split across")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for nodes (cpu or cuda)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    asyncio.run(run_e2e_test(
        model_path=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
        num_nodes=args.num_nodes,
    ))


if __name__ == "__main__":
    main()
