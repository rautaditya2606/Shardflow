"""
ShardFlow All-in-One Demo Script

Spins up a 2-node distributed GPU pipeline, attaches the OpenAI-compatible Gateway,
and streams text generation using the official OpenAI Python SDK.
"""

import asyncio
import logging
import time
from threading import Thread
import torch
from openai import OpenAI

from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.node import PipelineNode
from shardflow.orchestrator.orchestrator import Orchestrator
from shardflow.gateway.app import app, set_orchestrator
import uvicorn


MODEL_PATH = "./models/TinyLlama-1.1B-Chat-v1.0"
GATEWAY_PORT = 8000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


async def setup_shardflow_pipeline():
    logger = logging.getLogger("shardflow")
    logger.setLevel(logging.WARNING)

    print(f"\n{'='*60}")
    print(f"🚀 INITIALIZING SHARDFLOW PIPELINE (Device: {DEVICE})")
    print(f"{'='*60}\n")

    print("1. Loading GPU model layer slices...")
    slice0 = load_layer_slice(MODEL_PATH, layer_start=0, layer_end=11, include_norm=False, include_lm_head=False, device=DEVICE)
    slice1 = load_layer_slice(MODEL_PATH, layer_start=11, layer_end=22, include_norm=True, include_lm_head=True, device=DEVICE)

    print("2. Starting Pipeline Nodes...")
    node1 = PipelineNode(slice1, is_first_node=False, is_last_node=True, listen_host="127.0.0.1", listen_port=9101)
    node0 = PipelineNode(slice0, is_first_node=True, is_last_node=False, next_node_host="127.0.0.1", next_node_port=9101, listen_host="127.0.0.1", listen_port=9100)

    await node1.start()
    await asyncio.sleep(0.2)
    await node0.start()
    await asyncio.sleep(0.2)

    print("3. Initializing Orchestrator & Connecting to Nodes...")
    orchestrator = Orchestrator(model_path=MODEL_PATH, node_addresses=[("127.0.0.1", 9100)], device="cpu")
    await orchestrator.initialize()
    set_orchestrator(orchestrator)

    print(f"4. Launching API Gateway on http://127.0.0.1:{GATEWAY_PORT}...\n")
    config = uvicorn.Config(app, host="127.0.0.1", port=GATEWAY_PORT, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


def main():
    # Start Shardflow server thread
    server_thread = Thread(target=lambda: asyncio.run(setup_shardflow_pipeline()), daemon=True)
    server_thread.start()

    # Wait for startup
    time.sleep(6)

    print(f"{'='*60}")
    print("🤖 TESTING STREAMING VIA OPENAI PYTHON SDK")
    print(f"{'='*60}\n")

    client = OpenAI(base_url=f"http://127.0.0.1:{GATEWAY_PORT}/v1", api_key="not-needed")

    prompt = "Tell me a short story about a brave knight"
    print(f"Prompt: '{prompt}'\nCompletion: ", end="", flush=True)

    stream = client.chat.completions.create(
        model="tinyllama",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=40,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print(f"\n\n{'='*60}")
    print("✅ DEMO COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
