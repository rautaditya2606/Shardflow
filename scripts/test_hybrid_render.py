"""
Hybrid E2E Test Script:
Runs 2 local LLM GPU nodes with bore.pub TCP tunnels, registers them to https://shardflow.onrender.com,
and calls the deployed Render API Gateway to execute distributed inference!
"""

import asyncio
import logging
import requests
import time
import torch
import sys
import random
from openai import OpenAI

from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.node import PipelineNode
from shardflow.transport.tunnel import start_bore_tunnel

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REGISTRY_URL = "https://shardflow.onrender.com"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hybrid_test")

async def run_hybrid_test():
    logger.info("=== STEP 2: HYBRID TEST (Local Nodes + Render Orchestrator) ===")
    logger.info("Connecting to Registry at %s...", REGISTRY_URL)

    # Generate unique random ports for bore.pub to avoid collision
    port_base = random.randint(31000, 45000)
    remote_port0 = port_base
    remote_port1 = port_base + 1

    # 1. Start Tunnel 0 for Node 0 (port 9000 -> bore.pub:remote_port0)
    logger.info("Starting bore tunnel for Node 0 (port 9000 -> bore.pub:%d)...", remote_port0)
    proc0, host0, port0 = start_bore_tunnel(9000, remote_port=remote_port0)
    logger.info("Node 0 public endpoint: %s:%d", host0, port0)

    # 2. Start Tunnel 1 for Node 1 (port 9001 -> bore.pub:remote_port1)
    logger.info("Starting bore tunnel for Node 1 (port 9001 -> bore.pub:%d)...", remote_port1)
    proc1, host1, port1 = start_bore_tunnel(9001, remote_port=remote_port1)
    logger.info("Node 1 public endpoint: %s:%d", host1, port1)

    try:
        # 3. Register Node 0 and Node 1 with Registry
        # Register Node 0 (node-0)
        logger.info("Registering node-0 with Render registry...")
        r0 = requests.post(f"{REGISTRY_URL}/register", json={
            "node_id": "node-0",
            "addr": host0,
            "port": port0,
            "vram_available_mb": 15000,
            "vram_total_mb": 15000,
            "model_id": MODEL_ID,
        }, timeout=10)
        logger.info("Node 0 registration response: %s", r0.json())

        # Register Node 1 (node-1)
        logger.info("Registering node-1 with Render registry...")
        r1 = requests.post(f"{REGISTRY_URL}/register", json={
            "node_id": "node-1",
            "addr": host1,
            "port": port1,
            "vram_available_mb": 15000,
            "vram_total_mb": 15000,
            "model_id": MODEL_ID,
        }, timeout=10)
        logger.info("Node 1 registration response: %s", r1.json())

        # Check topology on Render
        topo = requests.get(f"{REGISTRY_URL}/topology", timeout=10).json()
        logger.info("Render Topology: %s", topo)

        # 4. Instantiate local model slices based on registry response
        # Node 0 slice: layers [0, 11)
        slice0 = load_layer_slice(MODEL_ID, layer_start=0, layer_end=11, include_norm=False, include_lm_head=False, device=DEVICE)
        # Node 1 slice: layers [11, 22)
        slice1 = load_layer_slice(MODEL_ID, layer_start=11, layer_end=22, include_norm=True, include_lm_head=True, device=DEVICE)

        node1 = PipelineNode(slice1, is_first_node=False, is_last_node=True, listen_host="0.0.0.0", listen_port=9001)
        node0 = PipelineNode(slice0, is_first_node=True, is_last_node=False, next_node_host=host1, next_node_port=port1, listen_host="0.0.0.0", listen_port=9000)

        logger.info("Starting local pipeline nodes...")
        await node1.start()
        await asyncio.sleep(0.5)
        await node0.start()
        await asyncio.sleep(0.5)

        logger.info("Local nodes ready and accepting connections!")

        # 5. Send completion request to deployed Render orchestrator
        logger.info("Sending chat completion request to Render Gateway (%s)...", REGISTRY_URL)
        client = OpenAI(base_url=f"{REGISTRY_URL}/v1", api_key="shardflow")

        prompt = "Hello! Please tell me in one sentence what artificial intelligence is."
        logger.info("Prompt: '%s'", prompt)

        start_t = time.time()
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.0,
            stream=False,
        )
        duration = time.time() - start_t

        print("\n" + "="*60)
        print("🎉 RENDER ORCHESTRATOR GENERATION SUCCESSFUL!")
        print("="*60)
        print("Response:", completion.choices[0].message.content)
        print("Duration: %.2fs" % duration)
        print("="*60 + "\n")

    finally:
        logger.info("Cleaning up tunnels...")
        proc0.terminate()
        proc1.terminate()

if __name__ == "__main__":
    asyncio.run(run_hybrid_test())
