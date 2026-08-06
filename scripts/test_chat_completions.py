"""
Comprehensive Chat Completion Test Script
1. Tests local LLM nodes + local Orchestrator/Gateway on http://127.0.0.1:8899
2. Tests local LLM nodes + deployed Render Orchestrator on https://shardflow.onrender.com
"""

import asyncio
import logging
import requests
import time
import torch
import uvicorn
import multiprocessing
from threading import Thread
from openai import OpenAI

from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.node import PipelineNode
from shardflow.orchestrator.orchestrator import Orchestrator
from shardflow.gateway.app import app, set_orchestrator
from shardflow.transport.tunnel import start_bore_tunnel

MODEL_PATH = "./models/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RENDER_URL = "https://shardflow.onrender.com"

logging.basicConfig(level=logging.WARNING)

def run_test_1():
    print("\n" + "="*70)
    print("1. TESTING CHAT COMPLETION (LOCAL LLM NODES + LOCAL FASTAPI ORCHESTRATOR)")
    print("="*70)

    # 1. Load slices
    slice0 = load_layer_slice(MODEL_PATH, layer_start=0, layer_end=11, include_norm=False, include_lm_head=False, device=DEVICE)
    slice1 = load_layer_slice(MODEL_PATH, layer_start=11, layer_end=22, include_norm=True, include_lm_head=True, device=DEVICE)

    # 2. Start nodes
    async def _setup():
        node1 = PipelineNode(slice1, is_first_node=False, is_last_node=True, listen_host="127.0.0.1", listen_port=9101)
        node0 = PipelineNode(slice0, is_first_node=True, is_last_node=False, next_node_host="127.0.0.1", next_node_port=9101, listen_host="127.0.0.1", listen_port=9100)
        await node1.start()
        await asyncio.sleep(0.2)
        await node0.start()
        await asyncio.sleep(0.2)
        orch = Orchestrator(model_path=MODEL_PATH, node_addresses=[("127.0.0.1", 9100)], device="cpu")
        await orch.initialize()
        set_orchestrator(orch)
        config = uvicorn.Config(app, host="127.0.0.1", port=8899, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    t = Thread(target=lambda: asyncio.run(_setup()), daemon=True)
    t.start()

    # Wait for gateway on 8899
    for _ in range(30):
        try:
            r = requests.get("http://127.0.0.1:8899/health", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.5)

    client = OpenAI(base_url="http://127.0.0.1:8899/v1", api_key="shardflow")
    prompt = "What is the capital of France?"
    print(f"User Prompt: '{prompt}'")
    
    start_t = time.time()
    response = client.chat.completions.create(
        model="tinyllama",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
        temperature=0.0,
        stream=False,
    )
    dur = time.time() - start_t
    content = response.choices[0].message.content
    print(f"Assistant Response: {content.strip()}")
    print(f"Latency: {dur:.2f}s")
    print("="*70 + "\n")


def run_test_2():
    async def _hybrid():
        print("\n" + "="*70)
        print("2. TESTING CHAT COMPLETION (LOCAL LLM NODES + DEPLOYED RENDER ORCHESTRATOR)")
        print("="*70)

        # Wake Render endpoint
        print("Waking Render instance at https://shardflow.onrender.com...")
        requests.get(f"{RENDER_URL}/health", timeout=15)

        # 1. Start Cloudflare Tunnels for local nodes
        print("Starting Cloudflare TCP Tunnels...")
        proc0, host0, port0 = start_cloudflare_tcp_tunnel(9200)
        proc1, host1, port1 = start_cloudflare_tcp_tunnel(9201)
        print(f"Node 0 public tunnel: {host0}:{port0}")
        print(f"Node 1 public tunnel: {host1}:{port1}")

        try:
            # 2. Register with Render Registry
            print("Registering nodes with Render Registry...")
            r0 = requests.post(f"{RENDER_URL}/register", json={
                "node_id": "hybrid-node-0",
                "addr": host0,
                "port": port0,
                "vram_available_mb": 15000,
                "vram_total_mb": 15000,
                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            }, timeout=10).json()

            r1 = requests.post(f"{RENDER_URL}/register", json={
                "node_id": "hybrid-node-1",
                "addr": host1,
                "port": port1,
                "vram_available_mb": 15000,
                "vram_total_mb": 15000,
                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            }, timeout=10).json()

            print(f"Registered node-0 -> [{r0['layer_start']}, {r0['layer_end']})")
            print(f"Registered node-1 -> [{r1['layer_start']}, {r1['layer_end']})")

            # 3. Load slices & start local node servers
            slice0 = load_layer_slice(MODEL_PATH, layer_start=r0["layer_start"], layer_end=r0["layer_end"], include_norm=False, include_lm_head=False, device=DEVICE)
            slice1 = load_layer_slice(MODEL_PATH, layer_start=r1["layer_start"], layer_end=r1["layer_end"], include_norm=True, include_lm_head=True, device=DEVICE)

            node1 = PipelineNode(slice1, is_first_node=False, is_last_node=True, listen_host="0.0.0.0", listen_port=9201)
            node0 = PipelineNode(slice0, is_first_node=True, is_last_node=False, next_node_host=host1, next_node_port=port1, listen_host="0.0.0.0", listen_port=9200)

            await node1.start()
            await asyncio.sleep(0.3)
            await node0.start()
            await asyncio.sleep(0.3)

            # 4. Invoke chat completion on Render orchestrator
            client = OpenAI(base_url=f"{RENDER_URL}/v1", api_key="shardflow")
            prompt = "Name three primary colors."
            print(f"User Prompt: '{prompt}'")

            start_t = time.time()
            response = client.chat.completions.create(
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.0,
                stream=False,
            )
            dur = time.time() - start_t
            content = response.choices[0].message.content
            print(f"Assistant Response: {content.strip()}")
            print(f"Latency: {dur:.2f}s")
            print("="*70 + "\n")

        finally:
            proc0.terminate()
            proc1.terminate()

    asyncio.run(_hybrid())


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    
    p1 = multiprocessing.Process(target=run_test_1)
    p1.start()
    p1.join()

    p2 = multiprocessing.Process(target=run_test_2)
    p2.start()
    p2.join()
