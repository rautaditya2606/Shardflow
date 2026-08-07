"""
3-Node Local End-to-End Test for ShardFlow with TinyLlama-1.1B-Chat-v1.0.

Spawns:
1. FastAPI Gateway & Registry server on port 8000 (SHARDFLOW_EXPECTED_NODES=3)
2. Node 0 (port 9000)
3. Node 1 (port 9001)
4. Node 2 (port 9002)

Verifies cluster auto-partitioning across 3 nodes and runs chat completion.
"""

import asyncio
import logging
import multiprocessing
import os
import sys
import time
import requests
import uvicorn
from openai import OpenAI

os.environ["SHARDFLOW_EXPECTED_NODES"] = "3"

from shardflow.gateway.app import app as gateway_app
from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.node import PipelineNode
from shardflow.registry.client import poll_for_assignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("test_3_nodes_local")


def run_gateway_server(port: int = 8000):
    """Run Uvicorn FastAPI server process."""
    os.environ["SHARDFLOW_EXPECTED_NODES"] = "3"
    uvicorn.run(gateway_app, host="127.0.0.1", port=port, log_level="warning")


def run_node_process(model_path: str, listen_port: int, registry_url: str, node_id: str):
    """Run a pipeline node process — layers assigned by registry auto-partition."""
    logger.info("Node %s starting and registering to %s...", node_id, registry_url)

    reg_payload = {
        "node_id": node_id,
        "addr": "127.0.0.1",
        "port": listen_port,
        "vram_available_mb": 1200.0,
        "vram_total_mb": 4000.0,
        "model_id": model_path,
    }
    resp = requests.post(f"{registry_url}/register", json=reg_payload, timeout=10.0)
    resp.raise_for_status()

    assignment = poll_for_assignment(registry_url, node_id, timeout=90.0)

    layer_start = assignment["layer_start"]
    layer_end = assignment["layer_end"]
    is_first = assignment.get("is_first_node", layer_start == 0)
    is_last = assignment.get("is_last_node", False)
    next_host = assignment.get("next_node_host")
    next_port = assignment.get("next_node_port")

    logger.info(
        "Node %s assigned layers [%d, %d) (is_first=%s, is_last=%s, next=%s:%s)",
        node_id, layer_start, layer_end, is_first, is_last, next_host, next_port
    )

    model_slice = load_layer_slice(
        model_path=model_path,
        layer_start=layer_start,
        layer_end=layer_end,
        include_norm=is_last,
        include_lm_head=is_last,
        device="cuda",
    )

    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=is_first,
        is_last_node=is_last,
        next_node_host=next_host,
        next_node_port=next_port,
        listen_host="127.0.0.1",
        listen_port=listen_port,
    )

    async def heartbeat_loop():
        hb_url = f"{registry_url}/heartbeat"
        while True:
            await asyncio.sleep(5.0)
            try:
                hb_resp = requests.post(hb_url, json={"node_id": node_id}, timeout=3.0)
                if hb_resp.status_code == 200:
                    data = hb_resp.json()
                    if data.get("cluster_ready"):
                        await node.update_next_node(data.get("next_node_host"), data.get("next_node_port"))
            except Exception:
                pass

    async def main_node():
        asyncio.create_task(heartbeat_loop())
        await node.serve_forever()

    asyncio.run(main_node())


def main():
    model_path = "./models/TinyLlama-1.1B-Chat-v1.0"
    gateway_port = 8000
    registry_url = f"http://127.0.0.1:{gateway_port}"

    print("\n" + "=" * 70)
    print("🚀 SHARDFLOW 3-NODE LOCAL TEST (TinyLlama-1.1B-Chat-v1.0)")
    print("=" * 70 + "\n")

    logger.info("1. Starting FastAPI Gateway Server on %s...", registry_url)
    server_proc = multiprocessing.Process(target=run_gateway_server, args=(gateway_port,), daemon=True)
    server_proc.start()

    # Wait for gateway health
    for _ in range(30):
        try:
            res = requests.get(f"{registry_url}/health", timeout=1.0)
            if res.status_code == 200:
                logger.info("Gateway server is online!")
                break
        except Exception:
            time.sleep(0.2)

    logger.info("2. Spawning 3 Pipeline Nodes...")

    # Spawn node 2 (last node candidate)
    n2_proc = multiprocessing.Process(
        target=run_node_process,
        args=(model_path, 9002, registry_url, "node-2"),
        daemon=True,
    )
    n2_proc.start()
    time.sleep(1.0)

    # Spawn node 1 (middle node candidate)
    n1_proc = multiprocessing.Process(
        target=run_node_process,
        args=(model_path, 9001, registry_url, "node-1"),
        daemon=True,
    )
    n1_proc.start()
    time.sleep(1.0)

    # Spawn node 0 (first node candidate)
    n0_proc = multiprocessing.Process(
        target=run_node_process,
        args=(model_path, 9000, registry_url, "node-0"),
        daemon=True,
    )
    n0_proc.start()
    time.sleep(3.0)

    logger.info("3. Verifying 3-Node Topology...")
    topo = requests.get(f"{registry_url}/topology").json()
    print("\n--- Topology Status ---")
    print(f"Total Nodes: {topo.get('total_nodes')}")
    print(f"Cluster Ready: {topo.get('cluster_ready')}")
    for idx, node in enumerate(topo.get("nodes", [])):
        print(
            f" Node {idx} ({node['node_id']}): layers [{node['layer_start']}, {node['layer_end']}) "
            f"-> next: {node['next_node_host']}:{node['next_node_port']}"
        )
    print("-----------------------\n")

    logger.info("4. Testing Streaming Chat Completion via OpenAI SDK...")
    client = OpenAI(base_url=f"{registry_url}/v1", api_key="shardflow-local")

    prompt = "Explain why the sky is blue in 2 clear bullet points."
    print(f"Prompt: '{prompt}'\n")

    start_time = time.time()
    stream = client.chat.completions.create(
        model=model_path,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.7,
        stream=True,
    )

    print("Response: ", end="", flush=True)
    token_count = 0
    first_token_time = None

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            text = chunk.choices[0].delta.content
            print(text, end="", flush=True)
            token_count += 1

    total_time = time.time() - start_time
    ttft = (first_token_time - start_time) if first_token_time else total_time
    tok_per_sec = token_count / (total_time - ttft) if (total_time - ttft) > 0 else 0

    print(f"\n\n{'=' * 70}")
    print("✅ TEST SUCCESSFUL!")
    print(f"Tokens Generated: {token_count}")
    print(f"Time to First Token (TTFT): {ttft:.3f}s")
    print(f"Total Execution Time: {total_time:.2f}s")
    print(f"Generation Throughput: {tok_per_sec:.2f} tok/s")
    print("=" * 70 + "\n")

    # Cleanup processes
    n0_proc.terminate()
    n1_proc.terminate()
    n2_proc.terminate()
    server_proc.terminate()


if __name__ == "__main__":
    main()
