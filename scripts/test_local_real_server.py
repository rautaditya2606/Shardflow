"""
Realistic Local E2E Server Test:
1. Boots the real FastAPI Gateway & Registry Uvicorn web server on http://127.0.0.1:8000.
2. Spawns 2 real LLM GPU Nodes (TinyLlama layers 0..11 and 11..22) registering to http://127.0.0.1:8000.
3. Sends an actual HTTP POST request to http://127.0.0.1:8000/v1/chat/completions.
4. Verifies response text and cleans up all processes.
"""

import asyncio
import logging
import multiprocessing
import os
import sys
import time
import requests
import uvicorn


from shardflow.gateway.app import app as gateway_app
from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.node import PipelineNode
from shardflow.registry.client import poll_for_assignment
from shardflow.transport.connection import NodeClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_real_server")


def run_gateway_server(port: int = 8000):
    """Run Uvicorn FastAPI server process."""
    uvicorn.run(gateway_app, host="127.0.0.1", port=port, log_level="warning")


def run_node_process(model_path: str, listen_port: int, registry_url: str, node_id: str):
    """Run a pipeline node process — layers assigned by registry auto-partition."""
    logger.info("Node %s registering (auto-partition mode)...", node_id)

    # Register with no layer bounds — registry assigns via AutoPartitionEngine
    reg_payload = {
        "node_id": node_id,
        "addr": "127.0.0.1",
        "port": listen_port,
        "vram_available_mb": 4000.0,
        "vram_total_mb": 8000.0,
        "model_id": model_path,
    }
    resp = requests.post(f"{registry_url}/register", json=reg_payload, timeout=10.0)
    resp.raise_for_status()

    assignment = poll_for_assignment(registry_url, node_id, timeout=90.0)

    layer_start = assignment["layer_start"]
    layer_end = assignment["layer_end"]
    is_last = assignment["is_last_node"]
    next_host = assignment.get("next_node_host")
    next_port = assignment.get("next_node_port")

    logger.info("Node %s auto-assigned layers [%d, %d), is_last=%s", node_id, layer_start, layer_end, is_last)

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
        is_first_node=(layer_start == 0),
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
                requests.post(hb_url, json={"node_id": node_id}, timeout=3.0)
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

    logger.info("==================================================")
    logger.info("1. Starting FastAPI Gateway Server on %s...", registry_url)
    server_proc = multiprocessing.Process(target=run_gateway_server, args=(gateway_port,), daemon=True)
    server_proc.start()

    # Wait for server to boot
    for _ in range(30):
        try:
            res = requests.get(f"{registry_url}/health", timeout=1.0)
            if res.status_code == 200:
                logger.info("FastAPI Gateway is online!")
                break
        except Exception:
            time.sleep(0.2)

    logger.info("==========================================")
    logger.info("2. Spawning Node 1 (auto-partition, LAST candidate)...")
    node1_proc = multiprocessing.Process(
        target=run_node_process,
        args=(model_path, 9001, registry_url, "node-1"),
        daemon=True,
    )
    node1_proc.start()
    time.sleep(2.0)

    logger.info("==========================================")
    logger.info("3. Spawning Node 0 (auto-partition, FIRST candidate)...")
    node0_proc = multiprocessing.Process(
        target=run_node_process,
        args=(model_path, 9000, registry_url, "node-0"),
        daemon=True,
    )
    node0_proc.start()
    time.sleep(2.0)

    logger.info("==================================================")
    logger.info("4. Checking Registry Topology...")
    topo_res = requests.get(f"{registry_url}/topology").json()
    logger.info("Topology: %s nodes registered", topo_res["total_nodes"])

    logger.info("==================================================")
    logger.info("5. Sending HTTP POST /v1/chat/completions request...")
    payload = {
        "model": model_path,
        "messages": [{"role": "user", "content": "Explain gravity in one short sentence."}],
        "max_tokens": 30,
        "temperature": 0.0,
    }

    t0 = time.time()
    resp = requests.post(f"{registry_url}/v1/chat/completions", json=payload, timeout=30.0)
    elapsed = time.time() - t0

    logger.info("HTTP Status: %d", resp.status_code)
    data = resp.json()
    logger.info("Full JSON Response:\n%s", data)

    if resp.status_code == 200:
        content = data["choices"][0]["message"]["content"]
        finish_reason = data["choices"][0]["finish_reason"]
        logger.info("==================================================")
        logger.info("SUCCESS! Generated Text:\n'%s'", content)
        logger.info("Finish Reason: %s (Time: %.2fs)", finish_reason, elapsed)
        logger.info("==================================================")
    else:
        logger.error("FAILED! Response: %s", data)

    # Cleanup
    node0_proc.terminate()
    node1_proc.terminate()
    server_proc.terminate()


if __name__ == "__main__":
    main()
