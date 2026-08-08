"""
Colab Node Runner script for ShardFlow.

Usage in Google Colab:
1. !pip install -q torch transformers tokenizers safetensors accelerate fastapi uvicorn requests pydantic sse-starlette
2. !git clone https://github.com/rautaditya2606/Shardflow.git /content/Shardflow && cd /content/Shardflow && pip install -e .
3. !python scripts/colab_runner.py --registry-url https://shardflow.onrender.com --model Qwen/Qwen2.5-7B-Instruct --node-id colab-node-1 --port 9500
"""

import argparse
import asyncio
import logging
import time
import requests
import torch

from shardflow.registry.client import poll_for_assignment
from shardflow.transport.tunnel import start_cloudflare_tcp_tunnel, start_bore_tunnel
from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.node import PipelineNode

logger = logging.getLogger("shardflow.colab_runner")


def main():
    parser = argparse.ArgumentParser(description="ShardFlow Colab Node Runner")
    parser.add_argument("--registry-url", required=True, help="Registry URL (e.g. https://shardflow.onrender.com)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model path or HF model ID")
    parser.add_argument("--port", type=int, default=9500, help="Local TCP port")
    parser.add_argument(
        "--tunnel",
        choices=["bore", "cloudflare"],
        default="bore",
        help="Tunnel backend (default: bore — bore.pub raw TCP proxy for high performance binary tensor transfer)",
    )
    parser.add_argument("--node-id", default=None, help="Unique node identifier")
    parser.add_argument("--expected-nodes", type=int, default=None, help="Expected total cluster nodes count (optional)")
    parser.add_argument("--layer-start", type=int, default=None, help="Explicit layer start (optional)")
    parser.add_argument("--layer-end", type=int, default=None, help="Explicit layer end (optional)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Quantize weights to 4-bit NF4 via bitsandbytes to save VRAM")
    parser.add_argument("--next-host", default=None, help="Explicit next node host (optional)")
    parser.add_argument("--next-port", type=int, default=None, help="Explicit next node port (optional)")
    parser.add_argument("--is-last", action="store_true", help="Explicitly mark as last node (optional)")
    parser.add_argument("--no-cuda-graphs", action="store_true", help="Disable CUDA Graphs and run in pure eager mode")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    node_id = args.node_id or f"colab-node-{int(time.time())}"
    local_port = args.port

    if args.tunnel == "bore":
        logger.info("Starting bore tunnel on local port %d...", local_port)
        tunnel_proc, pub_host, pub_port = start_bore_tunnel(local_port)
    else:
        logger.info("Starting Cloudflare TCP tunnel on local port %d...", local_port)
        tunnel_proc, pub_host, pub_port = start_cloudflare_tcp_tunnel(local_port)

    logger.info("Tunnel established at %s:%d", pub_host, pub_port)

    vram = 0.0
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

    reg_payload = {
        "node_id": node_id,
        "addr": pub_host,
        "port": pub_port,
        "vram_available_mb": vram,
        "vram_total_mb": vram,
        "model_id": args.model,
    }
    if args.expected_nodes is not None:
        reg_payload["expected_nodes"] = args.expected_nodes
    if args.layer_start is not None:
        reg_payload["layer_start"] = args.layer_start
    if args.layer_end is not None:
        reg_payload["layer_end"] = args.layer_end

    reg_url = f"{args.registry_url.rstrip('/')}/register"
    for attempt in range(3):
        try:
            logger.info("Registering node %s (attempt %d/3)...", node_id, attempt + 1)
            resp = requests.post(reg_url, json=reg_payload, timeout=30.0)
            resp.raise_for_status()
            break
        except Exception as e:
            logger.warning("Registration attempt %d failed: %s", attempt + 1, e)
            if attempt == 2:
                raise
            time.sleep(2)

    if args.layer_start is not None and args.layer_end is not None and (args.is_last or args.next_host is not None):
        assignment = {
            "layer_start": args.layer_start,
            "layer_end": args.layer_end,
            "is_first_node": args.layer_start == 0,
            "is_last_node": args.is_last,
            "next_node_host": args.next_host,
            "next_node_port": args.next_port,
            "topology_version": 0,
        }
    else:
        logger.info("Waiting for final cluster assignment before loading model weights...")
        assignment = poll_for_assignment(args.registry_url, node_id, timeout=180.0)

    layer_start = assignment["layer_start"]
    layer_end = assignment["layer_end"]
    total_layers = assignment.get("total_model_layers")
    is_first = assignment.get("is_first_node", layer_start == 0)
    is_last = assignment.get("is_last_node", False)
    next_host = assignment.get("next_node_host")
    next_port = assignment.get("next_node_port")
    topology_version = assignment.get("topology_version", 0)

    logger.info(
        "Cluster ready (topology v%d)! Assigned layers [%d, %d)%s (is_first=%s, is_last=%s)",
        topology_version,
        layer_start,
        layer_end,
        f" of {total_layers}" if total_layers else "",
        is_first,
        is_last,
    )
    if next_host:
        logger.info("Next node routing target: %s:%d", next_host, next_port)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading layer slice [%d, %d) onto device %s...", layer_start, layer_end, device)

    model_slice = load_layer_slice(
        model_path=args.model,
        layer_start=layer_start,
        layer_end=layer_end,
        include_norm=is_last,
        include_lm_head=is_last,
        device=device,
        load_in_4bit=args.load_in_4bit,
    )

    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=is_first,
        is_last_node=is_last,
        next_node_host=next_host,
        next_node_port=next_port,
        listen_host="0.0.0.0",
        listen_port=local_port,
        enable_cuda_graphs=not args.no_cuda_graphs,
    )

    seen_topology_version = topology_version

    async def heartbeat_loop():
        """
        Keep node alive and apply routing updates when topology_version changes.
        Re-register automatically if Render restarts and loses in-memory state.
        """
        nonlocal seen_topology_version
        hb_url = f"{args.registry_url.rstrip('/')}/heartbeat"
        hb_payload = {"node_id": node_id}
        while True:
            await asyncio.sleep(10.0)
            try:
                hb_resp = await asyncio.to_thread(
                    requests.post, hb_url, json=hb_payload, timeout=5.0
                )
                if hb_resp.status_code == 200:
                    data = hb_resp.json()
                    topo_v = data.get("topology_version", 0)
                    if data.get("cluster_ready") and topo_v != seen_topology_version:
                        logger.info("Topology changed v%d -> v%d, updating routing...", seen_topology_version, topo_v)
                        seen_topology_version = topo_v
                    if data.get("cluster_ready"):
                        await node.update_next_node(data.get("next_node_host"), data.get("next_node_port"))
                elif hb_resp.status_code == 404:
                    logger.warning("Registry lost node %s — re-registering...", node_id)
                    re_resp = await asyncio.to_thread(
                        requests.post, reg_url, json=reg_payload, timeout=10.0
                    )
                    if re_resp.status_code in (200, 201):
                        data = re_resp.json()
                        seen_topology_version = data.get("topology_version", seen_topology_version)
                        if data.get("cluster_ready"):
                            await node.update_next_node(data.get("next_node_host"), data.get("next_node_port"))
                        logger.info("Re-registration successful")
            except Exception as e:
                logger.debug("Heartbeat ping error: %s", e)

    async def run_node():
        asyncio.create_task(heartbeat_loop())
        await node.serve_forever()

    logger.info("Pipeline node running with background heartbeat & auto-reregistration...")
    asyncio.run(run_node())


if __name__ == "__main__":
    main()
