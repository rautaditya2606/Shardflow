"""
Colab Node Runner script for ShardFlow.

Usage in Google Colab:
1. !pip install -q torch transformers tokenizers safetensors accelerate fastapi uvicorn requests pydantic sse-starlette
2. !git clone https://github.com/adityaraut/Shardflow.git /content/Shardflow && cd /content/Shardflow && pip install -e .
3. !python scripts/colab_runner.py --registry-url https://your-registry-url.onrender.com --model meta-llama/Meta-Llama-3-8B
"""

import argparse
import asyncio
import logging
import time
import requests
import torch

from shardflow.transport.tunnel import start_cloudflare_tcp_tunnel, start_bore_tunnel
from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.node import PipelineNode

logger = logging.getLogger("shardflow.colab_runner")


def main():
    parser = argparse.ArgumentParser(description="ShardFlow Colab Node Runner")
    parser.add_argument("--registry-url", required=True, help="Registry URL (e.g. https://shardflow-v0-1-0.onrender.com)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="Model path or HF model ID")
    parser.add_argument("--port", type=int, default=9500, help="Local TCP port")
    parser.add_argument("--tunnel", choices=["bore", "cloudflare"], default="bore", help="Tunnel backend (default: bore)")
    parser.add_argument("--node-id", default=None, help="Unique node identifier")
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

    logger.info("Registering node %s with registry %s...", node_id, args.registry_url)
    reg_payload = {
        "node_id": node_id,
        "addr": pub_host,
        "port": pub_port,
        "vram_available_mb": vram,
        "vram_total_mb": vram,
        "model_id": args.model,
    }

    resp = requests.post(f"{args.registry_url.rstrip('/')}/register", json=reg_payload, timeout=15.0)
    resp.raise_for_status()
    assignment = resp.json()

    layer_start = assignment["layer_start"]
    layer_end = assignment["layer_end"]
    is_first = assignment["is_first_node"]
    is_last = assignment["is_last_node"]
    next_host = assignment.get("next_node_host")
    next_port = assignment.get("next_node_port")

    logger.info(
        "Successfully registered! Assigned layers [%d, %d) (is_first=%s, is_last=%s)",
        layer_start, layer_end, is_first, is_last
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
    )

    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=is_first,
        is_last_node=is_last,
        next_node_host=next_host,
        next_node_port=next_port,
        listen_host="0.0.0.0",
        listen_port=local_port,
    )

    logger.info("Pipeline node running...")
    asyncio.run(node.serve_forever())


if __name__ == "__main__":
    main()
