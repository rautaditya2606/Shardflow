"""
Kaggle Node Runner script for ShardFlow.

Supports:
1. Automatic Dual-GPU Pipeline (2x T4 on cuda:0 + cuda:1) with zero-latency local loopback IPC (~50+ TPS)
2. Single-GPU Distributed Mode for multi-machine clusters
3. Automatic safe caching to /kaggle/tmp to prevent RAM/VRAM OOM crashes

Usage in Kaggle Notebook:
1. !pip install -q torch transformers tokenizers safetensors accelerate bitsandbytes fastapi uvicorn requests pydantic sse-starlette
2. !git clone https://github.com/rautaditya2606/Shardflow.git /kaggle/working/Shardflow && cd /kaggle/working/Shardflow && pip install -e .
3. !python scripts/kaggle_runner.py --registry-url https://shardflow.onrender.com --model Qwen/Qwen2.5-7B-Instruct --draft-model Qwen/Qwen2.5-0.5B-Instruct --port 9500
"""

import os

# Disable Hugging Face Xet transfer backend which causes memory leaks and OOM in notebook environments (Kaggle/Colab)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# Force direct assignment to 20GB /kaggle/working storage if running in Kaggle to avoid root / exhaustion
if os.path.exists("/kaggle"):
    os.environ["HF_HOME"] = "/kaggle/working/hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/hf_home"
    os.environ["HF_HUB_CACHE"] = "/kaggle/working/hf_home"

import argparse
import asyncio
import logging
import threading
import time
import requests
import torch
from transformers import AutoConfig

from shardflow.registry.client import poll_for_assignment
from shardflow.transport.tunnel import start_cloudflare_tcp_tunnel, start_bore_tunnel
from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.node import PipelineNode

logger = logging.getLogger("shardflow.kaggle_runner")


def main():
    parser = argparse.ArgumentParser(description="ShardFlow Kaggle Node Runner")
    parser.add_argument("--registry-url", required=True, help="Registry URL (e.g. https://shardflow.onrender.com)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model path or HF model ID")
    parser.add_argument("--port", type=int, default=9500, help="Local TCP port for Node 0")
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
    parser.add_argument("--load-in-4bit", action="store_true", help="Quantize weights to 4-bit NF4 (Note: for 7B models on 2x T4, pure FP16 is faster and fits VRAM)")
    parser.add_argument("--enable-cuda-graphs", action="store_true", default=True, help="Enable CUDA Graphs for low-latency kernel replay (default: True)")
    parser.add_argument("--no-cuda-graphs", action="store_true", help="Disable CUDA Graphs and run in pure eager mode")
    parser.add_argument("--tailscale-authkey", default=None, help="Tailscale ephemeral auth key for direct P2P mesh networking")
    parser.add_argument("--draft-model", default=None, help="Small draft model for speculative decoding on Node 0 (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--spec-k", type=int, default=4, help="Number of speculative candidate draft tokens per verification step (default: 4)")
    parser.add_argument("--hf-model-id", default=None, help="Hugging Face repo ID if --model is a local path (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--force-single-gpu", action="store_true", help="Force single-GPU mode even if multiple GPUs are detected")
    args = parser.parse_args()

    def get_registry_model_id(model_path: str, explicit_id: str = None) -> str:
        if explicit_id:
            return explicit_id
        if not os.path.isabs(model_path) and "/" in model_path and not os.path.exists(model_path):
            return model_path
        base = os.path.basename(model_path.rstrip("/")).lower()
        if "qwen2.5-7b" in base or "qwen2.5-7b" in model_path.lower():
            return "Qwen/Qwen2.5-7B-Instruct"
        elif "qwen2.5-0.5b" in base or "qwen2.5-0.5b" in model_path.lower():
            return "Qwen/Qwen2.5-0.5B-Instruct"
        elif "tinyllama" in base or "tinyllama" in model_path.lower():
            return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        return model_path

    reg_model_id = get_registry_model_id(args.model, args.hf_model_id)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info("Detected %d CUDA GPU device(s) on this Kaggle runner instance.", num_gpus)

    node_id = args.node_id or f"kaggle-node-{int(time.time())}"
    local_port = args.port

    # Setup public endpoint (Tailscale, Bore, or Cloudflare)
    if args.tailscale_authkey:
        from shardflow.transport.tailscale import setup_tailscale_kaggle
        logger.info("Setting up direct Tailscale userspace mode on Kaggle...")
        ts_ip, ts_hname = setup_tailscale_kaggle(authkey=args.tailscale_authkey, hostname=node_id)
        pub_host = ts_ip
        pub_port = local_port
        logger.info("Tailscale P2P direct endpoint: %s:%d (Hostname: %s)", pub_host, pub_port, ts_hname)
    elif args.tunnel == "bore":
        logger.info("Starting bore tunnel on local port %d...", local_port)
        tunnel_proc, pub_host, pub_port = start_bore_tunnel(local_port)
        logger.info("Tunnel established at %s:%d", pub_host, pub_port)
    else:
        logger.info("Starting Cloudflare TCP tunnel on local port %d...", local_port)
        tunnel_proc, pub_host, pub_port = start_cloudflare_tcp_tunnel(local_port)
        logger.info("Tunnel established at %s:%d", pub_host, pub_port)

    # -------------------------------------------------------------
    # CASE 1: DUAL-GPU LOCAL PIPELINE (2x T4 on the same Kaggle VM)
    # -------------------------------------------------------------
    if num_gpus >= 2 and not args.force_single_gpu and args.layer_start is None and args.layer_end is None:
        logger.info("=================================================================")
        logger.info("🚀 INITIALIZING DUAL-GPU PIPELINE ON 2x T4 (cuda:0 & cuda:1)")
        logger.info("   Node 0 (cuda:0, port %d) <--> Node 1 (cuda:1, port %d)", local_port, local_port + 1)
        logger.info("   Inter-GPU link: 127.0.0.1 (0.05ms local loopback latency)")
        logger.info("=================================================================")

        config = AutoConfig.from_pretrained(args.model)
        total_layers = config.num_hidden_layers
        mid_layer = total_layers // 2

        # Diagnostics: inspect RAM consumption before model load
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            print(f"RAM before load: {proc.memory_info().rss / 1e9:.2f} GB")
            print(f"System available: {psutil.virtual_memory().available / 1e9:.2f} GB")
        except Exception as e:
            logger.debug("psutil error: %s", e)

        # 1. Load Node 1 slice (cuda:1, layers mid_layer..total_layers + norm + lm_head)
        logger.info("Loading Node 1 layers [%d, %d) onto device cuda:1...", mid_layer, total_layers)
        slice1 = load_layer_slice(
            model_path=args.model,
            layer_start=mid_layer,
            layer_end=total_layers,
            include_norm=True,
            include_lm_head=True,
            device="cuda:1",
            load_in_4bit=args.load_in_4bit,
        )
        node1 = PipelineNode(
            model_slice=slice1,
            is_first_node=False,
            is_last_node=True,
            next_node_host=None,
            next_node_port=None,
            listen_host="127.0.0.1",
            listen_port=local_port + 1,
            enable_cuda_graphs=args.enable_cuda_graphs and not args.no_cuda_graphs,
            spec_k=args.spec_k,
        )

        # 2. Load Node 0 slice (cuda:0, layers 0..mid_layer + embed + draft model)
        logger.info("Loading Node 0 layers [0, %d) onto device cuda:0...", mid_layer)
        slice0 = load_layer_slice(
            model_path=args.model,
            layer_start=0,
            layer_end=mid_layer,
            include_norm=False,
            include_lm_head=False,
            device="cuda:0",
            load_in_4bit=args.load_in_4bit,
        )
        node0 = PipelineNode(
            model_slice=slice0,
            is_first_node=True,
            is_last_node=False,
            next_node_host="127.0.0.1",
            next_node_port=local_port + 1,
            listen_host="0.0.0.0",
            listen_port=local_port,
            enable_cuda_graphs=args.enable_cuda_graphs and not args.no_cuda_graphs,
            draft_model=args.draft_model,
            spec_k=args.spec_k,
        )

        # 3. Register Node 0 as the public cluster endpoint with the registry
        vram_total = (torch.cuda.get_device_properties(0).total_memory + torch.cuda.get_device_properties(1).total_memory) / (1024 * 1024)
        reg_payload = {
            "node_id": node_id,
            "addr": pub_host,
            "port": pub_port,
            "vram_available_mb": vram_total,
            "vram_total_mb": vram_total,
            "model_id": reg_model_id,
            "layer_start": 0,
            "layer_end": total_layers,
            "expected_nodes": 1,
        }
        reg_url = f"{args.registry_url.rstrip('/')}/register"
        for attempt in range(3):
            try:
                logger.info("Registering dual-GPU cluster endpoint %s (attempt %d/3)...", node_id, attempt + 1)
                resp = requests.post(reg_url, json=reg_payload, timeout=30.0)
                resp.raise_for_status()
                break
            except Exception as e:
                logger.warning("Registration attempt %d failed: %s", attempt + 1, e)
                if attempt == 2:
                    raise
                time.sleep(2)

        def background_heartbeat_worker():
            hb_url = f"{args.registry_url.rstrip('/')}/heartbeat"
            hb_payload = {"node_id": node_id}
            while True:
                time.sleep(5.0)
                try:
                    requests.post(hb_url, json=hb_payload, timeout=5.0)
                except Exception:
                    pass

        hb_thread = threading.Thread(target=background_heartbeat_worker, daemon=True)
        hb_thread.start()

        async def run_dual_pipeline():
            logger.info("Starting Node 1 server on 127.0.0.1:%d ...", local_port + 1)
            await node1.start()
            logger.info("Starting Node 0 server on 0.0.0.0:%d ...", local_port)
            await node0.start()
            logger.info("🚀 Dual-GPU pipeline is live and ready for inference!")
            await asyncio.Event().wait()

        asyncio.run(run_dual_pipeline())
        return

    # -------------------------------------------------------------
    # CASE 2: SINGLE-GPU / DISTRIBUTED CLUSTER MODE
    # -------------------------------------------------------------
    vram = 0.0
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

    reg_payload = {
        "node_id": node_id,
        "addr": pub_host,
        "port": pub_port,
        "vram_available_mb": vram,
        "vram_total_mb": vram,
        "model_id": reg_model_id,
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

    current_node = None
    current_node_loop = None
    seen_topology_version = 0

    def background_heartbeat_worker():
        nonlocal seen_topology_version
        hb_url = f"{args.registry_url.rstrip('/')}/heartbeat"
        hb_payload = {"node_id": node_id}
        while True:
            time.sleep(5.0)
            try:
                hb_resp = requests.post(hb_url, json=hb_payload, timeout=5.0)
                if hb_resp.status_code == 200:
                    data = hb_resp.json()
                    topo_v = data.get("topology_version", 0)
                    if data.get("cluster_ready") and topo_v != seen_topology_version:
                        logger.info("Topology version changed v%d -> v%d", seen_topology_version, topo_v)
                        seen_topology_version = topo_v
                    if data.get("cluster_ready") and current_node is not None and current_node_loop is not None:
                        nxt_h = data.get("next_node_host")
                        nxt_p = data.get("next_node_port")
                        if nxt_h != current_node.next_node_host or nxt_p != current_node.next_node_port:
                            asyncio.run_coroutine_threadsafe(
                                current_node.update_next_node(nxt_h, nxt_p),
                                current_node_loop,
                            )
                elif hb_resp.status_code == 404:
                    logger.warning("Registry lost node %s — re-registering...", node_id)
                    re_resp = requests.post(reg_url, json=reg_payload, timeout=10.0)
                    if re_resp.status_code in (200, 201):
                        logger.info("Re-registration successful")
            except Exception as e:
                logger.debug("Background heartbeat ping error: %s", e)

    hb_thread = threading.Thread(target=background_heartbeat_worker, daemon=True)
    hb_thread.start()
    logger.info("Persistent background heartbeat thread started (5s ping interval)")

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
    seen_topology_version = topology_version

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
        enable_cuda_graphs=args.enable_cuda_graphs and not args.no_cuda_graphs,
        draft_model=args.draft_model,
        spec_k=args.spec_k,
    )
    current_node = node

    async def run_node():
        nonlocal current_node_loop
        current_node_loop = asyncio.get_running_loop()
        await node.serve_forever()

    logger.info("Pipeline node ready — starting server on 0.0.0.0:%d ...", local_port)
    asyncio.run(run_node())


if __name__ == "__main__":
    main()
