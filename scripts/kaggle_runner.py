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
    parser.add_argument("--expected-nodes", type=int, default=None, help="Expected total cluster nodes count (default: 2 for single-GPU distributed mode)")
    parser.add_argument("--reset-registry", action="store_true", help="Reset remote registry state before registering")
    parser.add_argument("--layer-start", type=int, default=None, help="Explicit layer start (optional)")
    parser.add_argument("--layer-end", type=int, default=None, help="Explicit layer end (optional)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Quantize weights to 4-bit NF4 (Note: for 7B models on 2x T4, pure FP16 is faster and fits VRAM)")
    parser.add_argument("--enable-cuda-graphs", action="store_true", default=True, help="Enable CUDA Graphs for low-latency kernel replay (default: True)")
    parser.add_argument("--no-cuda-graphs", action="store_true", help="Disable CUDA Graphs and run in pure eager mode")
    parser.add_argument("--tailscale-authkey", default=None, help="Tailscale ephemeral auth key for direct P2P mesh networking")
    parser.add_argument("--draft-model", default=None, help="Small draft model for speculative decoding on Node 0 (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--spec-k", type=int, default=4, help="Number of speculative candidate draft tokens per verification step (default: 4)")
    parser.add_argument("--hf-model-id", default=None, help="Hugging Face repo ID if --model is a local path (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--next-host", default=None, help="Next node public host (optional, for manual static topology)")
    parser.add_argument("--next-port", type=int, default=None, help="Next node public port (optional, for manual static topology)")
    parser.add_argument("--is-last", action="store_true", help="Explicitly mark this node as the final/terminal node")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Model and KV cache precision (default: float16 for maximum compatibility across P100/T4)")
    parser.add_argument("--force-single-gpu", action="store_true", help="Force single-GPU mode even if multiple GPUs are detected")
    args = parser.parse_args()

    if args.reset_registry:
        try:
            reset_url = f"{args.registry_url.rstrip('/')}/reset"
            logger.info("Resetting registry state at %s...", reset_url)
            requests.post(reset_url, timeout=10.0)
            logger.info("Registry state reset successfully.")
        except Exception as e:
            logger.warning("Could not reset registry: %s", e)

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

        import subprocess, sys, socket
        # 1. Start Node 1 subprocess (isolated CUDA context on GPU 1)
        node1_env = os.environ.copy()
        node1_env["CUDA_VISIBLE_DEVICES"] = "1"
        node1_cmd = [
            sys.executable, "-m", "shardflow.node.node",
            "--model", args.model,
            "--layer-start", str(mid_layer),
            "--layer-end", str(total_layers),
            "--host", "127.0.0.1",
            "--port", str(local_port + 1),
            "--device", "cuda",
        ]
        if args.no_cuda_graphs:
            node1_cmd.append("--no-cuda-graphs")

        node1_log_path = "/tmp/node1.log"
        node1_write_file = open(node1_log_path, "w")
        node1_read_file = open(node1_log_path, "r")
        logger.info("Spawning Node 1 isolated subprocess on GPU 1 (port %d, log=%s)...", local_port + 1, node1_log_path)
        node1_proc = subprocess.Popen(node1_cmd, env=node1_env, stdout=node1_write_file, stderr=subprocess.STDOUT)

        # 2. Wait until Node 1 is listening on 127.0.0.1:9501
        logger.info("Waiting for Node 1 to initialize and listen on 127.0.0.1:%d...", local_port + 1)
        for _ in range(120):
            line = node1_read_file.readline()
            while line:
                print(f"[Node 1] {line.rstrip()}", flush=True)
                line = node1_read_file.readline()

            if node1_proc.poll() is not None:
                # Flush remaining output
                line = node1_read_file.readline()
                while line:
                    print(f"[Node 1] {line.rstrip()}", flush=True)
                    line = node1_read_file.readline()
                logger.error("Node 1 exited: returncode=%s", node1_proc.returncode)
                raise RuntimeError(f"Node 1 process exited unexpectedly with returncode {node1_proc.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", local_port + 1), timeout=1.0):
                    logger.info("✅ Node 1 is online and listening!")
                    break
            except Exception:
                time.sleep(1.0)
        else:
            raise TimeoutError("Node 1 failed to start within 120 seconds")

        # 3. Start Node 0 subprocess (isolated CUDA context on GPU 0)
        node0_env = os.environ.copy()
        node0_env["CUDA_VISIBLE_DEVICES"] = "0"
        node0_cmd = [
            sys.executable, "-m", "shardflow.node.node",
            "--model", args.model,
            "--layer-start", "0",
            "--layer-end", str(mid_layer),
            "--next-host", "127.0.0.1",
            "--next-port", str(local_port + 1),
            "--host", "0.0.0.0",
            "--port", str(local_port),
            "--public-host", pub_host,
            "--public-port", str(pub_port),
            "--registry-url", args.registry_url,
            "--reg-layer-start", "0",
            "--reg-layer-end", str(total_layers),
            "--expected-nodes", "1",
            "--hf-model-id", reg_model_id,
            "--device", "cuda",
        ]
        if args.draft_model:
            node0_cmd.extend(["--draft-model", args.draft_model, "--spec-k", str(args.spec_k)])
        if args.no_cuda_graphs:
            node0_cmd.append("--no-cuda-graphs")

        node0_log_path = "/tmp/node0.log"
        node0_write_file = open(node0_log_path, "w")
        node0_read_file = open(node0_log_path, "r")
        logger.info("Spawning Node 0 isolated subprocess on GPU 0 (port %d, log=%s)...", local_port, node0_log_path)
        node0_proc = subprocess.Popen(node0_cmd, env=node0_env, stdout=node0_write_file, stderr=subprocess.STDOUT)

        try:
            logger.info("🚀 Dual-GPU pipeline is live and ready for inference!")
            while True:
                line0 = node0_read_file.readline()
                while line0:
                    print(f"[Node 0] {line0.rstrip()}", flush=True)
                    line0 = node0_read_file.readline()

                line1 = node1_read_file.readline()
                while line1:
                    print(f"[Node 1] {line1.rstrip()}", flush=True)
                    line1 = node1_read_file.readline()

                if node0_proc.poll() is not None:
                    logger.error("Node 0 exited: returncode=%s", node0_proc.returncode)
                    break
                if node1_proc.poll() is not None:
                    logger.error("Node 1 exited: returncode=%s", node1_proc.returncode)
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Shutting down dual-GPU pipeline...")
        finally:
            node1_proc.terminate()
            node0_proc.terminate()
            node1_write_file.close()
            node1_read_file.close()
            node0_write_file.close()
            node0_read_file.close()
        return

    # -------------------------------------------------------------
    # CASE 2: SINGLE-GPU / DISTRIBUTED CLUSTER MODE
    # -------------------------------------------------------------
    vram = 0.0
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

    expected_nodes_val = args.expected_nodes
    if expected_nodes_val is None and (args.layer_start is None or args.layer_end is None):
        expected_nodes_val = 2

    reg_payload = {
        "node_id": node_id,
        "addr": pub_host,
        "port": pub_port,
        "vram_available_mb": vram,
        "vram_total_mb": vram,
        "model_id": reg_model_id,
    }
    if expected_nodes_val is not None:
        reg_payload["expected_nodes"] = expected_nodes_val
    if args.layer_start is not None:
        reg_payload["layer_start"] = args.layer_start
    if args.layer_end is not None:
        reg_payload["layer_end"] = args.layer_end

    reg_url = f"{args.registry_url.rstrip('/')}/register"
    for attempt in range(3):
        try:
            logger.info("Registering node %s (attempt %d/3, expected_nodes=%s)...", node_id, attempt + 1, expected_nodes_val)
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

    if args.layer_start is not None and args.layer_end is not None and (getattr(args, "is_last", False) or getattr(args, "next_host", None) is not None):
        assignment = {
            "layer_start": args.layer_start,
            "layer_end": args.layer_end,
            "is_first_node": args.layer_start == 0,
            "is_last_node": getattr(args, "is_last", False),
            "next_node_host": getattr(args, "next_host", None),
            "next_node_port": getattr(args, "next_port", None),
            "topology_version": 0,
        }
    else:
        logger.info("Waiting for final cluster assignment before loading model weights...")
        assignment = poll_for_assignment(args.registry_url, node_id, timeout=180.0)

    config = AutoConfig.from_pretrained(args.model)
    total_layers = config.num_hidden_layers

    layer_start = args.layer_start if args.layer_start is not None else assignment["layer_start"]
    layer_end = args.layer_end if args.layer_end is not None else assignment["layer_end"]
    is_first = (layer_start == 0)
    is_last = (layer_end >= total_layers) or getattr(args, "is_last", False)
    next_host = getattr(args, "next_host", None) or assignment.get("next_node_host")
    next_port = getattr(args, "next_port", None) or assignment.get("next_node_port")
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
        dtype=getattr(torch, args.dtype) if args.dtype else torch.float16,
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
