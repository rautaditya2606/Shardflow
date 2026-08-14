#!/usr/bin/env python3
"""
ShardFlow Kaggle Remote Node 1 Runner (Kaggle Instance B).

Runs Node 1 (Layers 14..28 + LM Head) with aiohttp HTTP server on port 9502,
and exposes it over a Cloudflare Quick Tunnel (HTTPS).
"""

import os
import sys
import time
import re
import shutil
import socket
import subprocess
import threading
import requests
from pathlib import Path

# Add project root to sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

if os.path.exists("/kaggle"):
    os.environ["HF_HOME"] = "/kaggle/working/hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/hf_home"
    os.environ["HF_HUB_CACHE"] = "/kaggle/working/hf_home"


def ensure_cloudflared() -> str:
    """Find or download cloudflared binary."""
    cf_path = shutil.which("cloudflared")
    if cf_path:
        return cf_path

    local_cf = Path(repo_root) / "cloudflared"
    if local_cf.exists() and os.access(local_cf, os.X_OK):
        return str(local_cf)

    print("Downloading cloudflared binary...", flush=True)
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    r = requests.get(url, stream=True, timeout=30.0)
    with open(local_cf, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
    os.chmod(local_cf, 0o755)
    print("✅ cloudflared downloaded.", flush=True)
def kill_ports(ports: list[int]):
    """Kill any zombie processes from prior runs occupying our ports."""
    import signal
    for port in ports:
        try:
            subprocess.run(["fuser", "-k", "-n", "tcp", str(port)], capture_output=True, timeout=1.0)
        except Exception:
            pass
        try:
            res = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True, timeout=1.0)
            if res.returncode == 0 and res.stdout.strip():
                for pid_str in res.stdout.strip().split():
                    try:
                        pid = int(pid_str)
                        if pid != os.getpid():
                            os.kill(pid, signal.SIGKILL)
                    except Exception:
                        pass
        except Exception:
            pass
    time.sleep(0.5)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ShardFlow Node 1 (Kaggle B Remote Runner)")
    parser.add_argument("--model", default="/kaggle/working/models/Qwen2.5-7B-Instruct", help="Model path or HF ID")
    parser.add_argument("--layer-start", type=int, default=None, help="Starting layer index (default: half of total layers)")
    parser.add_argument("--layer-end", type=int, default=None, help="Ending layer index (default: total layers)")
    parser.add_argument("--4bit", action="store_true", help="Enable 4-bit NF4 loading")
    parser.add_argument("--port", type=int, default=9501, help="TCP port (default: 9501)")
    parser.add_argument("--http-port", type=int, default=9502, help="HTTP server port for tunnel (default: 9502)")
    parser.add_argument("--device", default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--spec-k", type=int, default=12, help="Speculative candidate tokens (default: 12)")
    parser.add_argument("--no-cuda-graphs", action="store_true", default=True, help="Disable CUDA Graphs in eager mode")
    parser.add_argument("--no-cloudflared", action="store_true", help="Skip cloudflared tunnel (for local loopback testing)")
    args = parser.parse_args()

    # Clean up any lingering processes from previous notebook runs
    kill_ports([args.port, args.http_port])

    model_path = args.model if os.path.exists(args.model) else "Qwen/Qwen2.5-7B-Instruct"

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    total_layers = getattr(config, "num_hidden_layers", 28)

    layer_start = args.layer_start if args.layer_start is not None else (total_layers // 2)
    layer_end = args.layer_end if args.layer_end is not None else total_layers

    print("=" * 70, flush=True)
    print("🚀 SHARDFLOW REMOTE NODE 1 (KAGGLE INSTANCE B)", flush=True)
    print(f"Model Shard:   {model_path} (Layers {layer_start}..{layer_end} + LM Head)")
    print(f"HTTP Port:     {args.http_port}")
    print(f"Device:        {args.device}")
    print("=" * 70, flush=True)

    # 1. Spawn Node 1 process
    print(f"\n[1/3] Loading Model Shard on {args.device} and starting HTTP server...", flush=True)
    node1_cmd = [
        sys.executable, "-m", "shardflow.node.node",
        "--model", model_path,
        "--layer-start", str(layer_start),
        "--layer-end", str(layer_end),
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--http-port", str(args.http_port),
        "--device", args.device,
        "--spec-k", str(args.spec_k),
    ]
    if getattr(args, "4bit", False) or "nf4" in model_path.lower() or "4bit" in model_path.lower():
        node1_cmd.append("--4bit")
    if args.no_cuda_graphs:
        node1_cmd.append("--no-cuda-graphs")

    node1_proc = subprocess.Popen(node1_cmd)

    # Wait for Node 1 HTTP server /health to be ready
    health_url = f"http://127.0.0.1:{args.http_port}/health"
    print("Waiting for weights to load and HTTP /health to respond...", flush=True)
    for _ in range(120):
        if node1_proc.poll() is not None:
            with open("/tmp/node1_remote.log", "r") as f:
                logs = f.read()
            raise RuntimeError(f"Node 1 failed to start (exit code {node1_proc.returncode}). Log:\n{logs}")
        try:
            r = requests.get(health_url, timeout=1.0)
            if r.status_code == 200:
                print("✅ Node 1 HTTP server is online and healthy!", flush=True)
                break
        except Exception:
            time.sleep(1.5)
    else:
        raise TimeoutError("Node 1 startup timed out after 120s")

    cf_proc = None
    tunnel_url = None

    if not args.no_cloudflared:
        # 2. Start Cloudflare Quick Tunnel
        print(f"\n[2/3] Starting Cloudflare Quick Tunnel on http://127.0.0.1:{args.http_port}...", flush=True)
        cf_bin = ensure_cloudflared()
        cf_cmd = [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{args.http_port}"]
        cf_proc = subprocess.Popen(cf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def read_output(pipe):
            nonlocal tunnel_url
            for line in pipe:
                match = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
                if match and tunnel_url is None:
                    tunnel_url = match.group(0)

        threading.Thread(target=read_output, args=(cf_proc.stdout,), daemon=True).start()
        threading.Thread(target=read_output, args=(cf_proc.stderr,), daemon=True).start()

        # Wait for tunnel URL to appear
        for _ in range(40):
            if tunnel_url:
                break
            time.sleep(1.0)

        if not tunnel_url:
            raise RuntimeError("Cloudflare tunnel failed to produce a public URL after 40s.")

        # Verify tunnel endpoint is healthy before presenting URL
        print("Verifying public tunnel endpoint...", flush=True)
        for _ in range(15):
            try:
                r = requests.get(f"{tunnel_url}/health", timeout=3.0)
                if r.status_code == 200:
                    print("✅ Public tunnel is live and routing traffic!", flush=True)
                    break
            except Exception:
                time.sleep(1.0)
    else:
        tunnel_url = f"http://127.0.0.1:{args.http_port}"

    # 3. Print Banner
    print("\n" + "=" * 70, flush=True)
    print("🌟 NODE 1 IS READY FOR INFERENCE!", flush=True)
    print("Copy and paste the following URL into Kaggle Instance A:", flush=True)
    print(f"\n   --node1-url {tunnel_url}\n", flush=True)
    print("Example command on Kaggle A:")
    print(f"   !python scripts/kaggle_node0.py --node1-url {tunnel_url}")
    print("=" * 70 + "\n", flush=True)

    # 4. Keepalive loop
    try:
        while True:
            time.sleep(10)
            if node1_proc.poll() is not None:
                print("⚠️ Node 1 process terminated unexpectedly!", flush=True)
                break
    except KeyboardInterrupt:
        print("\nStopping Node 1...", flush=True)
    finally:
        if node1_proc:
            node1_proc.terminate()
        if cf_proc:
            cf_proc.terminate()
        node1_log.close()
        print("Node 1 shutdown complete.")


if __name__ == "__main__":
    main()
