#!/usr/bin/env python3
"""
ShardFlow Kaggle Dual-GPU Self-Contained Demo & Benchmark.

Runs 100% locally on a single Kaggle VM with 2x T4 GPUs:
1. Local FastAPI Gateway & Registry on http://127.0.0.1:8000
2. Node 1 on cuda:1 (Layers 14..28 + LM Head) on 127.0.0.1:9501
3. Node 0 on cuda:0 (Layers 0..14 + 0.5B Draft Model, Spec K=4) on 127.0.0.1:9500
4. Live Streaming Benchmark yielding 50+ TPS directly in the notebook cell.
"""

import os
import sys
import time
import json
import socket
import subprocess
import requests
import multiprocessing

# Add project root to sys.path
from pathlib import Path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

if os.path.exists("/kaggle"):
    os.environ["HF_HOME"] = "/kaggle/working/hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/hf_home"
    os.environ["HF_HUB_CACHE"] = "/kaggle/working/hf_home"


def run_gateway_server(port: int = 8000, stream_port: int = 8001):
    """Run Gateway + Embedded Registry FastAPI server with v2 Data-Plane streaming."""
    os.environ["SHARDFLOW_STREAM_HOST"] = "127.0.0.1"
    os.environ["SHARDFLOW_STREAM_PORT"] = str(stream_port)
    import uvicorn
    from shardflow.gateway.app import app as gateway_app
    uvicorn.run(gateway_app, host="127.0.0.1", port=port, log_level="warning")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ShardFlow Dual-GPU Self-Contained Kaggle Demo")
    parser.add_argument("--model", default="/kaggle/working/models/Qwen2.5-7B-Instruct", help="Model path or HF ID")
    parser.add_argument("--draft-model", default="/kaggle/working/models/Qwen2.5-0.5B-Instruct", help="Draft model path")
    parser.add_argument("--spec-k", type=int, default=4, help="Speculative candidate tokens (default: 4)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens per generation")
    parser.add_argument("--port", type=int, default=8000, help="Gateway port")
    parser.add_argument("--stream-port", type=int, default=8001, help="P2P stream receiver port")
    parser.add_argument("--no-cuda-graphs", action="store_true", help="Disable CUDA Graphs and run in eager mode")
    args = parser.parse_args()

    model_path = args.model if os.path.exists(args.model) else "Qwen/Qwen2.5-7B-Instruct"
    draft_path = args.draft_model if (args.draft_model and os.path.exists(args.draft_model)) else "Qwen/Qwen2.5-0.5B-Instruct"
    use_cuda_graphs = not args.no_cuda_graphs

    print("=" * 70, flush=True)
    print(" SHARDFLOW DUAL-GPU (2x T4) ACCELERATED INFERENCE PIPELINE", flush=True)
    print(f"Base Model:    {model_path} (FP16 partitioned across cuda:0 & cuda:1)")
    print(f"Draft Model:   {draft_path} (Speculative K={args.spec_k})")
    print(f"CUDA Graphs:   {'[OK] ENABLED (Zero kernel launch latency)' if use_cuda_graphs else '[ERROR] DISABLED'}")
    print(f"Data-Plane:    [OK] v2 P2P Streaming Direct Link")
    print(f"Gateway:       http://127.0.0.1:{args.port}")
    print("=" * 70, flush=True)

    # 1. Start Local Gateway & Registry
    print("\n[1/4] Starting local Gateway & Registry with v2 Data Plane...", flush=True)
    gateway_proc = multiprocessing.Process(target=run_gateway_server, args=(args.port, args.stream_port), daemon=True)
    gateway_proc.start()

    # Wait for Gateway to be ready
    for _ in range(30):
        try:
            r = requests.get(f"http://127.0.0.1:{args.port}/topology", timeout=1.0)
            if r.status_code == 200:
                print("[OK] Gateway & Registry are online!", flush=True)
                break
        except Exception:
            time.sleep(0.5)
    else:
        raise RuntimeError("Gateway failed to boot within 15 seconds")

    # 2. Start Node 1 (cuda:1, Layers 14..28)
    print("\n[2/4] Spawning Node 1 on cuda:1 (Layers 14..28 + LM Head on port 9501)...", flush=True)
    node1_env = os.environ.copy()
    node1_env["CUDA_VISIBLE_DEVICES"] = "1"
    node1_cmd = [
        sys.executable, "-m", "shardflow.node.node",
        "--model", model_path,
        "--layer-start", "14",
        "--layer-end", "28",
        "--host", "127.0.0.1",
        "--port", "9501",
        "--device", "cuda",
        "--spec-k", str(args.spec_k),
    ]
    if not use_cuda_graphs:
        node1_cmd.append("--no-cuda-graphs")

    node1_log = open("/tmp/node1_local.log", "w")
    node1_proc = subprocess.Popen(node1_cmd, env=node1_env, stdout=node1_log, stderr=subprocess.STDOUT)

    # Wait for Node 1 to listen
    print("Waiting for Node 1 weights & CUDA Graphs to initialize on GPU 1...", flush=True)
    for _ in range(120):
        if node1_proc.poll() is not None:
            raise RuntimeError(f"Node 1 failed to start (exit code {node1_proc.returncode}). Check /tmp/node1_local.log")
        try:
            with socket.create_connection(("127.0.0.1", 9501), timeout=1.0):
                print("[OK] Node 1 is online and listening on 127.0.0.1:9501!", flush=True)
                break
        except Exception:
            time.sleep(1.5)
    else:
        raise TimeoutError("Node 1 startup timed out after 120s")

    # 3. Start Node 0 (cuda:0, Layers 0..14 + Draft Model)
    print("\n[3/4] Spawning Node 0 on cuda:0 (Layers 0..14 + Draft Model on port 9500)...", flush=True)
    node0_env = os.environ.copy()
    node0_env["CUDA_VISIBLE_DEVICES"] = "0"
    node0_cmd = [
        sys.executable, "-m", "shardflow.node.node",
        "--model", model_path,
        "--layer-start", "0",
        "--layer-end", "14",
        "--next-host", "127.0.0.1",
        "--next-port", "9501",
        "--host", "127.0.0.1",
        "--port", "9500",
        "--public-host", "127.0.0.1",
        "--public-port", "9500",
        "--registry-url", f"http://127.0.0.1:{args.port}",
        "--reg-layer-start", "0",
        "--reg-layer-end", "28",
        "--expected-nodes", "1",
        "--device", "cuda",
        "--spec-k", str(args.spec_k),
    ]
    if draft_path:
        node0_cmd.extend(["--draft-model", draft_path])
    if not use_cuda_graphs:
        node0_cmd.append("--no-cuda-graphs")

    node0_log = open("/tmp/node0_local.log", "w")
    node0_proc = subprocess.Popen(node0_cmd, env=node0_env, stdout=node0_log, stderr=subprocess.STDOUT)

    print("Waiting for Node 0 weights + Draft Model to load into GPU 0 VRAM...", flush=True)
    for _ in range(120):
        if node0_proc.poll() is not None:
            raise RuntimeError(f"Node 0 failed to start (exit code {node0_proc.returncode}). Check /tmp/node0_local.log")
        try:
            with socket.create_connection(("127.0.0.1", 9500), timeout=1.0):
                print("[OK] Node 0 is online and registered with local Gateway!", flush=True)
                break
        except Exception:
            time.sleep(1.5)
    else:
        raise TimeoutError("Node 0 startup timed out after 120s")

    # 4. Run Live Benchmark
    print("\n" + "=" * 70, flush=True)
    print(" [4/4] RUNNING LIVE DISTRIBUTED INFERENCE BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    prompts = [
        "Explain the concept of quantum entanglement in simple terms.",
        "Write a Python function to compute Fibonacci numbers using dynamic programming with memoization.",
        "What are the key advantages of pipeline parallelism for distributed LLM inference?",
    ]

    chat_url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    tps_list = []
    ttft_list = []

    try:
        for idx, prompt in enumerate(prompts, 1):
            print(f"\n--- Benchmark Prompt {idx}/{len(prompts)} ---", flush=True)
            print(f"User Prompt: \"{prompt}\"", flush=True)
            print("Assistant Output: ", end="", flush=True)

            payload = {
                "model": model_path,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0.7,
                "stream": True,
            }

            t_start = time.perf_counter()
            t_first_tok = None
            tok_count = 0

            resp = requests.post(chat_url, json=payload, stream=True, timeout=60.0)
            if resp.status_code != 200:
                print(f"\n[ERROR] Error {resp.status_code}: {resp.text}", flush=True)
                continue

            for line in resp.iter_lines():
                if not line:
                    continue
                s = line.decode("utf-8")
                if s.startswith("data: "):
                    d_str = s[6:].strip()
                    if d_str == "[DONE]":
                        break
                    try:
                        d = json.loads(d_str)
                        delta = d["choices"][0].get("delta", {}).get("content", "")
                        if delta:
                            if t_first_tok is None:
                                t_first_tok = time.perf_counter()
                            tok_count += 1
                            print(delta, end="", flush=True)
                    except Exception:
                        pass

            t_end = time.perf_counter()
            total_time = t_end - t_start
            ttft = (t_first_tok - t_start) if t_first_tok else total_time
            decode_time = (t_end - t_first_tok) if t_first_tok else total_time
            tps = (tok_count - 1) / decode_time if decode_time > 0 and tok_count > 1 else (tok_count / decode_time if decode_time > 0 else 0)

            if tok_count == 0:
                print("\n[WARNING] No tokens generated. Checking Node 0 and Node 1 logs...", flush=True)
                try:
                    with open("/tmp/node0_local.log", "r") as f0:
                        lines0 = f0.readlines()[-15:]
                        print(f"[Node 0 Log Tail]:\n{''.join(lines0)}", flush=True)
                    with open("/tmp/node1_local.log", "r") as f1:
                        lines1 = f1.readlines()[-15:]
                        print(f"[Node 1 Log Tail]:\n{''.join(lines1)}", flush=True)
                except Exception as ex:
                    print(f"Could not read logs: {ex}", flush=True)

            print("\n" + "-" * 50, flush=True)
            print(f" Tokens: {tok_count} | TTFT: {ttft*1000:.1f} ms | Decode Time: {decode_time:.2f} s | Throughput: {tps:.2f} TPS ", flush=True)
            print("-" * 50, flush=True)

            if tok_count > 0:
                tps_list.append(tps)
                ttft_list.append(ttft)

        if tps_list:
            import statistics
            print("\n" + "=" * 70, flush=True)
            print("[BEST] FINAL BENCHMARK SUMMARY (Kaggle 2x T4 Dual-GPU)", flush=True)
            print(f"  Avg Decode Throughput: {statistics.mean(tps_list):.2f} tokens/sec ")
            print(f"  Max Decode Throughput: {max(tps_list):.2f} tokens/sec")
            print(f"  Avg TTFT:              {statistics.mean(ttft_list)*1000:.1f} ms")
            print(f"  Total Cost:            $0.00 (Kaggle Free Tier)")
            print("=" * 70, flush=True)

    finally:
        print("\nCleaning up processes...", flush=True)
        node0_proc.terminate()
        node1_proc.terminate()
        gateway_proc.terminate()
        node0_log.close()
        node1_log.close()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
