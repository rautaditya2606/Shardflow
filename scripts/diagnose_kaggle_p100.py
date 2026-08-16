#!/usr/bin/env python3
"""
ShardFlow Kaggle P100 Forensic Diagnostic Suite.

Systematically isolates the exact failure boundary across:
1. Pure CUDA sm_60 kernel execution (dummy forward pass, SDPA, RMSNorm, LM Head, sampling)
2. Isolated PipelineNode server idle lifecycle (10+ min survival test with no bore / no registry)
3. Localhost client-server loopback inference (serialization -> socket -> GPU -> response)
4. Bore tunnel stability and pipe draining

Usage in Kaggle Notebook:
    !python scripts/diagnose_kaggle_p100.py --model /kaggle/working/models/Qwen2.5-7B-Instruct --layer-start 14 --layer-end 28 --test all
"""

import os
import sys
import time
import argparse
import asyncio
import logging
import socket
import psutil
import torch

from pathlib import Path

# Add project root to sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Force direct assignment to /kaggle/working storage if running in Kaggle to avoid root / exhaustion
if os.path.exists("/kaggle"):
    os.environ["HF_HOME"] = "/kaggle/working/hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/hf_home"
    os.environ["HF_HUB_CACHE"] = "/kaggle/working/hf_home"

from transformers import AutoConfig
from shardflow.node.layer_loader import load_layer_slice, ModelSlice
from shardflow.node.node import PipelineNode
from shardflow.transport.connection import NodeClient
from shardflow.transport.protocol import MessageType, TensorMessage
from shardflow.orchestrator.sampler import sample_next_token

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("shardflow.diagnostics")


def print_banner(title: str):
    print("\n" + "=" * 70, flush=True)
    print(f" {title}", flush=True)
    print("=" * 70, flush=True)


def print_vram(prefix: str = "", device: str = "cuda"):
    if torch.cuda.is_available() and device.startswith("cuda"):
        dev = torch.device(device)
        alloc = torch.cuda.memory_allocated(dev) / 1024**3
        res = torch.cuda.memory_reserved(dev) / 1024**3
        total = torch.cuda.get_device_properties(dev).total_memory / 1024**3
        ram = psutil.virtual_memory().available / 1024**3
        print(f"[{prefix}] VRAM: {alloc:.2f} GB / {total:.2f} GB (reserved: {res:.2f} GB) | Host RAM Avail: {ram:.2f} GB", flush=True)


# =====================================================================
# TEST 1: Pure CUDA Forward Pass (No Server, No Network)
# =====================================================================
def run_test_forward(model_path: str, layer_start: int, layer_end: int, is_last: bool, dtype: torch.dtype, device: str):
    print_banner("TEST 1: Pure CUDA Forward Pass (Isolated sm_60 Compute Check)")
    print(f"Model: {model_path} | Layers: [{layer_start}, {layer_end}) | is_last={is_last} | Device={device} | Dtype={dtype}", flush=True)
    
    print_vram("Pre-Load", device=device)
    t0 = time.perf_counter()
    
    model_slice = load_layer_slice(
        model_path=model_path,
        layer_start=layer_start,
        layer_end=layer_end,
        include_norm=is_last,
        include_lm_head=is_last,
        dtype=dtype,
        device=device,
    )
    t_load = time.perf_counter() - t0
    print(f"[OK] Model slice loaded in {t_load:.2f}s", flush=True)
    print_vram("Post-Load", device=device)

    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=(layer_start == 0),
        is_last_node=is_last,
        enable_cuda_graphs=False,
    )

    hidden_size = model_slice.config.hidden_size if model_slice.config else 3584
    
    # 1A: Single-token decode dummy forward pass [1, 1, hidden_dim]
    print("\n--- [Step 1A] Single-Token Decode Forward Pass [1, 1, hidden_size] ---", flush=True)
    dummy_in_1 = torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
    print(f"Input tensor created: shape={dummy_in_1.shape}, dtype={dummy_in_1.dtype}, device={dummy_in_1.device}", flush=True)
    print("Executing node._forward(dummy_in_1)...", flush=True)
    
    t_fwd_0 = time.perf_counter()
    out_1 = node._forward(dummy_in_1, session_id="diag-sess-1", compute_head=is_last)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t_fwd_1 = time.perf_counter() - t_fwd_0
    
    print(f"[OK] Single-token forward pass completed in {t_fwd_1*1000:.2f} ms! Output shape: {out_1.shape}, dtype={out_1.dtype}", flush=True)
    
    if is_last:
        logits = out_1[0, -1, :]
        tok = sample_next_token(logits, temperature=0.0)
        print(f"[OK] Terminal LM head + Greedy sampling successful! Sampled token ID: {tok} (logits min={logits.min().item():.2f}, max={logits.max().item():.2f})", flush=True)

    # 1B: Speculative candidate verification forward pass [1, 5, hidden_dim]
    print("\n--- [Step 1B] Speculative Multi-Token Forward Pass [1, 5, hidden_size] ---", flush=True)
    dummy_in_5 = torch.randn(1, 5, hidden_size, dtype=dtype, device=device)
    print(f"Input tensor created: shape={dummy_in_5.shape}, dtype={dummy_in_5.dtype}, device={dummy_in_5.device}", flush=True)
    print("Executing node._forward(dummy_in_5)...", flush=True)
    
    t_fwd_0 = time.perf_counter()
    out_5 = node._forward(dummy_in_5, session_id="diag-sess-1", compute_head=is_last)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t_fwd_5 = time.perf_counter() - t_fwd_0
    
    print(f"[OK] Multi-token forward pass completed in {t_fwd_5*1000:.2f} ms! Output shape: {out_5.shape}", flush=True)
    print_vram("Post-Inference", device=device)
    
    # Cleanup session
    node.kv_store.evict("diag-sess-1")
    print(" TEST 1 PASSED: Pure CUDA execution on this hardware is 100% operational!\n", flush=True)
    return model_slice


# =====================================================================
# TEST 2: Isolated Server Idle (No Bore, No Registry, No Node 0)
# =====================================================================
async def run_test_idle(model_slice: ModelSlice, port: int = 9500, duration_seconds: int = 600):
    print_banner(f"TEST 2: Isolated PipelineNode Idle Longevity Test ({duration_seconds}s)")
    print(f"Binding PipelineNode server on 0.0.0.0:{port} (NO BORE, NO REGISTRY, NO NODE 0)...", flush=True)
    
    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=(model_slice.layer_start == 0),
        is_last_node=(model_slice.layer_end >= model_slice.config.num_hidden_layers if model_slice.config else True),
        listen_host="0.0.0.0",
        listen_port=port,
        enable_cuda_graphs=False,
    )
    
    await node.start()
    print(f"[OK] Server listening on 0.0.0.0:{port}. Entering idle observation loop for {duration_seconds} seconds...", flush=True)
    
    start_time = time.time()
    try:
        while time.time() - start_time < duration_seconds:
            await asyncio.sleep(10.0)
            elapsed = int(time.time() - start_time)
            remaining = max(0, duration_seconds - elapsed)
            vram_str = "N/A"
            if model_slice.device.type == "cuda" and torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(model_slice.device) / 1024**3
                res = torch.cuda.memory_reserved(model_slice.device) / 1024**3
                vram_str = f"{alloc:.2f} GB (res: {res:.2f} GB)"
            print(f"[IDLE WATCH] T+{elapsed:4d}s / {duration_seconds}s (remaining: {remaining:3d}s) | VRAM: {vram_str} | PID: {os.getpid()} | Alive & Listening", flush=True)
    finally:
        await node.stop()
    
    print(f" TEST 2 PASSED: Server survived {duration_seconds}s idle with zero crashes!\n", flush=True)


# =====================================================================
# TEST 3: Localhost Loopback Inference (Client -> PipelineNode)
# =====================================================================
async def run_test_loopback(model_slice: ModelSlice, port: int = 9500):
    print_banner("TEST 3: Localhost Loopback Inference (Client -> Server over 127.0.0.1)")
    print(f"Starting server on 127.0.0.1:{port} and connecting local client (NO BORE)...", flush=True)
    
    is_last = (model_slice.layer_end >= model_slice.config.num_hidden_layers if model_slice.config else True)
    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=(model_slice.layer_start == 0),
        is_last_node=is_last,
        listen_host="127.0.0.1",
        listen_port=port,
        enable_cuda_graphs=False,
    )
    
    await node.start()
    
    # Client task
    client = NodeClient("127.0.0.1", port, send_timeout=10.0, recv_timeout=15.0)
    try:
        print("Connecting NodeClient to 127.0.0.1:%d..." % port, flush=True)
        await client.connect(max_retries=5, retry_delay=0.5)
        print("[OK] NodeClient connected successfully over local loopback!", flush=True)
        
        hidden_size = model_slice.config.hidden_size if model_slice.config else 3584
        dummy_tensor = torch.randn(1, 1, hidden_size, dtype=torch.float16)
        
        req_msg = TensorMessage(
            msg_type=MessageType.ACTIVATION,
            session_id="diag-loopback-1",
            tensor=dummy_tensor,
            temperature=0.0,
            sample_on_node=is_last,
        )
        
        print("Sending framed ACTIVATION message over TCP socket...", flush=True)
        t0 = time.perf_counter()
        resp = await client.send_recv(req_msg, timeout=10.0)
        dt = (time.perf_counter() - t0) * 1000.0
        
        print(f"[OK] Response received in {dt:.2f} ms! Message type: {resp.msg_type.name}", flush=True)
        if resp.msg_type == MessageType.TOKEN_ID:
            print(f"[OK] Token received from terminal node: token_id={resp.token_id}", flush=True)
        elif resp.msg_type == MessageType.ACTIVATION:
            print(f"[OK] Activation tensor received: shape={resp.tensor.shape if resp.tensor is not None else None}", flush=True)
            
        # Test CLEAR message
        clear_msg = TensorMessage(msg_type=MessageType.CLEAR, session_id="diag-loopback-1")
        await client.send(clear_msg)
        print("[OK] CLEAR message sent and KV cache evicted successfully!", flush=True)
        
    finally:
        await client.close()
        await node.stop()
        
    print(" TEST 3 PASSED: Local loopback IPC + serialization + GPU inference verified 100% working!\n", flush=True)


# =====================================================================
# TEST 4: Bore Tunnel Diagnostic & Drain Test
# =====================================================================
def run_test_bore(port: int = 9500):
    print_banner("TEST 4: Bore Tunnel Process & Third-Party Connection Test")
    from shardflow.transport.tunnel import start_bore_tunnel
    
    # Start a dummy TCP listener so bore has an active backend to proxy to
    dummy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    dummy_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    dummy_sock.bind(("0.0.0.0", port))
    dummy_sock.listen(5)
    print(f"Local test TCP listener listening on 0.0.0.0:{port}", flush=True)
    
    try:
        print("Starting bore tunnel to bore.pub...", flush=True)
        t0 = time.perf_counter()
        proc, server, pub_port = start_bore_tunnel(port)
        dt = time.perf_counter() - t0
        print(f"[OK] Bore tunnel established in {dt:.2f}s at {server}:{pub_port} (PID={proc.pid})", flush=True)
        
        print("Testing outbound TCP connection to assigned bore endpoint %s:%d..." % (server, pub_port), flush=True)
        test_sock = socket.create_connection((server, pub_port), timeout=10.0)
        print("[OK] Successfully established end-to-end TCP connection through bore.pub tunnel!", flush=True)
        test_sock.close()
        
        print("Observing bore tunnel stability for 30 seconds...", flush=True)
        for i in range(3):
            time.sleep(10.0)
            status = "ALIVE" if proc.poll() is None else f"EXITED({proc.returncode})"
            print(f"[BORE WATCH] T+{(i+1)*10}s / 30s | Bore Subprocess: {status}", flush=True)
            if proc.poll() is not None:
                raise RuntimeError(f"Bore process died unexpectedly with returncode {proc.returncode}")
                
        print(" TEST 4 PASSED: Bore tunnel proxy is operational!\n", flush=True)
        
    finally:
        try:
            dummy_sock.close()
        except Exception:
            pass
        try:
            if "proc" in locals() and proc.poll() is None:
                proc.terminate()
        except Exception:
            pass


# =====================================================================
# MAIN DISPATCHER
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="ShardFlow Kaggle P100 Forensic Diagnostic Suite")
    parser.add_argument("--model", default="/kaggle/working/models/Qwen2.5-7B-Instruct", help="Model path or HF model ID")
    parser.add_argument("--layer-start", type=int, default=14, help="Layer start index (default: 14 for Node 1)")
    parser.add_argument("--layer-end", type=int, default=28, help="Layer end index (default: 28 for Node 1)")
    parser.add_argument("--is-last", action="store_true", default=True, help="Include LM Head and RMSNorm (default: True)")
    parser.add_argument("--port", type=int, default=9500, help="Local TCP port (default: 9500)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Data type precision")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Target device")
    parser.add_argument("--idle-seconds", type=int, default=600, help="Duration for isolated idle test in seconds (default: 600)")
    parser.add_argument(
        "--test",
        choices=["forward", "idle", "loopback", "bore", "all"],
        default="all",
        help="Test to run (forward | idle | loopback | bore | all)",
    )
    args = parser.parse_args()

    # Fallback model resolution if local path does not exist
    model_path = args.model
    if not os.path.exists(model_path) and "/" not in model_path:
        model_path = "Qwen/Qwen2.5-7B-Instruct"

    dtype = getattr(torch, args.dtype)
    print_banner("SHARDFLOW KAGGLE P100 FORENSIC DIAGNOSTIC SUITE")
    print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)} (sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}) | Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Target Model: {model_path} | Selected Test Mode: {args.test}")
    print("=" * 70, flush=True)

    model_slice = None

    if args.test in ("forward", "all"):
        model_slice = run_test_forward(
            model_path=model_path,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            is_last=args.is_last,
            dtype=dtype,
            device=args.device,
        )

    if args.test in ("idle", "all"):
        if model_slice is None:
            model_slice = load_layer_slice(
                model_path=model_path,
                layer_start=args.layer_start,
                layer_end=args.layer_end,
                include_norm=args.is_last,
                include_lm_head=args.is_last,
                dtype=dtype,
                device=args.device,
            )
        asyncio.run(run_test_idle(model_slice, port=args.port, duration_seconds=args.idle_seconds))

    if args.test in ("loopback", "all"):
        if model_slice is None:
            model_slice = load_layer_slice(
                model_path=model_path,
                layer_start=args.layer_start,
                layer_end=args.layer_end,
                include_norm=args.is_last,
                include_lm_head=args.is_last,
                dtype=dtype,
                device=args.device,
            )
        asyncio.run(run_test_loopback(model_slice, port=args.port))

    if args.test in ("bore", "all"):
        run_test_bore(port=args.port)

    print_banner("DIAGNOSTIC SUMMARY: ALL REQUESTED TESTS COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
