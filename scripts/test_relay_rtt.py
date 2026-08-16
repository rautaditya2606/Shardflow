#!/usr/bin/env python3
"""
ShardFlow v2 — TCP Relay RTT & Bandwidth Benchmark.

Tests connectivity, HANDSHAKE, and 10KB tensor round-trip time (RTT)
through the EC2 Rust TCP Relay.

Usage:
1. On Node 1 (or Terminal 1):
   python scripts/test_relay_rtt.py --mode echo

2. On Node 0 (or Terminal 2):
   python scripts/test_relay_rtt.py --mode sender --iters 50
"""

import sys
import time
import argparse
import statistics
from pathlib import Path

# Add project root to sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from shardflow.transport.relay import (
    RELAY_HOST,
    RELAY_PORT,
    AUTH_BYTE,
    connect_to_relay,
    handshake,
    send_tensor,
    recv_tensor,
    send_token,
    recv_token,
)


def run_echo(host: str, port: int, auth_byte: bytes, iters: int):
    """Echo node (Node 1) — receives tensors and sends tokens back."""
    print("=" * 65)
    print(" SHARDFLOW RELAY TEST — ECHO NODE (NODE 1)")
    print(f"Target Relay: {host}:{port}")
    print("=" * 65)

    sock = connect_to_relay(host=host, port=port, auth_byte=auth_byte)
    try:
        handshake(sock, is_initiator=False)
        print("Waiting for tensors from sender...", flush=True)

        count = 0
        while True:
            try:
                tensor, drafts = recv_tensor(sock)
            except (ConnectionError, TimeoutError, EOFError, socket.error):
                break

            count += 1
            send_token(sock, token_id=12345 + count, accepted_count=1)
            if count % 10 == 0 or count == 1:
                print(f"  [Echo] Received tensor {list(tensor.shape)} ({tensor.dtype}) -> Sent token response #{count}", flush=True)

        print(f"[OK] Echo session completed ({count} requests handled).")
    finally:
        sock.close()


def run_sender(host: str, port: int, auth_byte: bytes, iters: int, dim: int, dtype_str: str):
    """Sender node (Node 0) — sends 10KB tensors, measures RTT, and reports realistic baseline TPS."""
    print("=" * 65)
    print(" SHARDFLOW RELAY TEST — SENDER NODE (NODE 0)")
    print(f"Target Relay: {host}:{port}")
    print(f"Payload Size: 1 x 1 x {dim} in {dtype_str} (~{(dim * 2) / 1024:.2f} KB)")
    print(f"Iterations:   {iters}")
    print("=" * 65)

    sock = connect_to_relay(host=host, port=port, auth_byte=auth_byte)
    dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16

    try:
        handshake(sock)
        print("\nStarting RTT benchmark...\n", flush=True)

        latencies_ms = []
        test_tensor = torch.randn(1, 1, dim, dtype=dtype)

        # Warmup iteration
        send_tensor(sock, test_tensor)
        recv_token(sock)

        for i in range(1, iters + 1):
            t_start = time.perf_counter()
            send_tensor(sock, test_tensor)
            tok_id, accepted_count, is_eos = recv_token(sock)
            t_end = time.perf_counter()

            rtt_ms = (t_end - t_start) * 1000.0
            latencies_ms.append(rtt_ms)

            if i % 10 == 0 or i == 1 or i == iters:
                print(f"  Iteration {i:3d}/{iters}: RTT = {rtt_ms:6.2f} ms | Received Token ID = {tok_id}", flush=True)

        print("\n" + "=" * 65)
        print(" BENCHMARK RESULTS")
        print("=" * 65)
        avg_rtt = statistics.mean(latencies_ms)
        min_rtt = min(latencies_ms)
        med_rtt = statistics.median(latencies_ms)
        max_rtt = max(latencies_ms)
        p95_rtt = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]

        print(f"  Min RTT:       {min_rtt:6.2f} ms")
        print(f"  Median RTT:    {med_rtt:6.2f} ms")
        print(f"  Mean RTT:      {avg_rtt:6.2f} ms")
        print(f"  95th-pct RTT:  {p95_rtt:6.2f} ms")
        print(f"  Max RTT:       {max_rtt:6.2f} ms")
        print("-" * 65)

        # Realistic Baseline TPS Projections (Pure Autoregressive Decode)
        tps_7b = 1000.0 / (avg_rtt + 20.0)    # ~20ms compute on 7B
        tps_14b = 1000.0 / (avg_rtt + 35.0)   # ~35ms compute on 14B

        print(" REALISTIC BASELINE DECODE TPS PROJECTIONS (Pure 1-Token Decode):")
        print(f"  Qwen2.5-7B  (RTT + ~20ms GPU compute):  {tps_7b:5.2f} tokens/sec")
        print(f"  Qwen2.5-14B (RTT + ~35ms GPU compute):  {tps_14b:5.2f} tokens/sec")
        print("=" * 65)

    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(description="ShardFlow TCP Relay RTT Benchmark")
    parser.add_argument("--mode", choices=["sender", "echo"], required=True, help="Role: 'echo' (start on Node 1) or 'sender' (start on Node 0)")
    parser.add_argument("--host", default=RELAY_HOST, help=f"Relay IP or host (default: {RELAY_HOST})")
    parser.add_argument("--port", type=int, default=RELAY_PORT, help=f"Relay port (default: {RELAY_PORT})")
    parser.add_argument("--iters", type=int, default=50, help="Number of benchmark iterations (default: 50, 0 for infinite in echo mode)")
    parser.add_argument("--dim", type=int, default=5120, help="Hidden dimension (default: 5120 for Qwen2.5-14B)")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16", help="Tensor precision (default: float16)")
    args = parser.parse_args()

    if args.mode == "echo":
        run_echo(host=args.host, port=args.port, auth_byte=AUTH_BYTE, iters=args.iters)
    else:
        run_sender(host=args.host, port=args.port, auth_byte=AUTH_BYTE, iters=args.iters, dim=args.dim, dtype_str=args.dtype)


if __name__ == "__main__":
    main()
