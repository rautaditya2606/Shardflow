#!/usr/bin/env python3
"""
ShardFlow Google Colab Node 1 Runner (Terminal Slice & Speculative Verifier).

Runs Node 1 (Layers 14..28 + LM Head on Qwen2.5-7B) on a Google Colab T4 GPU
connected directly to the AWS EC2 TCP Relay.

Usage on Colab (Notebook 1):
  !python scripts/colab_node1.py --model Qwen/Qwen2.5-7B-Instruct --relay-host 3.23.174.207 --relay-port 9500
"""

import os
import sys
import time
import socket
import argparse
import logging
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path

# Add project root to sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

if os.path.exists("/content"):
    os.environ["HF_HOME"] = "/content/hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/content/hf_home"
    os.environ["HF_HUB_CACHE"] = "/content/hf_home"

import torch
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available in this Colab session! "
        "Please enable GPU acceleration via: Runtime -> Change runtime type -> T4 GPU."
    )

from transformers import AutoConfig
from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.node import PipelineNode
from shardflow.node.draft_model import rewind_kv_cache
from shardflow.orchestrator.sampler import sample_next_token
from shardflow.transport.relay import (
    RELAY_HOST,
    RELAY_PORT,
    AUTH_BYTE,
    connect_to_relay,
    handshake,
    recv_tensor,
    recv_tensor_timed,
    send_token,
    send_token_timed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("colab_node1")


class Node1Profiler:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.total_tokens = 0
        self.accepted_drafts = 0
        self.total_drafts = 0
        self.node1_comp_times: List[float] = []

    def record(self, comp_ms: float, accepted: int = 1, drafted: int = 0):
        self.steps += 1
        self.total_tokens += accepted
        self.accepted_drafts += max(0, accepted - 1)
        self.total_drafts += drafted
        self.node1_comp_times.append(comp_ms)

    def summary(self) -> Dict[str, float]:
        if not self.node1_comp_times:
            return {}
        import numpy as np
        return {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "avg_comp_ms": float(np.mean(self.node1_comp_times)),
            "p50_comp_ms": float(np.percentile(self.node1_comp_times, 50)),
            "p95_comp_ms": float(np.percentile(self.node1_comp_times, 95)),
            "accept_rate": (self.accepted_drafts / max(1, self.total_drafts)) * 100.0,
        }


def main():
    parser = argparse.ArgumentParser(description="ShardFlow Google Colab Node 1 (Terminal Slice)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model ID or path")
    parser.add_argument("--layer-start", type=int, default=14, help="Starting layer index for Node 1 slice")
    parser.add_argument("--layer-end", type=int, default=None, help="Ending layer index (default: last layer)")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device to use (default: cuda:0)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--relay-host", type=str, default=RELAY_HOST, help="AWS EC2 relay IP/hostname")
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT, help="AWS EC2 relay port")
    parser.add_argument("--max-sessions", type=int, default=1000, help="Max requests before restart")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model)
    total_layers = getattr(config, "num_hidden_layers", 28)
    layer_start = args.layer_start
    layer_end = args.layer_end if args.layer_end is not None else total_layers

    target_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    if target_dtype == torch.bfloat16 and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.warning("Device does not natively support bfloat16 (capability < 8.0). Falling back to float16.")
            target_dtype = torch.float16

    print("=" * 70)
    print(" SHARDFLOW GOOGLE COLAB NODE 1 (TERMINAL SLICE & VERIFIER)")
    print(f"Base Model:   {args.model}")
    print(f"Layer Split:  [{layer_start}, {layer_end}) of {total_layers} layers")
    print(f"Device:       {args.device} ({torch.cuda.get_device_name(0)})")
    print(f"Relay Target: {args.relay_host}:{args.relay_port}")
    print("=" * 70)

    # 1. Load Model Slice
    t0 = time.perf_counter()
    model_slice = load_layer_slice(
        model_path=args.model,
        layer_start=layer_start,
        layer_end=layer_end,
        device=args.device,
        dtype=target_dtype,
    )
    t_load = time.perf_counter() - t0
    logger.info("Terminal model slice loaded in %0.2fs", t_load)

    # 2. Initialize Pipeline Node
    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=False,
        is_last_node=True,
    )

    profiler = Node1Profiler()

    # 3. Connection & Execution Loop
    while True:
        try:
            logger.info("Connecting to TCP relay at %s:%d ...", args.relay_host, args.relay_port)
            sock = connect_to_relay(host=args.relay_host, port=args.relay_port, auth_byte=AUTH_BYTE)
            logger.info("Connected to relay. Waiting for Node 0 to connect...")

            # Colab Node 1 is listener in the 2-way handshake
            handshake(sock, is_initiator=False)
            logger.info("HANDSHAKE COMPLETE! Entering pure compute decode loop...")

            session_id = "relay_session"
            step = 0
            last_verified_round_id = 0

            while True:
                # Receive activation frame from Node 0
                tensor, drafts, recv_stats = recv_tensor_timed(sock)
                round_id = recv_stats.get("round_id", 0)

                t_comp_0 = time.perf_counter()

                # Handle multi-token speculative verification
                if drafts and len(drafts) > 0:
                    num_candidates = tensor.shape[1]
                    output = node.forward(tensor, session_id=session_id, compute_head=True)
                    if output.is_cuda:
                        torch.cuda.synchronize(output.device)

                    logits = output[0]  # [num_candidates, vocab_size]
                    accepted_count = 0
                    is_eos = False
                    next_token = -1

                    # Verify candidate tokens sequentially
                    for k_idx in range(num_candidates):
                        cand_logits = logits[k_idx]
                        predicted_tok = int(cand_logits.argmax(dim=-1).item())

                        if k_idx < len(drafts):
                            draft_tok = drafts[k_idx]
                            if predicted_tok == draft_tok:
                                accepted_count += 1
                                if predicted_tok in (151643, 151645):  # EOS
                                    is_eos = True
                                    next_token = predicted_tok
                                    break
                            else:
                                next_token = predicted_tok
                                accepted_count += 1
                                break
                        else:
                            next_token = predicted_tok
                            accepted_count += 1
                            break

                    # Rollback KV cache on target model to accepted length
                    past_len = node.get_session_seq_len(session_id)
                    committed_len = past_len - (num_candidates - accepted_count)
                    if session_id in node.kv_pools:
                        rewind_kv_cache(node.kv_pools[session_id], committed_len)

                    t_comp_1 = time.perf_counter()
                    node1_comp_ms = (t_comp_1 - t_comp_0) * 1000.0

                    send_token_timed(
                        sock,
                        token_id=next_token,
                        accepted_count=accepted_count,
                        is_eos=is_eos,
                        node1_compute_ms=node1_comp_ms,
                    )
                    profiler.record(node1_comp_ms, accepted=accepted_count, drafted=len(drafts))

                else:
                    # Standard single-token autoregressive step
                    output = node.forward(tensor, session_id=session_id, compute_head=True)
                    if output.is_cuda:
                        torch.cuda.synchronize(output.device)

                    t_comp_1 = time.perf_counter()
                    node1_comp_ms = (t_comp_1 - t_comp_0) * 1000.0

                    next_token = int(output[0, -1, :].argmax(dim=-1).item())
                    is_eos = next_token in (151643, 151645)

                    send_token_timed(
                        sock,
                        token_id=next_token,
                        accepted_count=1,
                        is_eos=is_eos,
                        node1_compute_ms=node1_comp_ms,
                    )
                    profiler.record(node1_comp_ms, accepted=1, drafted=0)

                step += 1

        except (ConnectionError, TimeoutError, EOFError, socket.error) as e:
            logger.warning("Session ended or disconnected (%s). Re-entering connection loop in 2s...", e)
            time.sleep(2.0)
        except Exception as e:
            logger.error("Unexpected error in Colab Node 1: %s", e, exc_info=True)
            time.sleep(2.0)


if __name__ == "__main__":
    main()
