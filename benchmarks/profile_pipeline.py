"""
Pipeline Stage Profiler — measures exact latency breakdown (in ms) across all stages:
1. Embedding
2. Node 0 forward (CUDA compute vs CPU transfer)
3. Wire / TCP transfer
4. Node 1 forward (CUDA compute vs CPU transfer)
5. Sampling
"""

import argparse
import asyncio
import logging
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoConfig

from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.node import PipelineNode
from shardflow.orchestrator.orchestrator import Orchestrator
from shardflow.orchestrator.sampler import sample_next_token
from shardflow.transport.protocol import TensorMessage, MessageType


async def profile_pipeline(
    model_path: str,
    prompt: str = "Once upon a time",
    max_tokens: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"\n{'='*60}")
    print(f"PROFILING SHARDFLOW PIPELINE (Device: {device})")
    print(f"{'='*60}\n")

    config = AutoConfig.from_pretrained(model_path)
    total_layers = config.num_hidden_layers
    num_nodes = 2
    base_port = 9300

    # Load model slices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Loading model slices...")
    slice0 = load_layer_slice(
        model_path=model_path,
        layer_start=0,
        layer_end=4,
        include_norm=False,
        include_lm_head=False,
        device=device,
    )
    slice1 = load_layer_slice(
        model_path=model_path,
        layer_start=4,
        layer_end=8,
        include_norm=True,
        include_lm_head=True,
        device=device,
    )

    # Check device placement of parameters
    print(f"Node 0 layer 0 device: {next(slice0.layers[0].parameters()).device}")
    print(f"Node 1 layer 0 device: {next(slice1.layers[0].parameters()).device}")
    print(f"Node 0 layer 0 dtype: {next(slice0.layers[0].parameters()).dtype}")

    # Create nodes
    node1 = PipelineNode(
        model_slice=slice1,
        is_first_node=False,
        is_last_node=True,
        listen_host="127.0.0.1",
        listen_port=base_port + 1,
    )
    node0 = PipelineNode(
        model_slice=slice0,
        is_first_node=True,
        is_last_node=False,
        next_node_host="127.0.0.1",
        next_node_port=base_port + 1,
        listen_host="127.0.0.1",
        listen_port=base_port,
    )

    await node1.start()
    await asyncio.sleep(0.2)
    await node0.start()
    await asyncio.sleep(0.2)

    orchestrator = Orchestrator(
        model_path=model_path,
        node_addresses=[("127.0.0.1", base_port)],
        device="cpu",
    )
    await orchestrator.initialize()

    # Prepare input
    inputs = orchestrator.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Warmup step
    session_id = "profiling-session"
    h_warm = orchestrator._embed(input_ids)
    msg_warm = TensorMessage(msg_type=MessageType.ACTIVATION, session_id=session_id, tensor=h_warm.cpu())
    await orchestrator._node0_client.send_recv(msg_warm)

    # Profile decode steps
    embed_times = []
    transport_roundtrip_times = []
    sampling_times = []
    total_token_times = []

    next_token = 100
    print(f"\nProfiling {max_tokens} decode tokens...")
    print(f"{'Token':<8} | {'Embed':<10} | {'Pipeline Round-Trip':<22} | {'Hop Wire (ms)':<15} | {'Total (ms)':<12}")
    print("-" * 75)

    for step in range(max_tokens):
        t_start = time.perf_counter()

        # 1. Embed
        t0 = time.perf_counter()
        token_tensor = torch.tensor([[next_token]], dtype=torch.long)
        hidden_states = orchestrator._embed(token_tensor)
        t_embed = (time.perf_counter() - t0) * 1000

        # 2. TCP + Pipeline Roundtrip
        t1 = time.perf_counter()
        now_us = int(time.perf_counter() * 1_000_000)
        msg = TensorMessage(
            msg_type=MessageType.ACTIVATION,
            session_id=session_id,
            send_ts_us=now_us,
            tensor=hidden_states.cpu(),
            temperature=0.0,
            sample_on_node=True,
        )
        response = await orchestrator._node0_client.send_recv(msg)
        t_pipeline = (time.perf_counter() - t1) * 1000
        hop_wire_ms = orchestrator._node0_client.last_hop_latency_ms

        # 3. Sampling
        t2 = time.perf_counter()
        if response.msg_type == MessageType.TOKEN_ID:
            next_token = response.token_id
            t_sample = (time.perf_counter() - t2) * 1000
        elif response.msg_type == MessageType.LOGITS:
            logits = response.tensor[0, -1, :]
            next_token = sample_next_token(logits, temperature=0.0)
            t_sample = (time.perf_counter() - t2) * 1000
        else:
            raise RuntimeError(f"Unexpected response msg_type: {response.msg_type}")

        t_total = (time.perf_counter() - t_start) * 1000

        embed_times.append(t_embed)
        transport_roundtrip_times.append(t_pipeline)
        sampling_times.append(t_sample)
        total_token_times.append(t_total)

        print(f"{step+1:<8} | {t_embed:<8.2f}ms | {t_pipeline:<20.2f}ms | {hop_wire_ms:<13.2f}ms | {t_total:<10.2f}ms")

    # Summary
    avg_embed = sum(embed_times) / len(embed_times)
    avg_pipe = sum(transport_roundtrip_times) / len(transport_roundtrip_times)
    avg_sample = sum(sampling_times) / len(sampling_times)
    avg_total = sum(total_token_times) / len(total_token_times)
    tps = 1000.0 / avg_total if avg_total > 0 else 0

    print("\n" + "=" * 75)
    print("STAGE BREAKDOWN SUMMARY (Averages):")
    print(f"  1. Token Embedding:     {avg_embed:.2f} ms ({avg_embed/avg_total*100:.1f}%)")
    print(f"  2. Pipeline + TCP Hop:  {avg_pipe:.2f} ms ({avg_pipe/avg_total*100:.1f}%)")
    print(f"  3. Token Sampling:      {avg_sample:.2f} ms ({avg_sample/avg_total*100:.1f}%)")
    print(f"  -------------------------------------------------------------")
    print(f"  Total Per-Token Time:   {avg_total:.2f} ms")
    print(f"  Achieved Throughput:    {tps:.2f} tokens/sec")
    print(f"  Connection Reconnects:  {orchestrator._node0_client.reconnect_count}")
    print("=" * 75)

    # Cleanup
    await orchestrator.shutdown()
    await node0.stop()
    await node1.stop()


def main():
    parser = argparse.ArgumentParser(description="ShardFlow Pipeline Stage Profiler")
    parser.add_argument("--model", default="./models/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-tokens", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    asyncio.run(profile_pipeline(model_path=args.model, max_tokens=args.max_tokens, device=args.device))


if __name__ == "__main__":
    main()
