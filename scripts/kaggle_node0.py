#!/usr/bin/env python3
"""
ShardFlow v2 Kaggle Remote Node 0 Runner (Kaggle Instance A).

Runs Node 0 (Layers 0..24 + Embeddings on Qwen2.5-14B, or Layers 0..14 on Qwen2.5-7B)
connected directly to the AWS EC2 Rust TCP Relay without tunnels.

Drives the distributed generation loop peer-to-peer across the relay:
Node 0 (Embed + Layers 0..24) -> Relay -> Node 1 (Layers 24..48 + LM Head) -> Relay -> Node 0
"""

import os
import sys
import time
import socket
import argparse
import logging
import statistics
import queue
import threading
from typing import Optional, Tuple, List, Dict, Union, Callable, Set
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    gr = None

# Add project root to sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

if os.path.exists("/kaggle"):
    os.environ["HF_HOME"] = "/kaggle/working/hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/hf_home"
    os.environ["HF_HUB_CACHE"] = "/kaggle/working/hf_home"

import torch
from transformers import AutoConfig, AutoTokenizer

from shardflow.node.layer_loader import load_layer_slice, get_num_hidden_layers
from shardflow.node.node import PipelineNode
from shardflow.node.draft_model import DraftSampler, rewind_kv_cache
from shardflow.node.ngram_draft import NGramDraftSampler
from shardflow.transport.relay import (
    RELAY_HOST,
    RELAY_PORT,
    AUTH_BYTE,
    connect_to_relay,
    handshake,
    send_tensor,
    send_tensor_timed,
    recv_tensor,
    recv_token,
    recv_token_timed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("node0")


def get_eos_token_ids(tokenizer=None, custom_eos: Optional[int] = None) -> Set[int]:
    """Return all stop token IDs including chat turn endings (<|im_end|>, <|eot_id|>, </s>, etc.)."""
    ids = set()
    if tokenizer is not None:
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            if isinstance(tokenizer.eos_token_id, (list, tuple, set)):
                ids.update(tokenizer.eos_token_id)
            else:
                ids.add(tokenizer.eos_token_id)
        if hasattr(tokenizer, "all_special_ids") and tokenizer.all_special_ids:
            for name in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "</s>", "<eos>"]:
                try:
                    tok_id = tokenizer.convert_tokens_to_ids(name)
                    if tok_id is not None and isinstance(tok_id, int) and tok_id > 0 and tok_id != getattr(tokenizer, "unk_token_id", None):
                        ids.add(tok_id)
                except Exception:
                    pass
    if custom_eos is not None:
        ids.add(custom_eos)
    # Common end-of-turn / end-of-text IDs across Qwen, LLaMA-3, Mistral
    ids.update({151643, 151645, 128001, 128009, 2})
    return ids


class Node0Profiler:
    """Microsecond-accurate latency profiler for Node 0 decode steps."""

    def __init__(self):
        self.embed_times = []
        self.draft_gen_times = []
        self.draft_wait_times = []
        self.gpu0_fwd_times = []
        self.pcie_transfer_times = []
        self.gpu1_fwd_times = []
        self.node0_gpu_times = []
        self.gpu_to_cpu_times = []
        self.serialize_times = []
        self.tcp_send_times = []
        self.tcp_recv_wait_times = []
        self.node1_compute_times = []
        self.pure_network_rtt_times = []
        self.inter_round_bubble_times = []
        self.total_step_times = []
        # ponytail: per-round speculative tracking
        self.accepted_per_round = []
        self.drafted_per_round = []
        self.is_spec_step = []

    def record(
        self,
        embed_ms: float,
        gpu_fwd_ms: float,
        g2c_ms: float,
        ser_ms: float,
        send_ms: float,
        recv_ms: float,
        total_ms: float,
        draft_gen_ms: float = 0.0,
        draft_wait_ms: float = 0.0,
        gpu0_ms: float = 0.0,
        pcie_ms: float = 0.0,
        gpu1_ms: float = 0.0,
        node1_compute_ms: float = 0.0,
        network_rtt_ms: float = 0.0,
        bubble_ms: float = 0.0,
        accepted: int = 1,
        drafted: int = 0,
        is_spec: bool = False,
    ):
        self.embed_times.append(embed_ms)
        self.draft_gen_times.append(draft_gen_ms)
        self.draft_wait_times.append(draft_wait_ms)
        self.gpu0_fwd_times.append(gpu0_ms)
        self.pcie_transfer_times.append(pcie_ms)
        self.gpu1_fwd_times.append(gpu1_ms)
        self.node0_gpu_times.append(gpu_fwd_ms)
        self.gpu_to_cpu_times.append(g2c_ms)
        self.serialize_times.append(ser_ms)
        self.tcp_send_times.append(send_ms)
        self.tcp_recv_wait_times.append(recv_ms)
        self.node1_compute_times.append(node1_compute_ms)
        self.pure_network_rtt_times.append(network_rtt_ms)
        self.inter_round_bubble_times.append(bubble_ms)
        self.total_step_times.append(total_ms)
        self.accepted_per_round.append(accepted)
        self.drafted_per_round.append(drafted)
        self.is_spec_step.append(is_spec)

    def print_breakdown(self):
        if not self.total_step_times:
            return
        n = len(self.total_step_times)
        avg = lambda lst: (sum(lst) / len(lst)) if lst else 0.0
        p95 = lambda lst: sorted(lst)[int(len(lst) * 0.95)] if lst else 0.0

        avg_total = avg(self.total_step_times)
        print("\n" + "=" * 70, flush=True)
        print(f"⏱️ NODE 0 PER-TOKEN LATENCY PROFILER BREAKDOWN ({n} decode steps)", flush=True)
        print("=" * 70, flush=True)
        if any(t > 0 for t in self.draft_gen_times):
            print(f"  0a. Draft Gen (cuda:1 Async): {avg(self.draft_gen_times):6.2f} ms  (p95: {p95(self.draft_gen_times):6.2f} ms)")
        if any(t > 0 for t in self.draft_wait_times):
            print(f"  0b. Draft Wait at Recon:      {avg(self.draft_wait_times):6.2f} ms  (p95: {p95(self.draft_wait_times):6.2f} ms)")
        print(f"  1. Token Embeddings:          {avg(self.embed_times):6.2f} ms  (p95: {p95(self.embed_times):6.2f} ms)")

        if any(t > 0 for t in self.pcie_transfer_times):
            print(f"  2a. GPU 0 Forward (L0..L6):   {avg(self.gpu0_fwd_times):6.2f} ms  (p95: {p95(self.gpu0_fwd_times):6.2f} ms)")
            print(f"  2b. PCIe Transfer (G0 -> G1): {avg(self.pcie_transfer_times):6.2f} ms  (p95: {p95(self.pcie_transfer_times):6.2f} ms)")
            print(f"  2c. GPU 1 Forward (L7..L13):  {avg(self.gpu1_fwd_times):6.2f} ms  (p95: {p95(self.gpu1_fwd_times):6.2f} ms)")
            print(f"  2. Total Node 0 Forward (Sync):{avg(self.node0_gpu_times):6.2f} ms  (p95: {p95(self.node0_gpu_times):6.2f} ms)")
        else:
            print(f"  2. Node 0 GPU Forward (Sync): {avg(self.node0_gpu_times):6.2f} ms  (p95: {p95(self.node0_gpu_times):6.2f} ms)")

        print(f"  3. GPU -> CPU Transfer:       {avg(self.gpu_to_cpu_times):6.2f} ms  (p95: {p95(self.gpu_to_cpu_times):6.2f} ms)")
        print(f"  4. Tensor Serialization:      {avg(self.serialize_times):6.2f} ms  (p95: {p95(self.serialize_times):6.2f} ms)")
        print(f"  5. TCP Send (Node 0 -> EC2):  {avg(self.tcp_send_times):6.2f} ms  (p95: {p95(self.tcp_send_times):6.2f} ms)")
        print(f"  6. Total Recv Wait (Node 0):  {avg(self.tcp_recv_wait_times):6.2f} ms  (p95: {p95(self.tcp_recv_wait_times):6.2f} ms)")
        if any(t > 0 for t in self.node1_compute_times):
            print(f"     ├── Node 1 Remote Compute: {avg(self.node1_compute_times):6.2f} ms  (p95: {p95(self.node1_compute_times):6.2f} ms)")
            print(f"     └── Pure Network Wire RTT: {avg(self.pure_network_rtt_times):6.2f} ms  (p95: {p95(self.pure_network_rtt_times):6.2f} ms)")
        if any(t >= 0 for t in self.inter_round_bubble_times):
            print(f"  7. Inter-Round Bubble (T10-T9):{avg(self.inter_round_bubble_times):6.2f} ms  (p95: {p95(self.inter_round_bubble_times):6.2f} ms)")
        print("  " + "-" * 66, flush=True)
        print(f"  TOTAL STEP LATENCY:           {avg_total:6.2f} ms  ({1000.0/avg_total:.2f} TPS)", flush=True)

        spec_acc = [acc for acc, is_s in zip(self.accepted_per_round, self.is_spec_step) if is_s]
        spec_drf = [drf for drf, is_s in zip(self.drafted_per_round, self.is_spec_step) if is_s]
        if spec_acc:
            tot_acc = sum(spec_acc)
            tot_drf = sum(spec_drf)
            tok_per_round = tot_acc / len(spec_acc)
            acc_rate = (sum(max(0, a - 1) for a in spec_acc) / tot_drf * 100.0) if tot_drf > 0 else 0.0
            print("  " + "-" * 66, flush=True)
            print(f"  Spec Rounds: {len(spec_acc):3d} | Eff Tokens/Round: {tok_per_round:4.2f} | Bonus Accept Rate: {acc_rate:5.1f}%", flush=True)
        print("=" * 70, flush=True)


import queue
import threading


class AsyncTokenReceiver:
    """
    ponytail: Lightweight background receiver thread that pulls tokens from relay socket
    into a thread-safe bounded queue, unblocking GPU execution.
    """

    def __init__(self, sock: socket.socket, maxsize: int = 4):
        self.sock = sock
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            try:
                result = recv_token_timed(self.sock)
                self.queue.put(result)
            except Exception as e:
                self.queue.put(e)
                break

    def get(self, timeout: float = 30.0) -> Tuple[int, int, bool, dict]:
        item = self.queue.get(timeout=timeout)
        if isinstance(item, Exception):
            raise item
        return item

    def stop(self):
        self._stop_event.set()


def generate(
    prompt: str,
    tokenizer,
    node: PipelineNode,
    sock: socket.socket,
    max_tokens: int = 100,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    spec_k: int = 0,
    ngram_sampler: Optional[NGramDraftSampler] = None,
    eos_token_id: int = 151643,
    profiler: Optional[Node0Profiler] = None,
    receiver: Optional[AsyncTokenReceiver] = None,
    enable_async_spec: bool = False,
    spec_window: int = 1,
    token_callback: Optional[Callable[[str, dict], None]] = None,
    print_output: bool = True,
) -> dict:
    """Generate tokens for a prompt through the distributed relay pipeline with error handling."""
    eos_token_ids = get_eos_token_ids(tokenizer, eos_token_id)
    session_id = f"relay_session_{int(time.time()*1000)}"
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)

    # Initialize neural draft sampler if specified
    draft_sampler = node.draft_sampler if (spec_k > 0 and node.draft_sampler is not None) else None
    async_drafter = getattr(node, "async_draft_sampler", None) if (spec_k > 0 and getattr(node, "async_draft_sampler", None) is not None) else None
    if async_drafter:
        async_drafter.prefill(prompt_tokens)
    elif draft_sampler:
        draft_sampler.prefill(prompt_tokens)

    if spec_k > 0 and draft_sampler is None and ngram_sampler is None and async_drafter is None:
        if print_output:
            print(f"\n[WARNING] spec_k={spec_k} but neither draft_sampler nor ngram_sampler is active! Running 1-token decode.", flush=True)

    t_start = time.perf_counter()
    t_first_token = None
    generated_tokens = []
    token_history = list(prompt_tokens)
    total_drafted = 0
    total_accepted = 0

    if print_output:
        print(f"\nUser Prompt: \"{prompt}\"", flush=True)
        print("Assistant: ", end="", flush=True)

    try:
        # 1. Prefill Phase
        token_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=node.model_slice.device)
        if node.model_slice.embed_tokens is not None:
            hidden_states = node.model_slice.embed_tokens(token_tensor)
        else:
            hidden_states = token_tensor

        # Forward pass on Node 0
        output = node._forward(hidden_states, session_id=session_id, compute_head=False)

        # Transmit activation to Node 1 via relay
        send_tensor(sock, output)

        # Receive prefill token from Node 1
        if receiver is not None:
            next_token, _, is_eos, _ = receiver.get()
        else:
            next_token, _, is_eos = recv_token(sock)

        t_first_token = time.perf_counter()
        if next_token in eos_token_ids or is_eos:
            is_eos = True
        else:
            generated_tokens.append(next_token)
            token_history.append(next_token)

            word = tokenizer.decode([next_token], skip_special_tokens=True)
            if print_output and word:
                print(word, end="", flush=True)

            if token_callback is not None and word:
                token_callback(word, {
                    "tokens": 1,
                    "ttft_ms": (t_first_token - t_start) * 1000.0,
                    "tps": 0.0,
                    "total_drafted": 0,
                    "total_accepted": 0,
                })

        # Kick off first background draft job on cuda:1 immediately
        pending_draft_job = None
        if async_drafter and spec_k > 0 and not is_eos:
            pending_draft_job = async_drafter.submit(next_token, k=spec_k, temperature=temperature, top_k=top_k, top_p=top_p)

        # 2. Autoregressive Decode Loop
        step = 1
        current_round_counter = 1
        pending_responses = {}
        invalidated_rounds = set()

        def fetch_response(target_round_id: int):
            while target_round_id not in pending_responses:
                if receiver is not None:
                    tok, acc, eos, stats = receiver.get()
                else:
                    tok, acc, eos, stats = recv_token_timed(sock)
                r_id = stats.get("round_id", 0)
                is_stale = stats.get("is_stale_discard", False)
                if r_id in invalidated_rounds or is_stale:
                    invalidated_rounds.discard(r_id)
                    continue
                pending_responses[r_id] = (tok, acc, eos, stats)
            res = pending_responses.pop(target_round_id)
            return res

        while step < max_tokens and not is_eos:
            if next_token in eos_token_ids:
                break

            if spec_k > 0:
                t_step_0 = time.perf_counter()
                t_draft_0 = time.perf_counter()
                drafts = []
                draft_gen_ms = 0.0
                draft_wait_ms = 0.0

                if ngram_sampler and spec_k > 0:
                    drafts = ngram_sampler.find_candidates(token_history, k=spec_k)
                    draft_gen_ms = (time.perf_counter() - t_draft_0) * 1000.0
                    draft_wait_ms = 0.0
                elif draft_sampler and spec_k > 0:
                    drafts = draft_sampler.generate_drafts(
                        next_token,
                        k=spec_k,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    draft_gen_ms = (time.perf_counter() - t_draft_0) * 1000.0
                    draft_wait_ms = draft_gen_ms

                if drafts:
                    total_drafted += len(drafts)
                    candidate_tokens = [next_token] + drafts
                else:
                    candidate_tokens = [next_token]

                cand_tensor = torch.tensor([candidate_tokens], dtype=torch.long, device=node.model_slice.device)
                if node.model_slice.embed_tokens is not None:
                    cand_hidden = node.model_slice.embed_tokens(cand_tensor)
                else:
                    cand_hidden = cand_tensor

                cache = node.kv_store.get(session_id)
                past_seq_len = node._get_cache_seq_len(cache)

                t_fwd_0 = time.perf_counter()
                cand_output = node._forward(cand_hidden, session_id=session_id, compute_head=False)
                if cand_output.is_cuda:
                    torch.cuda.synchronize(cand_output.device)
                t_fwd_1 = time.perf_counter()

                current_round_counter += 1
                primary_round_id = current_round_counter
                send_stats = send_tensor_timed(sock, cand_output, draft_tokens=drafts, round_id=primary_round_id, parent_round_id=0)

                next_token, accepted_count, is_eos, recv_stats = fetch_response(primary_round_id)
                total_accepted += max(0, accepted_count - 1)

                committed_len = past_seq_len + accepted_count
                if cache is not None:
                    rewind_kv_cache(cache, committed_len)

                if draft_sampler is not None:
                    draft_sampler.rewind(committed_len)

                hit_eos = False
                if drafts and accepted_count > 1:
                    for d_idx in range(accepted_count - 1):
                        tok = drafts[d_idx]
                        if tok in eos_token_ids:
                            hit_eos = True
                            is_eos = True
                            break
                        generated_tokens.append(tok)
                        token_history.append(tok)
                        w = tokenizer.decode([tok], skip_special_tokens=True)
                        if print_output and w:
                            print(w, end="", flush=True)
                        if token_callback is not None and w:
                            cur_time = time.perf_counter()
                            cur_decode = (cur_time - t_first_token) if t_first_token else 0.0
                            cur_count = len(generated_tokens)
                            cur_tps = (cur_count - 1) / cur_decode if (cur_decode > 0 and cur_count > 1) else 0.0
                            token_callback(w, {
                                "tokens": cur_count,
                                "ttft_ms": (t_first_token - t_start) * 1000.0 if t_first_token else 0.0,
                                "tps": cur_tps,
                                "total_drafted": total_drafted,
                                "total_accepted": total_accepted,
                            })

                if not hit_eos:
                    if next_token in eos_token_ids or is_eos:
                        is_eos = True
                    else:
                        generated_tokens.append(next_token)
                        token_history.append(next_token)
                        w = tokenizer.decode([next_token], skip_special_tokens=True)
                        if print_output and w:
                            print(w, end="", flush=True)
                        if token_callback is not None and w:
                            cur_time = time.perf_counter()
                            cur_decode = (cur_time - t_first_token) if t_first_token else 0.0
                            cur_count = len(generated_tokens)
                            cur_tps = (cur_count - 1) / cur_decode if (cur_decode > 0 and cur_count > 1) else 0.0
                            token_callback(w, {
                                "tokens": cur_count,
                                "ttft_ms": (t_first_token - t_start) * 1000.0 if t_first_token else 0.0,
                                "tps": cur_tps,
                                "total_drafted": total_drafted,
                                "total_accepted": total_accepted,
                            })

                step += accepted_count
                t_step_1 = time.perf_counter()
                if profiler is not None:
                    fwd_bk = getattr(node, "last_forward_breakdown", {})
                    profiler.record(
                        embed_ms=0.0,
                        gpu_fwd_ms=(t_fwd_1 - t_fwd_0) * 1000.0,
                        g2c_ms=send_stats["gpu_to_cpu_ms"],
                        ser_ms=send_stats["serialize_ms"],
                        send_ms=send_stats["tcp_send_ms"],
                        recv_ms=recv_stats.get("tcp_recv_ms", 0.0),
                        total_ms=(t_step_1 - t_step_0) * 1000.0,
                        draft_gen_ms=draft_gen_ms,
                        draft_wait_ms=draft_wait_ms,
                        gpu0_ms=fwd_bk.get("gpu0_ms", 0.0),
                        pcie_ms=fwd_bk.get("pcie_ms", 0.0),
                        gpu1_ms=fwd_bk.get("gpu1_ms", 0.0),
                        node1_compute_ms=recv_stats.get("node1_compute_ms", 0.0),
                        network_rtt_ms=recv_stats.get("network_rtt_ms", 0.0),
                        bubble_ms=0.0 if (t_fwd_1 == t_fwd_0) else (t_fwd_1 - t_fwd_0) * 1000.0,
                        accepted=accepted_count,
                        drafted=len(drafts),
                        is_spec=True,
                    )
                if is_eos:
                    break

            else:
                # Standard 1-token decode with precise phase timing
                t_emb_0 = time.perf_counter()
                tok_tensor = torch.tensor([[next_token]], dtype=torch.long, device=node.model_slice.device)
                if node.model_slice.embed_tokens is not None:
                    hidden = node.model_slice.embed_tokens(tok_tensor)
                else:
                    hidden = tok_tensor
                t_emb_1 = time.perf_counter()

                t_fwd_0 = time.perf_counter()
                output = node._forward(hidden, session_id=session_id, compute_head=False)
                if output.is_cuda:
                    torch.cuda.synchronize(output.device)
                t_fwd_1 = time.perf_counter()

                current_round_counter += 1
                send_stats = send_tensor_timed(sock, output, round_id=current_round_counter, parent_round_id=0)
                next_token, _, is_eos, recv_stats = fetch_response(current_round_counter)
                
                if is_eos or next_token in eos_token_ids:
                    is_eos = True
                    break

                generated_tokens.append(next_token)
                token_history.append(next_token)
                w = tokenizer.decode([next_token], skip_special_tokens=True)
                if print_output and w:
                    print(w, end="", flush=True)
                if token_callback is not None and w:
                    cur_time = time.perf_counter()
                    cur_decode = (cur_time - t_first_token) if t_first_token else 0.0
                    cur_count = len(generated_tokens)
                    cur_tps = (cur_count - 1) / cur_decode if (cur_decode > 0 and cur_count > 1) else 0.0
                    token_callback(w, {
                        "tokens": cur_count,
                        "ttft_ms": (t_first_token - t_start) * 1000.0 if t_first_token else 0.0,
                        "tps": cur_tps,
                        "total_drafted": total_drafted,
                        "total_accepted": total_accepted,
                    })
                
                t_step_1 = time.perf_counter()
                step += 1

                if profiler is not None:
                    fwd_bk = getattr(node, "last_forward_breakdown", {})
                    profiler.record(
                        embed_ms=(t_emb_1 - t_emb_0) * 1000.0,
                        gpu_fwd_ms=(t_fwd_1 - t_fwd_0) * 1000.0,
                        g2c_ms=send_stats["gpu_to_cpu_ms"],
                        ser_ms=send_stats["serialize_ms"],
                        send_ms=send_stats["tcp_send_ms"],
                        recv_ms=recv_stats.get("tcp_recv_ms", 0.0),
                        total_ms=(t_step_1 - t_step_0) * 1000.0,
                        draft_gen_ms=draft_gen_ms,
                        gpu0_ms=fwd_bk.get("gpu0_ms", 0.0),
                        pcie_ms=fwd_bk.get("pcie_ms", 0.0),
                        gpu1_ms=fwd_bk.get("gpu1_ms", 0.0),
                        node1_compute_ms=recv_stats.get("node1_compute_ms", 0.0),
                        network_rtt_ms=recv_stats.get("network_rtt_ms", 0.0),
                        bubble_ms=(t_fwd_1 - t_fwd_0) * 1000.0,
                        accepted=1,
                        drafted=0,
                        is_spec=False,
                    )

    except TimeoutError as te:
        if print_output:
            print(f"\n[ERROR] [TIMEOUT ERROR]: {te}", flush=True)
            print("Kaggle Node 1 or EC2 Relay stopped responding. Please check Kaggle B status.", flush=True)
    except ConnectionError as ce:
        if print_output:
            print(f"\n[ERROR] [CONNECTION ERROR]: {ce}", flush=True)
    except Exception as ex:
        if print_output:
            print(f"\n[ERROR] [UNEXPECTED ERROR]: {ex}", flush=True)
    finally:
        node.kv_store.evict(session_id)

    t_end = time.perf_counter()
    total_time = t_end - t_start
    ttft = (t_first_token - t_start) if t_first_token else total_time
    decode_time = (t_end - t_first_token) if t_first_token else total_time
    tok_count = len(generated_tokens)
    tps = (tok_count - 1) / decode_time if (decode_time > 0 and tok_count > 1) else (tok_count / decode_time if decode_time > 0 else 0)

    final_stats = {
        "tokens": tok_count,
        "ttft_ms": ttft * 1000.0,
        "decode_time_s": decode_time,
        "total_time_s": total_time,
        "tps": tps,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
    }
    if token_callback is not None:
        token_callback("", final_stats)

    if print_output:
        print("\n" + "-" * 55, flush=True)
        stats_str = f" Tokens: {tok_count} | TTFT: {ttft*1000:.1f} ms | Decode Time: {decode_time:.2f} s | Speed: {tps:.2f} TPS "
        if total_drafted > 0:
            accept_rate = (total_accepted / total_drafted) * 100.0
            stats_str += f" | Draft Accept Rate: {accept_rate:.1f}% ({total_accepted}/{total_drafted})"
        print(stats_str, flush=True)
        print("-" * 55, flush=True)

    return {
        "tokens": tok_count,
        "ttft": ttft,
        "decode_time": decode_time,
        "tps": tps,
        "draft_accept_rate": (total_accepted / total_drafted * 100.0) if total_drafted > 0 else None,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
    }


def format_chat_prompt(tokenizer, message: str, history: list, system_prompt: str = "") -> str:
    """Format conversation turns and system prompt using model's chat template or ChatML fallback."""
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    if history:
        for item in history:
            if isinstance(item, dict):
                if "role" in item and "content" in item and item["content"]:
                    messages.append({"role": item["role"], "content": item["content"]})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                u, a = item
                if u:
                    messages.append({"role": "user", "content": str(u)})
                if a:
                    messages.append({"role": "assistant", "content": str(a)})

    messages.append({"role": "user", "content": message})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    # ChatML fallback
    prompt_parts = []
    for m in messages:
        prompt_parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")
    return "\n".join(prompt_parts)


def launch_gradio_ui(
    tokenizer,
    node: PipelineNode,
    sock: socket.socket,
    args,
    eos_id: int,
    receiver: Optional[AsyncTokenReceiver],
    ngram_sampler: Optional[NGramDraftSampler],
    model_path: str,
    layer_start: int,
    layer_end: int,
    total_layers: int,
):
    """Launch interactive real-time streaming Gradio Chat Web UI."""
    if gr is None:
        print("\n[ERROR] Gradio is not installed. Please run 'pip install gradio' or pass --cli for terminal mode.")
        return

    logger.info("Initializing Gradio Web Chat UI...")

    def chat_fn(
        message: str,
        history: list,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        spec_k: int,
    ):
        if not message or not message.strip():
            return

        formatted_prompt = format_chat_prompt(tokenizer, message.strip(), history, system_prompt)

        token_q = queue.Queue()
        done_event = threading.Event()

        def _callback(token_str: str, stats: dict):
            token_q.put((token_str, stats))

        def _worker():
            try:
                generate(
                    prompt=formatted_prompt,
                    tokenizer=tokenizer,
                    node=node,
                    sock=sock,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    spec_k=int(spec_k),
                    ngram_sampler=ngram_sampler,
                    eos_token_id=eos_id,
                    profiler=None,
                    receiver=receiver,
                    enable_async_spec=args.async_spec,
                    spec_window=args.spec_window,
                    token_callback=_callback,
                    print_output=True,
                )
            except Exception as e:
                logger.error("Generation error: %s", e)
                token_q.put((f"\n\n[Generation Error: {e}]", {}))
            finally:
                done_event.set()

        worker_thread = threading.Thread(target=_worker, daemon=True)
        worker_thread.start()

        accumulated_text = ""
        last_stats = {}
        while not done_event.is_set() or not token_q.empty():
            try:
                token_str, stats = token_q.get(timeout=0.03)
                if stats:
                    last_stats.update(stats)
                if token_str:
                    accumulated_text += token_str
                    yield accumulated_text
            except queue.Empty:
                continue

        worker_thread.join(timeout=1.0)

        # Append telemetry performance footer to the message
        if last_stats and last_stats.get("tokens", 0) > 0:
            tps = last_stats.get("tps", 0.0)
            ttft = last_stats.get("ttft_ms", 0.0)
            toks = last_stats.get("tokens", 0)
            dec_time = last_stats.get("decode_time_s", 0.0)
            drafted = last_stats.get("total_drafted", 0)
            accepted = last_stats.get("total_accepted", 0)

            footer_items = [
                f"⚡ **Speed:** `{tps:.2f} TPS`",
                f"⏱️ **TTFT:** `{ttft:.1f} ms`",
                f"📊 **Tokens:** `{toks}` in `{dec_time:.2f}s`" if dec_time > 0 else f"📊 **Tokens:** `{toks}`",
            ]
            if drafted > 0:
                acc_rate = (accepted / drafted) * 100.0
                footer_items.append(f"🎯 **Draft Hit:** `{acc_rate:.1f}%` ({accepted}/{drafted})")

            footer_str = "\n\n---\n" + " • ".join(footer_items)
            yield accumulated_text + footer_str

    # Styling
    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
    )

    title = "⚡ ShardFlow Distributed LLM Chat"
    description = f"""
**Distributed Topology Active:**
- **Model:** `{model_path}` ({total_layers} total layers)
- **Node 0 Layers:** `[{layer_start}..{layer_end})` + Embeddings
- **Transport:** Direct TCP Relay (`{args.relay_host}:{args.relay_port}`)
- **Speculative Acceleration:** Active (K={args.spec_k})
"""

    additional_inputs = [
        gr.Textbox(
            value=getattr(args, "system_prompt", None) or "You are a helpful and concise AI assistant.",
            label="System Prompt",
            lines=2,
        ),
        gr.Slider(
            minimum=16,
            maximum=2048,
            value=args.max_tokens or 256,
            step=16,
            label="Max Tokens",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=1.5,
            value=0.0,
            step=0.05,
            label="Temperature (0.0 = Greedy)",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=1.0,
            step=0.05,
            label="Top-P (Nucleus Sampling)",
        ),
        gr.Slider(
            minimum=0,
            maximum=8,
            value=args.spec_k,
            step=1,
            label="Speculative Draft Depth (K)",
        ),
    ]

    demo = gr.ChatInterface(
        fn=chat_fn,
        title=title,
        description=description,
        additional_inputs=additional_inputs,
        additional_inputs_accordion=gr.Accordion("⚙️ Inference & Speculative Settings", open=False),
        theme=custom_theme,
    )

    print("\n" + "=" * 70)
    print("🚀 Launching ShardFlow Gradio Chat Web UI...")
    print("=" * 70, flush=True)

    demo.queue().launch(
        share=getattr(args, "share", True),
        server_name="0.0.0.0",
        server_port=getattr(args, "ui_port", 7860),
        inline=True,
    )


def run_interactive_cli(
    tokenizer,
    node: PipelineNode,
    sock: socket.socket,
    args,
    eos_id: int,
    receiver: Optional[AsyncTokenReceiver],
    ngram_sampler: Optional[NGramDraftSampler],
):
    """Run interactive continuous prompt CLI loop in terminal."""
    print("\n" + "=" * 70)
    print("💬 SHARDFLOW INTERACTIVE CHAT CONSOLE")
    print("Type your prompt and press Enter. Commands: /clear, /chat, /tokens <N>, /temp <T>, /spec <K>, /stats, /exit")
    print("=" * 70 + "\n", flush=True)

    history = []
    multiturn = True
    max_tokens = args.max_tokens or 256
    temperature = 0.0
    top_p = 1.0
    spec_k = args.spec_k
    show_stats = False
    system_prompt = getattr(args, "system_prompt", None) or "You are a helpful and concise AI assistant."

    while True:
        try:
            mode_tag = "Multi-Turn" if multiturn else "Single-Turn"
            user_input = input(f"\n[{mode_tag}] You > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit", "/q", "exit", "quit"):
                print("Exiting ShardFlow chat. Goodbye!")
                break
            elif user_input.lower() in ("/clear", "/reset"):
                history = []
                print("[OK] Conversation history and session KV cache cleared.")
                continue
            elif user_input.lower() == "/chat":
                multiturn = not multiturn
                print(f"[OK] Multi-turn chat mode: {'ENABLED' if multiturn else 'DISABLED (Single-turn QA)'}")
                continue
            elif user_input.lower() == "/stats":
                show_stats = not show_stats
                print(f"[OK] Detailed latency breakdown: {'ENABLED' if show_stats else 'DISABLED'}")
                continue
            elif user_input.lower().startswith("/tokens"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1].isdigit():
                    max_tokens = int(parts[1])
                    print(f"[OK] Max tokens updated to {max_tokens}.")
                else:
                    print(f"Current max tokens: {max_tokens}. Usage: /tokens <N>")
                continue
            elif user_input.lower().startswith("/temp"):
                parts = user_input.split()
                try:
                    temperature = float(parts[1])
                    print(f"[OK] Temperature updated to {temperature:.2f}.")
                except Exception:
                    print(f"Current temperature: {temperature}. Usage: /temp <float>")
                continue
            elif user_input.lower().startswith("/spec"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1].isdigit():
                    spec_k = int(parts[1])
                    print(f"[OK] Speculative K updated to {spec_k}.")
                else:
                    print(f"Current Speculative K: {spec_k}. Usage: /spec <K>")
                continue
            elif user_input.lower() == "/help":
                print("\nAvailable Commands:")
                print("  /clear, /reset   : Clear conversation history and reset KV cache")
                print("  /chat            : Toggle between Multi-Turn and Single-Turn mode")
                print("  /tokens <N>      : Set max generation tokens (e.g. /tokens 256)")
                print("  /temp <T>        : Set sampling temperature (e.g. /temp 0.7)")
                print("  /spec <K>        : Set speculative draft lookahead (e.g. /spec 4)")
                print("  /stats           : Toggle detailed per-token profiler breakdown")
                print("  /exit, /quit, /q : Disconnect and exit")
                continue

            current_history = history if multiturn else []
            formatted_prompt = format_chat_prompt(tokenizer, user_input, current_history, system_prompt)

            prompt_prof = Node0Profiler() if show_stats else None
            stats = generate(
                prompt=formatted_prompt,
                tokenizer=tokenizer,
                node=node,
                sock=sock,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                spec_k=spec_k,
                ngram_sampler=ngram_sampler,
                eos_token_id=eos_id,
                profiler=prompt_prof,
                receiver=receiver,
                enable_async_spec=args.async_spec,
                spec_window=args.spec_window,
                print_output=True,
            )

            if show_stats and prompt_prof:
                prompt_prof.print_breakdown()

            if multiturn:
                history.append({"role": "user", "content": user_input})
                # We can track assistant response if needed

        except KeyboardInterrupt:
            print("\n[Interrupted generation]")
            continue
        except Exception as e:
            print(f"\n[Error during interaction: {e}]")


def main():
    parser = argparse.ArgumentParser(description="ShardFlow v2 Node 0 (Kaggle A TCP Relay Runner)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="Model path or HF ID (default: Qwen/Qwen2.5-14B-Instruct)")
    parser.add_argument("--draft-model", default=None, help="Draft model path for speculative decoding (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--draft-device", default=None, help="Device for draft model (e.g. cuda:1 to run on secondary T4 GPU)")
    parser.add_argument("--spec-k", type=int, default=0, help="Speculative candidate tokens (default: 0 to disable, 4 to test)")
    parser.add_argument("--layer-start", type=int, default=0, help="Starting layer index (default: 0)")
    parser.add_argument("--layer-end", type=int, default=None, help="Ending layer index (default: half of total layers, e.g. 24 for 14B)")
    parser.add_argument("--4bit", "--load-in-4bit", dest="4bit", action="store_true", help="Enable 4-bit NF4 loading")
    parser.add_argument("--relay-host", default=RELAY_HOST, help=f"EC2 Relay IP (default: {RELAY_HOST})")
    parser.add_argument("--relay-port", type=int, default=RELAY_PORT, help=f"EC2 Relay Port (default: {RELAY_PORT})")
    parser.add_argument("--device", default="cuda", help="Target device (default: cuda)")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16", help="Precision (default: float16)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per generation (default: 256)")
    parser.add_argument("--prompt", default=None, help="Single prompt to execute directly (optional)")
    parser.add_argument("--benchmark", action="store_true", default=False, help="Run fixed 3-prompt benchmark evaluation suite")
    parser.add_argument("--web-ui", action="store_true", default=False, help="Launch Gradio Web Chat UI")
    parser.add_argument("--cli", action="store_true", default=False, help="Run in interactive CLI terminal mode")
    parser.add_argument("--share", action="store_true", default=True, help="Enable public gradio.live link (default: True)")
    parser.add_argument("--no-share", action="store_false", dest="share", help="Disable public gradio.live link")
    parser.add_argument("--ui-port", type=int, default=7860, help="Gradio server port (default: 7860)")
    parser.add_argument("--system-prompt", default="You are a helpful and concise AI assistant.", help="Default system prompt for chat")
    parser.add_argument("--cuda-graphs", action="store_true", default=False, help="Enable CUDA Graphs and Static KV cache")
    parser.add_argument("--no-cuda-graphs", action="store_false", dest="cuda_graphs", help="Disable CUDA Graphs and use DynamicCache fallback (default)")
    parser.add_argument("--static-kv", action="store_true", default=False, help="Enable Static KV cache on GPU")
    parser.add_argument("--async-recv", action="store_true", default=False, help="Enable asynchronous token receiver thread")
    parser.add_argument("--async-spec", action="store_true", default=False, help="Enable one-step-ahead speculative execution during network flight")
    parser.add_argument("--spec-window", type=int, default=1, help="In-flight speculative window depth (default: 1, e.g. 1, 2, 3)")
    args = parser.parse_args()

    model_path = args.model if os.path.exists(args.model) else args.model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    total_layers = get_num_hidden_layers(config, default=64)

    layer_start = args.layer_start
    layer_end = args.layer_end if args.layer_end is not None else (total_layers // 2)

    target_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    if target_dtype == torch.bfloat16 and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            logger.warning("GPU does not support native BF16 (T4 is CC 7.5). Overriding to torch.float16.")
            target_dtype = torch.float16

    use_cuda = (args.device == "cuda" or (isinstance(args.device, str) and args.device.startswith("cuda"))) and torch.cuda.is_available()
    enable_cuda_graphs = args.cuda_graphs and use_cuda
    enable_static_kv = (args.static_kv or enable_cuda_graphs) and use_cuda

    ngram_sampler = None
    if args.spec_k > 0 and not args.draft_model:
        ngram_sampler = NGramDraftSampler(max_ngram_size=3, min_ngram_size=1, spec_k=args.spec_k)
        draft_info = f"Prompt-Lookup N-gram (<0.1ms CPU lookup) (Speculative K={args.spec_k})"
    elif args.spec_k > 0 and args.draft_model:
        draft_dev_str = f" on {args.draft_device}" if args.draft_device else ""
        draft_info = f"Neural Draft ({args.draft_model}{draft_dev_str}) (Speculative K={args.spec_k})"
    else:
        draft_info = "DISABLED (spec_k=0)"

    use_async_recv = args.async_recv or args.async_spec
    print("=" * 70, flush=True)
    print(" SHARDFLOW v2 REMOTE NODE 0 (KAGGLE INSTANCE A)", flush=True)
    print(f"Base Model:    {model_path}")
    print(f"Layer Range:   [{layer_start}..{layer_end}) -> Indices {layer_start}..{layer_end-1} ({layer_end - layer_start}/{total_layers} layers + Embeddings)")
    print(f"Draft Model:   {draft_info}")
    print(f"Async Spec:    {'ENABLED ' if args.async_spec else 'DISABLED'}")
    print(f"Async Recv:    {'ENABLED ' if use_async_recv else 'DISABLED'}")
    print(f"CUDA Graphs:   {'ENABLED ' if enable_cuda_graphs else 'DISABLED (eager mode)'}")
    print(f"Static KV:     {'ENABLED (GPU StaticCache)' if enable_static_kv else 'DISABLED (DynamicCache)'}")
    print(f"Relay Target:  {args.relay_host}:{args.relay_port}")
    print(f"Precision:     {target_dtype}")
    print(f"Device:        {args.device}")
    print("=" * 70, flush=True)

    # 1. Load Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    eos_id = getattr(tokenizer, "eos_token_id", 151643)

    # 2. Load Model Slice (Layers 0..layer_end + Embedding Table)
    logger.info("Loading model shard (layers %d..%d + Embeddings)...", layer_start, layer_end)
    t0 = time.perf_counter()
    model_slice = load_layer_slice(
        model_path=model_path,
        layer_start=layer_start,
        layer_end=layer_end,
        device=args.device,
        dtype=target_dtype,
        load_in_4bit=getattr(args, "4bit", False),
    )
    logger.info("[OK] Model slice loaded in %.2f s", time.perf_counter() - t0)

    # 3. Initialize Pipeline Node & Draft Sampler
    node = PipelineNode(
        model_slice=model_slice,
        is_first_node=True,
        is_last_node=False,
        draft_model=args.draft_model if args.spec_k > 0 else None,
        draft_device=args.draft_device,
        spec_k=args.spec_k,
        enable_cuda_graphs=enable_cuda_graphs,
    )
    node.kv_store.enable_static_cache = enable_static_kv

    if model_slice.config is not None and enable_static_kv:
        node.kv_store.initialize_static_pool(
            config=model_slice.config,
            device=model_slice.device,
            dtype=target_dtype,
        )
        if node.kv_store._static_slots and enable_cuda_graphs:
            logger.info("Capturing CUDA Graphs on Node 0...")
            captured = node.graph_runner.capture(node.kv_store._static_slots[0].cache)
            if captured:
                logger.info("[OK] CUDA Graphs captured & active on Node 0!")

    # 4. Connect to Relay and perform handshake with Node 1
    logger.info("Connecting to TCP relay at %s:%d ...", args.relay_host, args.relay_port)
    sock = connect_to_relay(host=args.relay_host, port=args.relay_port, auth_byte=AUTH_BYTE)
    logger.info("[OK] Connected to relay. Executing READY handshake with Node 1...")
    handshake(sock)
    logger.info(" HANDSHAKE COMPLETE! Cluster is paired and ready for inference.")

    receiver = AsyncTokenReceiver(sock) if use_async_recv else None

    # 5. Determine Execution Mode
    try:
        if args.prompt is not None:
            # Mode A: Single Prompt Execution
            print(f"\n" + "=" * 60)
            print(" EXECUTING SINGLE PROMPT")
            print("=" * 60)
            generate(
                prompt=args.prompt,
                tokenizer=tokenizer,
                node=node,
                sock=sock,
                max_tokens=args.max_tokens,
                temperature=0.0,
                spec_k=args.spec_k,
                ngram_sampler=ngram_sampler,
                eos_token_id=eos_id,
                profiler=Node0Profiler(),
                receiver=receiver,
                enable_async_spec=args.async_spec,
                spec_window=args.spec_window,
                print_output=True,
            )

        elif args.benchmark:
            # Mode B: Automated Benchmark Mode
            prompts = [
                "Explain quantum entanglement in simple terms.",
                "Write a Python function to compute Fibonacci numbers using dynamic programming.",
                "What are the key advantages of pipeline parallelism for distributed LLM inference?",
            ]
            tps_results = []
            ttft_results = []
            global_profiler = Node0Profiler()

            for idx, prompt in enumerate(prompts, 1):
                print(f"\n" + "=" * 60)
                print(f" BENCHMARK PROMPT {idx}/{len(prompts)}")
                print("=" * 60)

                prompt_profiler = Node0Profiler()
                stats = generate(
                    prompt=prompt,
                    tokenizer=tokenizer,
                    node=node,
                    sock=sock,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    spec_k=args.spec_k,
                    ngram_sampler=ngram_sampler,
                    eos_token_id=eos_id,
                    profiler=prompt_profiler,
                    receiver=receiver,
                    enable_async_spec=args.async_spec,
                    spec_window=args.spec_window,
                    print_output=True,
                )
                if stats["tokens"] > 1:
                    tps_results.append(stats["tps"])
                    ttft_results.append(stats["ttft"])
                    for i in range(len(prompt_profiler.total_step_times)):
                        global_profiler.record(
                            embed_ms=prompt_profiler.embed_times[i],
                            gpu_fwd_ms=prompt_profiler.node0_gpu_times[i],
                            g2c_ms=prompt_profiler.gpu_to_cpu_times[i],
                            ser_ms=prompt_profiler.serialize_times[i],
                            send_ms=prompt_profiler.tcp_send_times[i],
                            recv_ms=prompt_profiler.tcp_recv_wait_times[i],
                            total_ms=prompt_profiler.total_step_times[i],
                            draft_gen_ms=prompt_profiler.draft_gen_times[i],
                            accepted=prompt_profiler.accepted_per_round[i],
                            drafted=prompt_profiler.drafted_per_round[i],
                            is_spec=prompt_profiler.is_spec_step[i],
                        )

                prompt_profiler.print_breakdown()

            if tps_results:
                print("\n" + "=" * 70)
                print("[BEST] FINAL BENCHMARK SUMMARY (ShardFlow v2 over Direct TCP Relay)")
                print(f"  Model:                 {model_path}")
                print(f"  Avg Decode Throughput: {statistics.mean(tps_results):.2f} tokens/sec ")
                print(f"  Max Decode Throughput: {max(tps_results):.2f} tokens/sec")
                print(f"  Avg TTFT:              {statistics.mean(ttft_results)*1000:.1f} ms")
                print(f"  Transport:             Direct TCP Relay ({args.relay_host}:{args.relay_port})")
                print("=" * 70)
                global_profiler.print_breakdown()

        elif args.cli or (gr is None and not args.web_ui):
            # Mode C: Interactive Terminal CLI
            run_interactive_cli(
                tokenizer=tokenizer,
                node=node,
                sock=sock,
                args=args,
                eos_id=eos_id,
                receiver=receiver,
                ngram_sampler=ngram_sampler,
            )

        else:
            # Mode D: Interactive Gradio Chat Web UI (Default)
            launch_gradio_ui(
                tokenizer=tokenizer,
                node=node,
                sock=sock,
                args=args,
                eos_id=eos_id,
                receiver=receiver,
                ngram_sampler=ngram_sampler,
                model_path=model_path,
                layer_start=layer_start,
                layer_end=layer_end,
                total_layers=total_layers,
            )

    finally:
        if receiver is not None:
            receiver.stop()
        sock.close()


if __name__ == "__main__":
    main()
