#!/usr/bin/env python3
"""
ShardFlow — Draft Model Microbenchmark & Profiler.
Measures per-token latency of Qwen2.5-0.5B under:
1. Baseline eager mode
2. No-sync GPU argmax (single .tolist() at the end)
3. StaticCache (zero-allocation KV)
4. torch.compile()
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache, StaticCache


def profile_draft(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", k: int = 4, iters: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    print(f"Loading {model_id} on {device} ({dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    config = model.config
    prompt = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13]
    prompt_len = len(prompt)

    # 1. Baseline eager mode (current implementation)
    print("\n--- 1. Baseline (Current implementation) ---")
    cache = DynamicCache()
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.inference_mode():
        model(input_ids=input_ids, past_key_values=cache, use_cache=True)

    latencies_baseline = []
    for _ in range(iters):
        # Reset to prompt len
        cache.crop(prompt_len)
        next_tok = 1234
        t0 = time.perf_counter()
        drafts = []
        with torch.inference_mode():
            for _ in range(k):
                inp = torch.tensor([[next_tok]], dtype=torch.long, device=device)
                out = model(input_ids=inp, past_key_values=cache, use_cache=True)
                tok_id = out.logits[0, -1, :].argmax(dim=-1).item()
                drafts.append(tok_id)
                next_tok = tok_id
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_baseline.append((t1 - t0) * 1000.0)

    avg_base = sum(latencies_baseline) / len(latencies_baseline)
    print(f"Baseline K={k}: {avg_base:.2f} ms total ({avg_base/k:.2f} ms/token)")

    # 2. Optimized No-Sync GPU Argmax (Keep on GPU, .tolist() at the end)
    print("\n--- 2. No-Sync GPU Argmax ---")
    cache = DynamicCache()
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.inference_mode():
        model(input_ids=input_ids, past_key_values=cache, use_cache=True)

    latencies_nosync = []
    for _ in range(iters):
        cache.crop(prompt_len)
        next_tok_tensor = torch.tensor([[1234]], dtype=torch.long, device=device)
        draft_gpu_tokens = []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(k):
                out = model(input_ids=next_tok_tensor, past_key_values=cache, use_cache=True)
                next_tok_tensor = out.logits[:, -1:, :].argmax(dim=-1)
                draft_gpu_tokens.append(next_tok_tensor)
            # single CPU transfer at the very end
            drafts = [t[0, 0].item() for t in draft_gpu_tokens]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_nosync.append((t1 - t0) * 1000.0)

    avg_nosync = sum(latencies_nosync) / len(latencies_nosync)
    print(f"No-Sync GPU Argmax K={k}: {avg_nosync:.2f} ms total ({avg_nosync/k:.2f} ms/token)")

    # 3. StaticCache (Zero Allocations)
    print("\n--- 3. StaticCache (Pre-allocated KV) ---")
    static_cache = StaticCache(config=config, max_batch_size=1, max_cache_len=512, device=device, dtype=dtype)
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.inference_mode():
        model(input_ids=input_ids, past_key_values=static_cache, use_cache=True)

    latencies_static = []
    for _ in range(iters):
        static_cache._seen_tokens = prompt_len
        next_tok_tensor = torch.tensor([[1234]], dtype=torch.long, device=device)
        draft_gpu_tokens = []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(k):
                out = model(input_ids=next_tok_tensor, past_key_values=static_cache, use_cache=True)
                next_tok_tensor = out.logits[:, -1:, :].argmax(dim=-1)
                draft_gpu_tokens.append(next_tok_tensor)
            drafts = [t[0, 0].item() for t in draft_gpu_tokens]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_static.append((t1 - t0) * 1000.0)

    # 4. DraftSampler with CUDA Graph Replay
    print("\n--- 4. DraftSampler (StaticCache + CUDA Graph) ---")
    try:
        from shardflow.node.draft_model import DraftSampler
        sampler = DraftSampler(
            model_path=model_id,
            device=device,
            dtype=dtype,
            spec_k=k,
            enable_cuda_graphs=True,
        )
        sampler.prefill(prompt)

        latencies_graph = []
        for _ in range(iters):
            sampler.rewind(prompt_len)
            t0 = time.perf_counter()
            _ = sampler.generate_drafts(current_token=1234, k=k)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            latencies_graph.append((t1 - t0) * 1000.0)

        avg_graph = sum(latencies_graph) / len(latencies_graph)
        print(f"DraftSampler CUDA Graph K={k}: {avg_graph:.2f} ms total ({avg_graph/k:.2f} ms/token)")
        if avg_base > 0:
            print(f"Speedup vs Baseline: {avg_base/avg_graph:.2f}x")
    except Exception as e:
        print(f"DraftSampler CUDA Graph benchmark skipped: {e}")


if __name__ == "__main__":
    profile_draft()
