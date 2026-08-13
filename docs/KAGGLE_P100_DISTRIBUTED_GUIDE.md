# 🚀 ShardFlow Distributed Inference on Kaggle: Complete Technical Guide

This document provides a comprehensive technical breakdown of running ShardFlow's distributed speculative inference pipeline across separate Kaggle GPU instances (Tesla P100 / Tesla T4), documenting all forensic discoveries, architectural fixes, benchmarks, and operating rules.

---

## 1. System Architecture & Topology

ShardFlow partitions large LLMs (e.g. `Qwen/Qwen2.5-7B-Instruct`) across independent compute nodes communicating over high-speed binary TCP frames via reverse tunnels (`bore.pub` or WireGuard/Tailscale):

```
┌────────────────────────────────────────────────────────┐
│              Client / Local Gateway                   │
│        http://127.0.0.1:8000/v1/chat/completions       │
└──────────────────────────┬─────────────────────────────┘
                           │ TCP Stream
                           ▼
┌────────────────────────────────────────────────────────┐
│  Kaggle Instance 1 (Node 0 — Intermediate Node)        │
│  - GPU: 1x Tesla P100 (16GB VRAM)                      │
│  - Assigned Layers: [0, 14) + Embeddings Table         │
│  - Draft Model: Qwen2.5-0.5B-Instruct (Speculative K=4)│
│  - VRAM Footprint: ~7.11 GB (8.8 GB Free Headroom)     │
│  - Tunnel: bore.pub (e.g. bore.pub:55249)              │
└──────────────────────────┬─────────────────────────────┘
                           │ Activation Tensors [1, 5, 3584] (FP16)
                           ▼
┌────────────────────────────────────────────────────────┐
│  Kaggle Instance 2 (Node 1 — Terminal Node)            │
│  - GPU: 1x Tesla P100 (16GB VRAM)                      │
│  - Assigned Layers: [14, 28) + Final RMSNorm + LM Head │
│  - Static KV Pool: 4 Pre-allocated Slots (FP16)        │
│  - Speculative Verification Engine: Causal Top-1 Match  │
│  - VRAM Footprint: ~7.11 GB (8.8 GB Free Headroom)     │
│  - Tunnel: bore.pub (e.g. bore.pub:43396)              │
└────────────────────────────────────────────────────────┘
```

---

## 2. Forensic Discoveries & Root Causes

During development and stress-testing on Kaggle container instances, multiple subtle failure modes were investigated, measured with telemetry, and definitively resolved:

### A. The PyTorch Architecture Incompatibility (`sm_60` on Kaggle 2026 Image)
- **Symptom**: Kaggle's base Python 3.12 / PyTorch 2.5+ environment threw warnings on startup:
  `UserWarning: Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation (sm_70 to sm_120)`.
- **Cause**: PyTorch upstream dropped Pascal (`sm_60`) support from default wheels starting in PyTorch 2.5. Memory allocations worked, but compute kernels failed or aborted.
- **Resolution**: Installing PyTorch builds compiled with `sm_60` (or running on Turing `sm_75` T4 / Colab instances) completely resolves GPU kernel execution.

### B. Memory-Mapped Safetensors & Async DMA Staging Buffer Queue
- **Symptom**: When loading Shard 2 on Node 0 (which has 85 weight tensors), the container process was abruptly killed despite RAM showing 28 GB free and VRAM showing 8 GB free.
- **Cause**:
  1. `safe_open()` maintains an open memory-mapped (`mmap`) file descriptor on the container's `/dev/loop1` filesystem.
  2. In the loading loop, calling `current_param.data.copy_(tensor)` without synchronization scheduled 85 rapid asynchronous `cudaMemcpyAsync` DMA transfers. PyTorch's internal host allocator retained pinned host memory staging buffers until DMA completed. Rapidly queuing 3.5 GB of transfers hit container `RLIMIT_MEMLOCK` limits.
- **Resolution**:
  1. Replaced `safe_open` with `safetensors.torch.load_file(shard_path, device="cpu")` which reads the binary sequentially and closes the file descriptor immediately.
  2. Enforced synchronous DMA copying (`current_param.data.copy_(tensor, non_blocking=False)`).
  3. Added `torch.cuda.synchronize(target_device)` and `gc.collect()` after every shard.

### C. Registry Fallback Overwriting Layer Bounds (`is_first` / `is_last`)
- **Symptom**: Node 0 was marked as `is_first=True, is_last=True`, causing it to load the 1.09 GB `lm_head` and `norm` on top of layers `[0, 14)`, inflating VRAM to 8.2+ GB.
- **Cause**: When Node 0 registered with Render before Node 1 was online, Render's 120s timeout fallback treated the cluster as a 1-node topology (`[0, 28)`). The runner trusted the registry's `is_last_node` boolean.
- **Resolution**: Enforced deterministic layer derivation in [kaggle_runner.py](file:///home/adityaraut/Documents/Shardflow/scripts/kaggle_runner.py#L345) and [node.py](file:///home/adityaraut/Documents/Shardflow/shardflow/node/node.py#L990):
  ```python
  is_first = (layer_start == 0)
  is_last = (layer_end >= total_layers)
  ```
  Intermediate nodes (`[0, 14)`) are mathematically guaranteed to never allocate the LM head or RMSNorm.

### D. FP16 vs BF16 Hardware Compatibility
- **Symptom**: Qwen2.5 defaults to `bfloat16` in `config.json`. Tesla P100 (`sm_60`) does not have hardware ALUs for BF16, leading to software emulation overhead.
- **Resolution**: Added automatic hardware detection in `layer_loader.py` that casts `bfloat16` to `torch.float16` whenever `torch.cuda.is_bf16_supported()` is False. Both the main model, draft model, and StaticKVSlots run natively in FP16.

### E. Jupyter `ipykernel` Silent Output Timeout
- **Symptom**: Running `await node1.serve_forever()` in a notebook cell caused Kaggle's web frontend to disconnect after ~60 seconds of silent waiting.
- **Resolution**: Added a 10-second periodic health heartbeat logger inside `serve_forever()`:
  `Pipeline node live & healthy — listening on 0.0.0.0:9500 | VRAM=7.11 GB`.
  This maintains continuous frontend activity and provides real-time telemetry.

---

## 3. Verified Performance & Benchmark Results

With the pipeline deployed across 2x Kaggle Tesla P100 instances, live OpenAI-compatible chat completions were verified using `scripts/benchmark_tps.py`:

```
=================================================================
⚡ SHARDFLOW DISTRIBUTED INFERENCE BENCHMARK
Target:     http://127.0.0.1:8000/v1/chat/completions
Max Tokens: 100 | Runs: 3 | Speculative K: 4
=================================================================

--- Benchmark Run 1/3 ---
Prompt: "Explain the concept of quantum entanglement in simple terms."
Response: "Quantum entanglement is a phenomenon in physics where two or more particles become connected..."
-----------------------------------------------------------------
  TTFT (Time-to-First-Token):  284.1 ms
  Total Tokens Generated:      100
  Total Time:                  2.25 s
  Decode Time:                 1.97 s
  Decode Throughput:           50.36 TPS 🚀
-----------------------------------------------------------------

--- Benchmark Run 2/3 ---
  TTFT: 271.7 ms | Total Time: 2.24 s | Decode Throughput: 50.32 TPS 🚀

--- Benchmark Run 3/3 ---
  TTFT: 273.4 ms | Total Time: 2.24 s | Decode Throughput: 50.31 TPS 🚀

=================================================================
📊 BENCHMARK SUMMARY
=================================================================
  Avg Decode Throughput:   50.33 tokens/sec
  Max Decode Throughput:   50.36 tokens/sec
  Avg TTFT:                276.38 ms
  Total Generation Time:   2.24 seconds
  Cost:                    $0.00 (Free Tier)
=================================================================
```

---

## 4. Operational Instructions & Launch Commands

### 💻 Kaggle Notebook 1 (P100 — Node 0: Layers 0..14 + Draft Model)

```bash
%cd /kaggle/working/Shardflow
!git pull origin main

!python scripts/kaggle_runner.py \
    --model /kaggle/working/models/Qwen2.5-7B-Instruct \
    --draft-model /kaggle/working/models/Qwen2.5-0.5B-Instruct \
    --registry-url https://shardflow.onrender.com \
    --layer-start 0 \
    --layer-end 14 \
    --expected-nodes 2 \
    --dtype float16 \
    --spec-k 4 \
    --tunnel bore \
    --no-cuda-graphs \
    --port 9500
```

### 💻 Kaggle Notebook 2 (P100 — Node 1: Layers 14..28 + LM Head)

```bash
%cd /kaggle/working/Shardflow
!git pull origin main

!python scripts/kaggle_runner.py \
    --model /kaggle/working/models/Qwen2.5-7B-Instruct \
    --registry-url https://shardflow.onrender.com \
    --layer-start 14 \
    --layer-end 28 \
    --expected-nodes 2 \
    --dtype float16 \
    --tunnel bore \
    --no-cuda-graphs \
    --port 9500
```

---

## 5. Critical Rules to Remember

1. **Explicit Cache Directory**: Always set `os.environ["HF_HOME"] = "/kaggle/working/hf_home"`. The root filesystem (`/`) has only 1.6 GB and will trigger `SIGKILL` if caches write there.
2. **Explicit Layer Bounds**: Always pass `--layer-start` and `--layer-end` when running multi-notebook setups to guarantee strict 7.11 GB slice isolation regardless of registry timing.
3. **Execution Order**: Always start **Node 1 (Notebook 2)** first so its public tunnel is live when Node 0 registers and connects.
4. **Standalone Process**: Launch runners with `!python scripts/kaggle_runner.py` (or Python `subprocess`) to avoid holding Jupyter kernel variable references that duplicate VRAM.

---

## 6. Forensic Diagnostic & Binary Split Suite

If Node 1 or Node 0 encounters an unexpected disconnect or silent crash on Kaggle, use `scripts/diagnose_kaggle_p100.py` to systematically isolate the failure boundary in seconds:

### Step 1: Pure CUDA Forward Pass (No Server, No Network)
Verify whether PyTorch on Pascal `sm_60` executes the transformer decoder layers, RMSNorm, and LM head without CUDA kernel errors:
```bash
!python scripts/diagnose_kaggle_p100.py \
    --model /kaggle/working/models/Qwen2.5-7B-Instruct \
    --layer-start 14 \
    --layer-end 28 \
    --test forward \
    --device cuda
```
- **If this passes**: PyTorch SDPA (Math backend), RoPE embeddings, RMSNorm, Linear, and greedy sampling are 100% functional on the P100.

### Step 2: Isolated Server Idle Longevity (10+ Minutes)
Verify whether `PipelineNode` server, asyncio loop, and local socket bindings survive without dying when no external network traffic or tunnels are present:
```bash
!python scripts/diagnose_kaggle_p100.py \
    --model /kaggle/working/models/Qwen2.5-7B-Instruct \
    --layer-start 14 \
    --layer-end 28 \
    --test idle \
    --idle-seconds 600 \
    --device cuda
```
- **If this survives**: The Python runtime, asyncio event loop, and socket listener are stable. `serve_forever()` emits periodic 10-second heartbeats to prevent Jupyter kernel timeout.

### Step 3: Localhost Loopback Inference (Client → Server)
Verify the complete serialization → TCP wire transfer → deserialization → GPU inference → response pipeline over `127.0.0.1` without third-party tunnels:
```bash
!python scripts/diagnose_kaggle_p100.py \
    --model /kaggle/working/models/Qwen2.5-7B-Instruct \
    --layer-start 14 \
    --layer-end 28 \
    --test loopback \
    --device cuda
```
- **If this passes**: The entire ShardFlow data-plane protocol, tensor serialization, socket buffer handlers, and GPU execution are bug-free.

### Step 4: Bore Tunnel Diagnostic
Test the stability of the public `bore.pub` proxy and drain thread independently:
```bash
!python scripts/diagnose_kaggle_p100.py --test bore
```

### Forensic Decision Matrix

| Test 1 (CUDA) | Test 2 (Idle) | Test 3 (Loopback) | Test 4 (Bore) | Root Cause & Actionable Fix |
|---|---|---|---|---|
| ❌ Fails | - | - | - | **PyTorch sm_60 incompatibility**: Install sm_60-enabled PyTorch build or use T4 GPU. |
| ✅ Passes | ❌ Dies | - | - | **Kaggle container memory/watchdog**: Check host RAM limit or run via standalone python process. |
| ✅ Passes | ✅ Survives | ❌ Fails | - | **Tensor deserialization / dtype mismatch**: Check FP16 tensor framing and protocol version. |
| ✅ Passes | ✅ Survives | ✅ Passes | ❌ Drops | **Bore tunnel / Kaggle network policy**: Third-party TCP proxy connection dropped or reset. |

