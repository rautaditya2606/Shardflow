# ShardFlow

A general-purpose distributed LLM inference framework that automatically partitions any HuggingFace transformer across N GPU machines (e.g. Google Colab T4 GPUs, local GPUs, or cloud instances), manages per-node KV caches, schedules concurrent requests through a micro-batch pipeline, and exposes a single OpenAI-compatible endpoint.

> **Live API Gateway:** [https://shardflow.onrender.com](https://shardflow.onrender.com)  
> **Status:** Fully functional & verified for distributed multi-GPU inference across Google Colab accounts and local GPUs.

---

## System Architecture

```
Client (OpenAI SDK / Web App / curl)
          │
          ▼  POST https://shardflow.onrender.com/v1/chat/completions
┌─────────────────────────────────────────────────────────┐
│ Layer 2: API Gateway (FastAPI on Render)                │
├─────────────────────────────────────────────────────────┤
│ Layer 3: Request Scheduler (asyncio.Queue)             │
├─────────────────────────────────────────────────────────┤
│ Layer 4: Inference Orchestrator (Tokenizer, Embed, Samp)│
└──────────────────────────┬──────────────────────────────┘
                           │ TCP / bore.pub Tunnels
                           ▼
              ┌──────────────────────────┐
              │ Node 0: Layers 0-14      │ (Colab GPU Tab 1 - Account 1)
              │ (Per-node DynamicCache)  │
              └────────────┬─────────────┘
                           │ TCP: activations
                           ▼
              ┌──────────────────────────┐
              │ Node 1: Layers 14-28     │ (Colab GPU Tab 2 - Account 2)
              │ (+ LM Head & GPU Sampler)│
              └────────────┬─────────────┘
                           │ 8-Byte TOKEN_ID response
                           ▼
                      Orchestrator
```

### Layer Breakdown

- **Layer 1 — Client:** Standard OpenAI SDK or `curl` hitting `POST /v1/chat/completions`. Supports real-time SSE streaming (`stream=True`).
- **Layer 2 — API Gateway (FastAPI):** Exposes `/v1/chat/completions`, `/health`, `/metrics`, `/docs` (Swagger UI).
- **Layer 3 — Request Scheduler:** Async queue (`asyncio.Queue`) for pending requests, session tracking, concurrency control.
- **Layer 4 — Inference Orchestrator:** Tokenization, zero-weight memory loading, GPU sampling handler, and async TCP client.
- **Layer 5 — Topology Registry:** Solves Colab's dynamic IP problem (`POST /register`, `GET /topology` with heartbeats and auto-reregistration).
- **Layer 6 — Pipeline Nodes:** Standalone Python processes loading layer slices via **Zero-RAM Meta-Device Slicing** (`accelerate.init_empty_weights`) with per-session `DynamicCache` and 60s background TTL eviction.

---

## 🚀 Running on Google Colab (2 T4 GPUs)

Run an **8 Billion parameter model** (e.g. `Qwen/Qwen2.5-7B-Instruct`) split across two free Google Colab accounts.

### Step 1: Start Colab Tab 2 (Account 2 - Incognito Tab)
Run this in **Colab Tab 2** (owns **Layers 14 ➔ 28 + LM Head**):

```python
%cd /content
!rm -rf /content/Shardflow
!git clone https://github.com/rautaditya2606/Shardflow.git /content/Shardflow
%cd /content/Shardflow
!pip install -q -e .

!python scripts/colab_runner.py \
  --registry-url https://shardflow.onrender.com \
  --model Qwen/Qwen2.5-7B-Instruct \
  --node-id colab-node-1 \
  --port 9501 \
  --layer-start 14 \
  --layer-end 28
```
*Wait until it logs: `Node ready — layers [14, 28), LAST node (has LM head)`*

---

### Step 2: Start Colab Tab 1 (Account 1 - Normal Tab)
Run this in **Colab Tab 1** (owns **Layers 0 ➔ 14**):

```python
%cd /content
!rm -rf /content/Shardflow
!git clone https://github.com/rautaditya2606/Shardflow.git /content/Shardflow
%cd /content/Shardflow
!pip install -q -e .

!python scripts/colab_runner.py \
  --registry-url https://shardflow.onrender.com \
  --model Qwen/Qwen2.5-7B-Instruct \
  --node-id colab-node-0 \
  --port 9500 \
  --layer-start 0 \
  --layer-end 14
```
*Wait until it logs: `Node ready — layers [0, 14), INTERMEDIATE node`*

---

### Step 3: Call the OpenAI Endpoint

#### Option A: Using `curl`
```bash
curl https://shardflow.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Explain artificial intelligence in one simple sentence."}],
    "max_tokens": 40
  }'
```

#### Option B: Using OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://shardflow.onrender.com/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Explain cloud computing simply."}],
    max_tokens=40,
)

print(response.choices[0].message.content)
```

---

## Key Performance Optimizations

1. **Zero-RAM Meta-Device Loading**: Model skeletons are instantiated on PyTorch `meta` device in 0.00s with **0 MB CPU RAM overhead**, allocating memory *only* for the assigned layer slice to prevent CPU RAM OOMs on 14B/32B/70B models.
2. **GPU-Side Token Sampling**: Last node samples next token directly on GPU logits and returns an **8-byte `TOKEN_ID`** message, reducing back-propagation payload from 64 KB to 8 bytes.
3. **Zero-Copy Serialization**: Memoryviews (`torch.frombuffer`) and `socket.TCP_NODELAY` socket options for sub-millisecond network framing.
4. **Heartbeat Auto-Reregistration**: Nodes automatically re-register with the registry if the server process restarts.

---

## Local Development & Testing

### Installation

```bash
git clone https://github.com/rautaditya2606/Shardflow.git
cd Shardflow
pip install -e ".[dev]"
```

### Run Local E2E Pipeline Test

```bash
python scripts/test_local_real_server.py
```

---

## Benchmark Results

Using TinyLlama 1.1B split across 2 pipeline nodes:

| Device | Framework / Strategy | Speed (tok/s) | Duration (20 tokens) |
|---|---|---|---|
| **CPU** | No KV Cache (Phase 1) | 0.1 tok/s | 258.5s |
| **CPU** | Initial DynamicCache baseline | 0.9 tok/s | 23.0s |
| **CPU** | **Optimized ShardFlow** (DynamicCache + Zero-Copy) | **1.12 tok/s** | **18.7s** |
| **RTX 3050 GPU** | Initial TCP Protocol baseline (64KB Logits Relay) | 1.8 tok/s | 11.75s |
| **RTX 3050 GPU** | **Optimized ShardFlow** (GPU Sampling + Zero-Copy + `TCP_NODELAY`) | **23.0 tok/s** | **0.91s (13x speedup)** |

---

## License

MIT License
