# ShardFlow

A general-purpose distributed inference framework that automatically partitions any HuggingFace transformer across N GPU machines, manages per-node KV caches, schedules concurrent requests through a micro-batch pipeline, and exposes a single OpenAI-compatible endpoint.

> **Status:** All build phases (Phase 1 to Phase 5) completed.  
> **Live API Gateway:** [https://shardflow.onrender.com](https://shardflow.onrender.com)

---

## System Architecture

```
Client (OpenAI SDK / Web App)
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
              │ Node 0: Layers 0-11      │ (Colab GPU 1 / Local GPU)
              │ (Per-node DynamicCache)  │
              └────────────┬─────────────┘
                           │ TCP: activations
                           ▼
              ┌──────────────────────────┐
              │ Node 1: Layers 11-22     │ (Colab GPU 2 / Local GPU)
              │ (+ LM Head & GPU Sampler)│
              └────────────┬─────────────┘
                           │ 8-Byte TOKEN_ID response
                           ▼
                      Orchestrator
```

### Layer Breakdown

- **Layer 1 — Client:** Standard OpenAI SDK or `curl` hitting `POST /v1/chat/completions`. Supports real-time SSE streaming (`stream=True`).
- **Layer 2 — API Gateway (FastAPI):** Exposes `/v1/chat/completions`, `/health`, `/metrics`, `/docs` (Swagger UI), and cancellation via `DELETE /v1/sessions/{id}`.
- **Layer 3 — Request Scheduler:** Async queue (`asyncio.Queue`) for pending requests, session tracking, concurrency control.
- **Layer 4 — Inference Orchestrator:** Tokenization, embedding, GPU sampling handler, and async TCP client (`asyncio.open_connection`).
- **Layer 5 — Topology Registry:** Solves Colab's no-static-IP problem (`POST /register`, `GET /topology` with auto-eviction).
- **Layer 6 — Pipeline Nodes:** Standalone Python processes loading layer slices via **Zero-RAM Meta-Device Slicing** (`accelerate.init_empty_weights`) with per-session `DynamicCache` and 60s background TTL eviction.

---

## Key Performance Optimizations

1. **Zero-RAM Meta-Device Loading**: Model skeletons are instantiated on PyTorch `meta` device in 0.00s with **0 MB CPU RAM overhead**, allocating memory *only* for the assigned layer slice to prevent CPU RAM OOMs on 14B/32B/70B models.
2. **GPU-Side Token Sampling**: Last node samples next token directly on GPU logits and returns an **8-byte `TOKEN_ID`** message, reducing back-propagation payload from 64 KB to 8 bytes.
3. **Zero-Copy Serialization**: Memoryviews (`torch.frombuffer`) and `socket.TCP_NODELAY` socket options for sub-millisecond network framing.

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/rautaditya2606/Shardflow.git
cd Shardflow
pip install -e ".[dev]"
```

### 2. Run Local 2-Node End-to-End Test

```bash
python tests/test_e2e_localhost.py \
  --model ./models/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Tell me a short story about a brave knight" \
  --max-tokens 30 \
  --num-nodes 2
```

### 3. Run Benchmark Suite

```bash
python benchmarks/run_benchmarks.py \
  --model ./models/TinyLlama-1.1B-Chat-v1.0 \
  --max-tokens 20 \
  --num-nodes 2
```

---

## OpenAI Python SDK Example

```python
from openai import OpenAI

# Connect to live ShardFlow Render Gateway
client = OpenAI(
    base_url="https://shardflow.onrender.com/v1",
    api_key="not-needed",
)

# Real-time SSE streaming
stream = client.chat.completions.create(
    model="tinyllama",
    messages=[{"role": "user", "content": "Explain cloud computing simply."}],
    max_tokens=40,
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Benchmark Results

Using TinyLlama 1.1B split across 2 pipeline nodes on RTX 3050 GPU / Localhost:

| Device | Framework / Strategy | Speed (tok/s) | Duration (20 tokens) |
|---|---|---|---|
| **CPU** | No KV Cache (Phase 1) | 0.1 tok/s | 258.5s |
| **CPU** | Initial DynamicCache baseline | 0.9 tok/s | 23.0s |
| **CPU** | **Optimized ShardFlow** (DynamicCache + Zero-Copy) | **1.12 tok/s** | **18.7s** |
| **RTX 3050 GPU** | Initial TCP Protocol baseline (64KB Logits Relay) | 1.8 tok/s | 11.75s |
| **RTX 3050 GPU** | **Optimized ShardFlow** (GPU Sampling + Zero-Copy + `TCP_NODELAY`) | **23.0 tok/s** | **0.91s (13x speedup)** |

---

## License

MIT Licenseontent": "Tell me a short story."}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

---

## Benchmark Results

Using TinyLlama 1.1B split across 2 pipeline nodes on localhost:

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
