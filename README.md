# ShardFlow

A general-purpose distributed inference framework that automatically partitions any HuggingFace transformer across N GPU machines, manages per-node KV caches, schedules concurrent requests through a micro-batch pipeline, and exposes a single OpenAI-compatible endpoint.

> **Status:** All build phases (Phase 1 to Phase 5) completed.

---

## System Architecture

```
Client (OpenAI SDK / curl)
          │
          ▼  POST /v1/chat/completions
┌─────────────────────────────────────────────────────────┐
│ Layer 2: API Gateway (FastAPI)                          │
├─────────────────────────────────────────────────────────┤
│ Layer 3: Request Scheduler (asyncio.Queue)             │
├─────────────────────────────────────────────────────────┤
│ Layer 4: Inference Orchestrator (Tokenizer, Embed, Samp)│
└──────────────────────────┬──────────────────────────────┘
                           │ TCP: length-prefix tensor framing
                           ▼
              ┌──────────────────────────┐
              │ Node 0: Layers 0-10      │
              │ (Per-node DynamicCache)  │
              └────────────┬─────────────┘
                           │ TCP: activations
                           ▼
              ┌──────────────────────────┐
              │ Node 1: Layers 11-21     │
              │ (+ Final Norm + LM Head) │
              └──────────────────────────┘
                           │ TCP: logits (backward flow)
                           ▼
                      Orchestrator
```

### Layer Breakdown

- **Layer 1 — Client:** Standard OpenAI SDK or `curl` hitting `POST /v1/chat/completions`. Supports SSE streaming.
- **Layer 2 — API Gateway (FastAPI):** Exposes `/v1/chat/completions`, handles SSE streaming back to client, cancellation via `DELETE /v1/sessions/{id}`, and `/metrics`.
- **Layer 3 — Request Scheduler:** Async queue (`asyncio.Queue`) for pending requests, session tracking, concurrency control.
- **Layer 4 — Inference Orchestrator:** Tokenization, embedding (CPU lookup), sampling (temperature, top-p, top-k, greedy), and async TCP client (`asyncio.open_connection`).
- **Layer 5 — Topology Registry (FastAPI):** Solves Colab's no-static-IP problem (`POST /register`, `GET /topology`, `POST /heartbeat` with 30s auto-eviction).
- **Layer 6 — Pipeline Nodes:** Standalone Python processes on GPU/Colab machines loading layer slices with `DynamicCache` per session and 60s background TTL eviction.

---

## Components & Services

| Service | Script / Entrypoint | Description |
|---|---|---|
| **Topology Registry** | `python -m shardflow.registry.app` | FastAPI registry for node discovery and health tracking |
| **API Gateway** | `python -m shardflow.gateway.app` | OpenAI-compatible FastAPI gateway (`/v1/chat/completions`) |
| **Pipeline Node** | `python -m shardflow.node.node` | Individual layer slice runner |
| **Orchestrator** | `python -m shardflow.orchestrator.orchestrator` | Central inference controller |
| **Colab Notebook** | `notebooks/colab_node.ipynb` | Quickstart setup for Google Colab nodes |

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
  --prompt "Once upon a time" \
  --max-tokens 20 \
  --num-nodes 2
```

### 3. Run Unit & Integration Test Suite

```bash
python -m pytest -p no:opik tests/
```

### 4. Run Benchmark Suite

```bash
python benchmarks/run_benchmarks.py \
  --model ./models/TinyLlama-1.1B-Chat-v1.0 \
  --max-tokens 20 \
  --num-nodes 2
```

---

## OpenAI API Usage Example

Start the API Gateway:

```python
import uvicorn
from shardflow.gateway.app import app, set_orchestrator
from shardflow.orchestrator.orchestrator import Orchestrator

# Initialize Orchestrator and attach to Gateway
orchestrator = Orchestrator(
    model_path="./models/TinyLlama-1.1B-Chat-v1.0",
    node_addresses=[("127.0.0.1", 9000), ("127.0.0.1", 9001)],
)
# Run in your server startup:
# await orchestrator.initialize()
# set_orchestrator(orchestrator)
```

Client request via `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "messages": [
      {"role": "user", "content": "Tell me a short story."}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

---

## Benchmark Results

Using TinyLlama 1.1B split across 2 pipeline nodes:

| Phase | Decode Strategy | Speed (tok/s) | Duration (20 tokens) |
|---|---|---|---|
| **Phase 1** | No KV Cache (O(n²) full sequence resend) | 0.1 tok/s | 258.5s |
| **Phase 2+** | Per-Node `DynamicCache` Incremental Decode | **0.9 tok/s** | **23.0s (11x speedup)** |

---

## License

MIT License
