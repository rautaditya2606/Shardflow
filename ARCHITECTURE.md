# ShardFlow Architecture Record

> Living document. Last updated: 2026-08-06.
> Purpose: capture architectural decisions and context so any future session can pick up without re-deriving history.

---

## What ShardFlow Is (v1)

A distributed LLM inference framework that:
- Splits any HuggingFace transformer across N GPU machines
- Uses length-prefixed TCP for activation transfer between nodes
- Exposes an OpenAI-compatible HTTP endpoint (Render)
- Solves dynamic IP discovery via a topology registry (heartbeat + auto-registration)
- Supports Google Colab T4 GPUs via bore.pub tunnels

**Not a Colab product.** Colab is the cheapest demo. The architecture must be deployment-agnostic.

---

## v1 Architecture (Current)

```
Client
  ↓ HTTP
Gateway (Render)
  ↓ owns the decode loop
Orchestrator (tokenizer + embed + sampling config)
  ↓ TCP — sends ACTIVATION every token
Node 0  [layers 0-N/2]
  ↓ TCP — forwards ACTIVATION
Node 1  [layers N/2-N, LM head]
  ↓ TCP — returns TOKEN_ID (8 bytes)
Node 0
  ↓ TCP — returns TOKEN_ID
Orchestrator
  ↓ HTTP SSE
Client
```

### Per-token cost in v1 (measured, Colab+bore.pub)
- Raw GPU compute (both nodes): ~68ms/token → 14.7 tok/s
- End-to-end via Render: ~440ms/token → 2.27 tok/s
- **Gap: 372ms/token = gateway + orchestrator in every token loop**

### Per-token cost locally (measured, RTX 3050, TinyLlama)
| Stage | Latency |
|---|---|
| Embedding | 0.03ms |
| TCP + full node pipeline | 24.56ms |
| Sampling (GPU) | 0.00ms |
| **Total** | **24.60ms = 40.7 tok/s** (Up from 28.8 tok/s baseline) |

**The GPU is idle >90% of the time waiting for TCP. GPU compute is essentially free.**

---

## What Was Optimized (Aug 2026 Session)

### Applied
1. **Deleted 90-line duplicate decode loop** from `gateway/app.py`. Added `generate_stream()` async generator to `Orchestrator` as single source of truth.
2. **Conditional `drain()`** — removed unconditional `await writer.drain()` per message. Now drains only when write buffer exceeds 64KB. Eliminates one kernel RTT per token.
3. **Auto-partition wired** — `AutoPartitionEngine` (already written in `partition/engine.py`, never called) is now invoked on `/register`. VRAM-weighted layer splits, LM head budget deducted from last node.
4. **`/assignment/{node_id}` polling endpoint** — nodes register with no `--layer-start`/`--layer-end`, poll until assigned. No manual layer math needed.
5. **`SHARDFLOW_EXPECTED_NODES` env var** (default 2) — registry waits for N nodes before running partition.
6. **Fast Tensor Serialization** — replaced `bytes(tensor.untyped_storage())` (which took 10.6ms per token due to Python C-API element-by-element loop) with `tensor.view(torch.uint8).cpu().numpy().tobytes()` (0.005ms). Reduced local token pipeline time from 34.7ms to 24.6ms (**40.7 tok/s on RTX 3050**). Supports all dtypes (fp16, bf16, fp32, int64).

### Benchmarked, not changed
- `.clone()` in `decode_message` — **kept**. PyTorch warns `frombuffer` on bytes is non-writable; clone is load-bearing.
- `bytes(cuda_tensor.untyped_storage())` directly — **5x SLOWER** than `.cpu()`. Rejected.
- `tensor.numpy(force=True).tobytes()` — **700x faster** than `.cpu()+bytes()` on RTX 3050 (0.017ms vs 12ms). Float16 only (bfloat16 not supported by numpy). Not yet applied — pending dtype handling.
- Triple-allocation in `encode_message` — not applied (lower priority after network analysis confirmed network >> serialization).

### Root cause of the 6.5x end-to-end slowness
The orchestrator is deployed on Render, geographically far from Colab nodes. Every token requires 2+ intercontinental bore.pub round trips. This is architectural, not a tuning problem.

---

## The Core Architectural Problem

The gateway sits inside the per-token decode loop:

```
GW → N0 → N1 → GW → N0 → N1 → GW → ...
        (every single token)
```

Every token inherits the gateway's full round-trip latency.
Performance is tightly coupled to where the gateway is hosted.
This does not improve regardless of how fast the GPU is.

---

## v2 Direction: Control / Data Plane Separation

> The gateway must exit the per-token critical path.
> The data plane (nodes) owns the decode loop once a session starts.
> The control plane (gateway) manages session lifecycle only.

### The principle

**Control plane** — gateway, scheduler, registry, auth:
- Handles session start and end
- Runs anywhere
- Not in the token loop

**Data plane** — nodes:
- Handles inference
- Drives the decode loop peer-to-peer
- Platform-agnostic

### v2 per-token flow

```
Session start (once):
  Client → Gateway → Node 0  [START_SESSION message]

Per token (gateway NOT involved):
  Node 0 → Node 1 → ... → Node N → (back to Node 0)

Token stream (back to client):
  Node N → stream → Gateway → SSE → Client

Session end:
  Gateway → CLEAR → Node 0  (propagates through chain)
```

### New protocol primitive needed: START_SESSION
Sent once by gateway to Node 0. Contains:
- Prompt token IDs
- Sampling config (temperature, top_k, top_p, max_tokens)
- Stream-back address (where to send generated tokens)

### What stays completely unchanged
- Node layer slicing
- KV cache (per-session DynamicCache)
- Registry + heartbeat + auto-partition
- TCP transport framing
- CLEAR message
- All deployment environments

### What changes

**Node 0** gains:
- Tokenizer (lightweight, just a lookup table)
- Decode driver loop

**Last node** gains:
- Stream-back path: sends TOKEN_ID to a gateway-facing stream, not back through the chain

**Gateway** becomes:
- Session initiator only
- Lightweight stream forwarder for SSE

### Why NOT to move the orchestrator to Node 0 (rejected option)

This makes Node 0 "special":
- Node 0 becomes: layers + tokenizer + scheduler + API + embedding + orchestration
- Breaks the uniform "a node is just a layer slice" abstraction
- Scaling is awkward (Node 0 is now a control node, not a worker)
- Privileges one deployment environment

The data-plane-owns-the-loop design avoids this. Node 0 drives the decode loop as a **session owner**, not a controller. Its layer-processing logic is unchanged.

---

## Deployment Targets

| Platform | Tunnel needed | Est. tok/s today | Est. tok/s v2 | Notes |
|---|:---:|:---:|:---:|---|
| Google Colab (T4) | bore.pub | ~14 | ~25+ | Current demo |
| Kaggle (T4/P100) | bore.pub | ~12 | ~20+ | Same runner |
| RunPod / Lambda / Vast | none | ~28+ | ~50+ | Static IPs, direct TCP |
| Local GPUs | none | ~28 | ~50+ | Localhost TCP |

With rented GPUs (static IPs): bore.pub overhead disappears entirely.
With v2 architecture: Render/gateway overhead also disappears.
Remaining latency: inter-node TCP + GPU compute only.

---

## Pending Work

### Current scope (short term)
- [x] Fast activation serialization (`numpy(force=True)`) for fp16/fp32 activations
- [x] `encode_message` single-buffer pre-allocation (eliminated triple-copy allocations)
- [x] `layer_loader.py`: `accelerate` device_map partial loading with meta device
- [x] Kaggle runner script (`scripts/kaggle_runner.py`)
- [x] Rented GPU runner script (`scripts/runpod_runner.py` for direct IP mode)

### v2 scope (future milestone)
- [x] `START_SESSION` message type & framing in protocol (`protocol.py`)
- [ ] Node 0: decode driver + tokenizer
- [ ] Last node: direct TCP stream-back path to gateway
- [ ] Gateway: session handoff & SSE forwarding proxy
- [ ] Protocol versioning

---

## Key Files

| File | Role |
|---|---|
| `shardflow/gateway/app.py` | HTTP gateway, SSE streaming, session lifecycle |
| `shardflow/orchestrator/orchestrator.py` | Decode loop, `generate()`, `generate_stream()` |
| `shardflow/node/node.py` | Layer forward, KV cache, TCP server/client |
| `shardflow/transport/protocol.py` | Wire format, encode/decode, send/recv |
| `shardflow/transport/connection.py` | NodeServer, NodeClient |
| `shardflow/registry/app.py` | Node registration, topology, auto-partition |
| `shardflow/partition/engine.py` | VRAM-weighted layer split algorithm |
| `shardflow/node/kv_cache.py` | Per-session DynamicCache with TTL eviction |
| `benchmarks/profile_pipeline.py` | Stage latency profiler (embed / TCP+pipeline / sample) |

---

## v2 Design Decisions (Locked, Aug 2026)

These were debated and settled. Don't re-derive them.

### Last node streams directly to gateway — NOT back through the chain

**Rejected:** TOKEN_ID forwarded Node N → N-1 → ... → 0 → Gateway.
- Every token traverses N extra hops
- Intermediate nodes own work they don't logically belong to
- A single intermediate node failure kills both forward inference and reverse streaming

**Accepted:** Node N holds a direct TCP connection to the gateway.
- Registry tells the terminal node: "you are last, stream to gateway at X"
- Intermediate nodes never need the gateway address
- Failure domains are clean: forward pass is separate from stream-back

### Node 0 owning the tokenizer is fine

Tokenizer is part of inference, not control. It's small, deterministic, tied to the model. Node 0 owning tokenizer + embeddings + first transformer layers is a clean separation of responsibilities. The gateway becomes truly model-agnostic.

### Registry tells the terminal node about the gateway

Registry already computes the full topology. It can include `gateway_stream_addr` in the terminal node's registration response. Only one node needs this information.

---

## v2 Brainstorm (Do Not Implement in v1)

- Gateway outside per-token critical path (START_SESSION protocol)
- Data plane vs control plane separation
- Direct streaming from terminal node to gateway
- Heterogeneous GPU scheduling (different VRAM, different model layers)
- Multi-node (>2) optimizations
- Kaggle runner script
- Rented GPU runner (no tunnel, direct IP mode)
- Protocol versioning

---

## v1 Release Status: COMPLETED ✅

1. [x] `layer_loader.py` — `accelerate` device_map partial loading with zero-RAM meta device shell and direct safetensors slice loading.
2. [x] Fast Tensor Serialization — `.view(torch.uint8).cpu().numpy().tobytes()` (40.7 tok/s).
3. [x] Inverted Topology & Auto-Partitioning — `AutoPartitionEngine` with VRAM weighting and `/assignment/{node_id}` polling.
4. [x] Single source stream decode loop in `Orchestrator.generate_stream()`.
5. [x] Registry offline model layer map & fallback handling (`KNOWN_MODEL_LAYERS`).
6. [x] Test suite 100% passing (`10 passed, 1 skipped`).
7. [x] v1 Complete & Ready for `v1.0.0` Tagging.

