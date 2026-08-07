"""
Layer 5 — Topology Registry (FastAPI)

Central service for node registration, health tracking, and topology discovery.
Solves the dynamic IP / tunnel address discovery problem for Colab nodes.

Endpoints:
- POST /register — register a new node or update layer assignment
- GET /topology — get ordered list of active nodes
- POST /heartbeat — update last-seen timestamp for a node
"""

import logging
import os
import time
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status

from fastapi import FastAPI, HTTPException, status, APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()

app = FastAPI(
    title="ShardFlow Topology Registry",
    description="Node registration, topology discovery, and heartbeat monitoring",
    version="0.1.0",
)
app.include_router(router)



from shardflow.partition.engine import AutoPartitionEngine, NodeVRAMInfo

EXPECTED_NODES = int(os.getenv("SHARDFLOW_EXPECTED_NODES", "2"))

# ponytail: Offline model layer map to eliminate network latency during /register
KNOWN_MODEL_LAYERS: Dict[str, int] = {
    "tinyllama/tinyllama-1.1b-chat-v1.0": 22,
    "qwen/qwen2.5-7b-instruct": 28,
    "qwen/qwen2.5-14b-instruct": 48,
    "qwen/qwen2.5-3b-instruct": 36,
    "qwen/qwen2.5-1.5b-instruct": 28,
    "qwen/qwen2.5-0.5b-instruct": 24,
    "deepseek-ai/deepseek-r1-distill-qwen-14b": 48,
    "deepseek-ai/deepseek-r1-distill-qwen-7b": 28,
    "meta-llama/meta-llama-3-8b": 32,
    "meta-llama/meta-llama-3-8b-instruct": 32,
    "meta-llama/llama-2-7b-hf": 32,
    "mistralai/mistral-7b-v0.1": 32,
}


def get_model_total_layers(model_id: str) -> int:
    """
    Get total hidden layers for model.
    Checks fast offline lookup first to eliminate network latency during registration.
    """
    key = model_id.lower()
    if key in KNOWN_MODEL_LAYERS:
        return KNOWN_MODEL_LAYERS[key]

    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id)
        if hasattr(cfg, "num_hidden_layers") and cfg.num_hidden_layers > 0:
            logger.info("Loaded total layers (%d) for model %s via AutoConfig", cfg.num_hidden_layers, model_id)
            return cfg.num_hidden_layers
    except Exception as e:
        logger.warning(
            "AutoConfig.from_pretrained failed for '%s': %s. Falling back to default.",
            model_id, e
        )

    return 48 if "14b" in key else (28 if "7b" in key else 22)


class NodeRegistration(BaseModel):
    node_id: str = Field(..., description="Unique node identifier")
    addr: str = Field(..., description="Public IP or Cloudflare tunnel host")
    port: int = Field(..., description="TCP port for activations")
    layer_start: Optional[int] = Field(None, description="Assigned start layer (optional/auto-calculated)")
    layer_end: Optional[int] = Field(None, description="Assigned end layer (optional/auto-calculated)")
    vram_available_mb: float = Field(0.0, description="Available VRAM in MB")
    vram_total_mb: float = Field(0.0, description="Total VRAM in MB")
    model_id: str = Field("TinyLlama/TinyLlama-1.1B-Chat-v1.0", description="Model ID")


class NodeStatus(BaseModel):
    node_id: str
    addr: str
    port: int
    layer_start: int
    layer_end: int
    vram_available_mb: float = 0.0
    vram_total_mb: float = 0.0
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    last_heartbeat: float
    is_active: bool = True
    is_first_node: bool = False
    is_last_node: bool = False
    next_node_host: Optional[str] = None
    next_node_port: Optional[int] = None


class NodeRegistrationResponse(NodeStatus):
    total_model_layers: int


class HeartbeatPayload(BaseModel):
    node_id: str


class TopologyResponse(BaseModel):
    nodes: List[NodeStatus]
    total_nodes: int
    updated_at: float


# In-memory registry store
_nodes: Dict[str, NodeStatus] = {}
HEARTBEAT_TIMEOUT = 30.0  # Evict nodes silent for > 30 seconds


def _cleanup_inactive_nodes() -> None:
    now = time.time()
    dead_nodes = [
        nid for nid, node in _nodes.items()
        if now - node.last_heartbeat > HEARTBEAT_TIMEOUT
    ]
    for nid in dead_nodes:
        logger.warning("Evicting dead node %s (no heartbeat for %.1fs)", nid, now - _nodes[nid].last_heartbeat)
        del _nodes[nid]


def _rebalance_assignments(model_id: str) -> None:
    """Run AutoPartitionEngine once all expected nodes are registered."""
    if not _nodes:
        return

    total_layers = get_model_total_layers(model_id)

    active = [n for n in _nodes.values() if n.is_active]
    if len(active) < EXPECTED_NODES:
        logger.info(
            "Waiting for nodes: %d/%d registered", len(active), EXPECTED_NODES
        )
        # ponytail: single node owns all layers until full cluster joins
        if len(active) == 1:
            n0 = active[0]
            n0.layer_start = 0
            n0.layer_end = total_layers
            n0.is_first_node = True
            n0.is_last_node = True
        return
    # ponytail: fast offline lookup to eliminate HF network latency during /register
    key = model_id.lower()
    if "14b" in key:
        hidden_size, vocab_size = 5120, 152064
    elif "7b" in key:
        hidden_size, vocab_size = 3584, (152064 if "qwen" in key else 32000)
    else:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_id)
            hidden_size = cfg.hidden_size
            vocab_size = cfg.vocab_size
        except Exception:
            hidden_size, vocab_size = 4096, 32000

    engine = AutoPartitionEngine(
        total_layers=total_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )
    # ponytail: Default to 15000 MB if node reports 0 VRAM (e.g., tests or nodes without VRAM reporting)
    vram_infos = [
        NodeVRAMInfo(
            node_id=n.node_id,
            vram_available_mb=n.vram_available_mb if n.vram_available_mb > 0 else 15000.0,
            vram_total_mb=n.vram_total_mb if n.vram_total_mb > 0 else 15000.0,
        )
        for n in active
    ]

    try:
        assignments = engine.compute_partition(vram_infos)
    except ValueError as e:
        logger.error("AutoPartitionEngine failed: %s", e)
        return

    # Write assignments back, then update routing
    sorted_node_ids = [a.node_id for a in assignments]
    for a in assignments:
        _nodes[a.node_id].layer_start = a.layer_start
        _nodes[a.node_id].layer_end = a.layer_end

    for i, nid in enumerate(sorted_node_ids):
        is_last = (i == len(sorted_node_ids) - 1)
        _nodes[nid].is_first_node = (i == 0)
        _nodes[nid].is_last_node = is_last
        _nodes[nid].next_node_host = None
        _nodes[nid].next_node_port = None
        if not is_last:
            nxt = _nodes[sorted_node_ids[i + 1]]
            _nodes[nid].next_node_host = nxt.addr
            _nodes[nid].next_node_port = nxt.port

    logger.info(
        "Partitioned %d layers across %d nodes: %s",
        total_layers,
        len(assignments),
        [(a.node_id, a.layer_start, a.layer_end) for a in assignments],
    )


@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=NodeRegistrationResponse)
def register_node(payload: NodeRegistration):
    """Register or update a pipeline node and receive dynamic layer assignments."""
    _cleanup_inactive_nodes()
    now = time.time()

    # Pre-register entry
    node_status = NodeStatus(
        node_id=payload.node_id,
        addr=payload.addr,
        port=payload.port,
        layer_start=payload.layer_start or 0,
        layer_end=payload.layer_end or 0,
        vram_available_mb=payload.vram_available_mb,
        vram_total_mb=payload.vram_total_mb,
        model_id=payload.model_id,
        last_heartbeat=now,
        is_active=True,
    )
    _nodes[payload.node_id] = node_status

    # Recalculate dynamic splits across active pool
    _rebalance_assignments(payload.model_id)

    updated_status = _nodes[payload.node_id]
    total_layers = get_model_total_layers(payload.model_id)

    logger.info(
        "Registered node %s (%s:%d) -> assigned layers [%d, %d) (first=%s, last=%s)",
        updated_status.node_id, updated_status.addr, updated_status.port,
        updated_status.layer_start, updated_status.layer_end,
        updated_status.is_first_node, updated_status.is_last_node,
    )

    return NodeRegistrationResponse(
        **updated_status.model_dump(),
        total_model_layers=total_layers,
    )


@router.post("/heartbeat", status_code=status.HTTP_200_OK)
def heartbeat(payload: HeartbeatPayload):
    """Receive heartbeat ping from a node."""
    if payload.node_id not in _nodes:
        raise HTTPException(status_code=404, detail="Node not registered")
    
    _nodes[payload.node_id].last_heartbeat = time.time()
    _nodes[payload.node_id].is_active = True
    return {"status": "ok", "node_id": payload.node_id}


@router.api_route("/topology", methods=["GET", "HEAD"], response_model=TopologyResponse)
def get_topology():
    """Return ordered topology of active nodes sorted by layer_start."""
    _cleanup_inactive_nodes()
    sorted_nodes = sorted(_nodes.values(), key=lambda n: n.layer_start)
    return TopologyResponse(
        nodes=sorted_nodes,
        total_nodes=len(sorted_nodes),
        updated_at=time.time(),
    )


@router.delete("/nodes/{node_id}")
def unregister_node(node_id: str):
    """Manually unregister a node."""
    if node_id in _nodes:
        del _nodes[node_id]
        return {"status": "unregistered", "node_id": node_id}
    raise HTTPException(status_code=404, detail="Node not found")


@router.get("/assignment/{node_id}")
def get_assignment(node_id: str):
    """
    Poll for a node's layer assignment.

    Returns 200 with slice info once AutoPartitionEngine has run.
    Returns 202 while waiting for all expected nodes to register.
    Nodes call this after registering with no layer bounds.
    """
    if node_id not in _nodes:
        raise HTTPException(status_code=404, detail="Node not registered")

    node = _nodes[node_id]
    # layer_end == 0 means assignment hasn't run yet
    if node.layer_end == 0:
        active_count = sum(1 for n in _nodes.values() if n.is_active)
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=202,
            content={
                "status": "pending",
                "message": f"Waiting for nodes: {active_count}/{EXPECTED_NODES}",
            },
        )

    return {
        "status": "assigned",
        "node_id": node_id,
        "layer_start": node.layer_start,
        "layer_end": node.layer_end,
        "is_first_node": node.is_first_node,
        "is_last_node": node.is_last_node,
        "next_node_host": node.next_node_host,
        "next_node_port": node.next_node_port,
    }



def main():
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="ShardFlow Topology Registry")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=8001, help="Listen port")
    args = parser.parse_args()

    uvicorn.run("shardflow.registry.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
