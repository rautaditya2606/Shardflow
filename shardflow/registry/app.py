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



KNOWN_MODEL_LAYERS = {
    "tinyllama/tinyllama-1.1b-chat-v1.0": 22,
    "meta-llama/meta-llama-3-8b": 32,
    "meta-llama/meta-llama-3-8b-instruct": 32,
    "meta-llama/llama-2-7b": 32,
    "mistralai/mistral-7b-v0.1": 32,
    "qwen/qwen2-7b": 28,
}


def get_model_total_layers(model_id: str) -> int:
    """
    Fetch total hidden layers directly from HuggingFace AutoConfig.
    Falls back to known offline dict or default only if network/HF lookup fails.
    """
    # 1. Primary path: Fetch from HuggingFace AutoConfig
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id)
        if hasattr(cfg, "num_hidden_layers") and cfg.num_hidden_layers > 0:
            logger.info("Loaded total layers (%d) for model %s via AutoConfig", cfg.num_hidden_layers, model_id)
            return cfg.num_hidden_layers
    except Exception as e:
        logger.warning(
            "AutoConfig.from_pretrained failed for '%s': %s. Falling back to offline layer mapping.",
            model_id, e
        )

    # 2. Secondary fallback path: Offline lookup map
    key = model_id.lower()
    if key in KNOWN_MODEL_LAYERS:
        fallback_layers = KNOWN_MODEL_LAYERS[key]
        logger.warning("Using offline fallback mapping for model '%s': %d layers", model_id, fallback_layers)
        return fallback_layers

    # 3. Last resort fallback
    logger.error(
        "Could not determine layer count for model '%s' via AutoConfig or offline map. "
        "Defaulting to 22 layers (TinyLlama).", model_id
    )
    return 22


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
    """Recalculate dynamic layer bounds and next-node routing across active nodes."""
    if not _nodes:
        return

    # Sort nodes deterministically by layer_start, then node_id
    sorted_node_ids = sorted(_nodes.keys(), key=lambda nid: (_nodes[nid].layer_start, nid))
    total_layers = get_model_total_layers(model_id)
    n_nodes = len(sorted_node_ids)

    # Only perform auto-split if nodes did not specify explicit non-zero layer bounds
    all_auto = all(_nodes[nid].layer_end == 0 for nid in sorted_node_ids)
    if all_auto:
        base, rem = divmod(total_layers, n_nodes)
        curr_layer = 0
        for i, nid in enumerate(sorted_node_ids):
            count = base + (1 if i < rem else 0)
            _nodes[nid].layer_start = curr_layer
            _nodes[nid].layer_end = curr_layer + count
            curr_layer += count

    for i, nid in enumerate(sorted_node_ids):
        is_first = (i == 0)
        is_last = (i == n_nodes - 1)

        next_host = None
        next_port = None
        if not is_last:
            next_node = _nodes[sorted_node_ids[i + 1]]
            next_host = next_node.addr
            next_port = next_node.port

        node = _nodes[nid]
        node.is_first_node = is_first
        node.is_last_node = is_last
        node.next_node_host = next_host
        node.next_node_port = next_port


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
