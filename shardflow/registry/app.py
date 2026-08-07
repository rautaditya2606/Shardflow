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
REGISTRATION_TIMEOUT = float(os.getenv("SHARDFLOW_REGISTRATION_TIMEOUT", "60.0"))
HEARTBEAT_TIMEOUT = float(os.getenv("SHARDFLOW_HEARTBEAT_TIMEOUT", "90.0"))
HEARTBEAT_GRACE = float(os.getenv("SHARDFLOW_HEARTBEAT_GRACE", "60.0"))

# Offline model metadata — avoids HuggingFace Hub calls during /register
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
    "mistralai/mistral-7b-instruct-v0.2": 32,
    "google/gemma-2-9b-it": 42,
    "google/gemma-2-2b-it": 26,
}

KNOWN_MODEL_DIMS: Dict[str, tuple[int, int]] = {
    "qwen/qwen2.5-14b-instruct": (5120, 152064),
    "deepseek-ai/deepseek-r1-distill-qwen-14b": (5120, 152064),
    "qwen/qwen2.5-7b-instruct": (3584, 152064),
    "deepseek-ai/deepseek-r1-distill-qwen-7b": (3584, 152064),
    "qwen/qwen2.5-3b-instruct": (2048, 151936),
    "meta-llama/meta-llama-3-8b": (4096, 128256),
    "meta-llama/meta-llama-3-8b-instruct": (4096, 128256),
    "meta-llama/llama-2-7b-hf": (4096, 32000),
    "mistralai/mistral-7b-v0.1": (4096, 32000),
    "tinyllama/tinyllama-1.1b-chat-v1.0": (2048, 32000),
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
    expected_nodes: Optional[int] = Field(None, description="Expected cluster total nodes count")


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
    cluster_ready: bool = False
    topology_version: int = 0


class HeartbeatPayload(BaseModel):
    node_id: str


class TopologyResponse(BaseModel):
    nodes: List[NodeStatus]
    total_nodes: int
    updated_at: float
    topology_version: int = 0
    cluster_ready: bool = False


# In-memory registry store
_nodes: Dict[str, NodeStatus] = {}
_topology_version: int = 0
_expected_nodes_target: Optional[int] = None
_first_registration_time: Optional[float] = None
_cluster_partition_calculated: bool = False


def _reset_registry_state() -> None:
    global _nodes, _topology_version, _expected_nodes_target, _first_registration_time, _cluster_partition_calculated
    _nodes.clear()
    _topology_version = 0
    _expected_nodes_target = None
    _first_registration_time = None
    _cluster_partition_calculated = False


def _get_target_node_count() -> int:
    if _expected_nodes_target is not None and _expected_nodes_target > 0:
        return _expected_nodes_target
    return EXPECTED_NODES


def _active_nodes_for_model(model_id: str) -> List[NodeStatus]:
    key = model_id.lower()
    return [n for n in _nodes.values() if n.is_active and n.model_id.lower() == key]


def _is_cluster_ready(model_id: str) -> bool:
    if _cluster_partition_calculated:
        return True
    target_count = _get_target_node_count()
    active_count = len(_active_nodes_for_model(model_id))
    if active_count >= target_count and target_count > 0:
        return True
    if _first_registration_time is not None and (time.time() - _first_registration_time) > REGISTRATION_TIMEOUT:
        return True
    return False


def _bump_topology_version() -> None:
    global _topology_version
    _topology_version += 1


def _cleanup_inactive_nodes() -> None:
    now = time.time()
    for node in _nodes.values():
        silence = now - node.last_heartbeat
        if silence > HEARTBEAT_GRACE and node.is_active:
            logger.warning(
                "Node %s missed heartbeats for %.1fs — marking inactive (grace=%.0fs, evict=%.0fs)",
                node.node_id, silence, HEARTBEAT_GRACE, HEARTBEAT_TIMEOUT,
            )
            node.is_active = False

    dead_nodes = [
        nid for nid, node in _nodes.items()
        if now - node.last_heartbeat > HEARTBEAT_TIMEOUT
    ]
    if dead_nodes:
        _bump_topology_version()
    for nid in dead_nodes:
        logger.warning("Evicting dead node %s (no heartbeat for %.1fs)", nid, now - _nodes[nid].last_heartbeat)
        del _nodes[nid]


def _get_model_dims(model_id: str) -> tuple[int, int]:
    key = model_id.lower()
    if key in KNOWN_MODEL_DIMS:
        return KNOWN_MODEL_DIMS[key]
    if "14b" in key:
        return 5120, 152064
    if "7b" in key:
        return 3584, (152064 if "qwen" in key else 32000)
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id)
        return cfg.hidden_size, cfg.vocab_size
    except Exception:
        return 4096, 32000


def _rebalance_assignments(model_id: str) -> None:
    """Run AutoPartitionEngine once expected nodes are registered or timeout window elapses."""
    global _cluster_partition_calculated, _first_registration_time
    if not _nodes:
        return

    total_layers = get_model_total_layers(model_id)
    active = _active_nodes_for_model(model_id)
    target_count = _get_target_node_count()

    now = time.time()
    if _first_registration_time is None and active:
        _first_registration_time = now

    timeout_elapsed = False
    if _first_registration_time is not None and (now - _first_registration_time) >= REGISTRATION_TIMEOUT:
        timeout_elapsed = True

    if len(active) < target_count and not timeout_elapsed and not _cluster_partition_calculated:
        time_left = max(0.0, REGISTRATION_TIMEOUT - (now - (_first_registration_time or now)))
        logger.info(
            "Waiting for nodes: %d/%d registered for model %s (timeout window: %.1fs left)",
            len(active), target_count, model_id, time_left,
        )
        # Provisional slice for the first node prevents OOM if a runner loads early,
        # but /assignment stays pending until the cluster is complete.
        if len(active) == 1:
            n0 = active[0]
            n0.layer_start = 0
            n0.layer_end = total_layers // max(1, target_count)
            n0.is_first_node = True
            n0.is_last_node = (target_count == 1)
            n0.next_node_host = None
            n0.next_node_port = None
        return

    hidden_size, vocab_size = _get_model_dims(model_id)

    engine = AutoPartitionEngine(
        total_layers=total_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )
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

    _cluster_partition_calculated = True
    _bump_topology_version()
    logger.info(
        "Partitioned %d layers across %d nodes (topology v%d, target=%d, timeout_fallback=%s): %s",
        total_layers,
        len(assignments),
        _topology_version,
        target_count,
        timeout_elapsed,
        [(a.node_id, a.layer_start, a.layer_end) for a in assignments],
    )


@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=NodeRegistrationResponse)
def register_node(payload: NodeRegistration):
    """Register or update a pipeline node and receive dynamic layer assignments."""
    global _expected_nodes_target, _first_registration_time
    _cleanup_inactive_nodes()
    now = time.time()

    if payload.expected_nodes is not None and payload.expected_nodes > 0:
        if _expected_nodes_target != payload.expected_nodes:
            logger.info("Dynamic expected nodes target updated to %d by node %s", payload.expected_nodes, payload.node_id)
            _expected_nodes_target = payload.expected_nodes

    if _first_registration_time is None:
        _first_registration_time = now

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
        cluster_ready=_is_cluster_ready(payload.model_id),
        topology_version=_topology_version,
    )


class HeartbeatResponse(BaseModel):
    status: str = "ok"
    node_id: str
    next_node_host: Optional[str] = None
    next_node_port: Optional[int] = None
    is_first_node: bool = False
    is_last_node: bool = False
    layer_start: int = 0
    layer_end: int = 0
    cluster_ready: bool = False
    topology_version: int = 0


@router.post("/heartbeat", status_code=status.HTTP_200_OK, response_model=HeartbeatResponse)
def heartbeat(payload: HeartbeatPayload):
    """Receive heartbeat ping from a node and return latest routing assignment."""
    if payload.node_id not in _nodes:
        raise HTTPException(status_code=404, detail="Node not registered")
    
    node = _nodes[payload.node_id]
    node.last_heartbeat = time.time()
    node.is_active = True

    return HeartbeatResponse(
        status="ok",
        node_id=node.node_id,
        next_node_host=node.next_node_host if _is_cluster_ready(node.model_id) else None,
        next_node_port=node.next_node_port if _is_cluster_ready(node.model_id) else None,
        is_first_node=node.is_first_node,
        is_last_node=node.is_last_node,
        layer_start=node.layer_start,
        layer_end=node.layer_end,
        cluster_ready=_is_cluster_ready(node.model_id),
        topology_version=_topology_version,
    )


@router.api_route("/topology", methods=["GET", "HEAD"], response_model=TopologyResponse)
def get_topology():
    """Return ordered topology of active nodes sorted by layer_start."""
    _cleanup_inactive_nodes()
    sorted_nodes = sorted(_nodes.values(), key=lambda n: n.layer_start)
    model_id = sorted_nodes[0].model_id if sorted_nodes else ""
    return TopologyResponse(
        nodes=sorted_nodes,
        total_nodes=len(sorted_nodes),
        updated_at=time.time(),
        topology_version=_topology_version,
        cluster_ready=_is_cluster_ready(model_id) if model_id else False,
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

    Returns 200 with slice info once AutoPartitionEngine has run and the cluster is ready.
    Returns 202 while waiting for all expected nodes to register.
    """
    if node_id not in _nodes:
        raise HTTPException(status_code=404, detail="Node not registered")

    node = _nodes[node_id]
    cluster_ready = _is_cluster_ready(node.model_id)
    active_count = len(_active_nodes_for_model(node.model_id))

    target_count = _get_target_node_count()
    if not cluster_ready or node.layer_end == 0:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=202,
            content={
                "status": "pending",
                "cluster_ready": False,
                "topology_version": _topology_version,
                "message": f"Waiting for nodes: {active_count}/{target_count}",
            },
        )

    return {
        "status": "assigned",
        "cluster_ready": True,
        "topology_version": _topology_version,
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
