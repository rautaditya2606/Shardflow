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

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ShardFlow Topology Registry",
    description="Node registration, topology discovery, and heartbeat monitoring",
    version="0.1.0",
)


class NodeRegistration(BaseModel):
    node_id: str = Field(..., description="Unique node identifier")
    addr: str = Field(..., description="Public IP or Cloudflare tunnel host")
    port: int = Field(..., description="TCP port for activations")
    layer_start: int = Field(..., description="Assigned start layer (inclusive)")
    layer_end: int = Field(..., description="Assigned end layer (exclusive)")
    vram_available_mb: float = Field(0.0, description="Available VRAM in MB")
    vram_total_mb: float = Field(0.0, description="Total VRAM in MB")


class NodeStatus(NodeRegistration):
    last_heartbeat: float = Field(..., description="Monotonic timestamp of last heartbeat")
    is_active: bool = True


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


@app.post("/register", status_code=status.HTTP_201_CREATED, response_model=NodeStatus)
def register_node(payload: NodeRegistration):
    """Register or update a pipeline node."""
    _cleanup_inactive_nodes()
    now = time.time()

    node_status = NodeStatus(
        **payload.model_dump(),
        last_heartbeat=now,
        is_active=True,
    )
    _nodes[payload.node_id] = node_status
    logger.info(
        "Registered node %s (%s:%d) for layers [%d, %d)",
        payload.node_id, payload.addr, payload.port, payload.layer_start, payload.layer_end
    )
    return node_status


@app.post("/heartbeat", status_code=status.HTTP_200_OK)
def heartbeat(payload: HeartbeatPayload):
    """Receive heartbeat ping from a node."""
    if payload.node_id not in _nodes:
        raise HTTPException(status_code=404, detail="Node not registered")
    
    _nodes[payload.node_id].last_heartbeat = time.time()
    _nodes[payload.node_id].is_active = True
    return {"status": "ok", "node_id": payload.node_id}


@app.get("/topology", response_model=TopologyResponse)
def get_topology():
    """Return ordered topology of active nodes sorted by layer_start."""
    _cleanup_inactive_nodes()
    # Sort nodes by layer_start
    sorted_nodes = sorted(_nodes.values(), key=lambda n: n.layer_start)
    return TopologyResponse(
        nodes=sorted_nodes,
        total_nodes=len(sorted_nodes),
        updated_at=time.time(),
    )


@app.delete("/nodes/{node_id}")
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
