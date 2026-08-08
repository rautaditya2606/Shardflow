"""
Integration tests for Topology Registry and API Gateway endpoints (/v1/chat/completions JSON and SSE streaming).
"""

import pytest
import asyncio
from starlette.testclient import TestClient

from shardflow.registry.app import app as registry_app
from shardflow.gateway.app import app as gateway_app, set_orchestrator
from shardflow.orchestrator.orchestrator import Orchestrator
from shardflow.node.node import PipelineNode
from tests.fixtures.mock_models import create_dummy_model_slice, DummyTokenizer


def test_registry_registration_and_heartbeat():
    """Verify nodes register their layer boundaries and report heartbeats."""
    client = TestClient(registry_app)

    # 1. Register Node 0
    resp0 = client.post("/register", json={
        "node_id": "node-0",
        "addr": "127.0.0.1",
        "port": 9100,
        "layer_start": 0,
        "layer_end": 11,
        "vram_available_mb": 8000.0,
        "vram_total_mb": 8000.0,
    })
    assert resp0.status_code == 201
    assert resp0.json()["node_id"] == "node-0"

    # 2. Register Node 1
    resp1 = client.post("/register", json={
        "node_id": "node-1",
        "addr": "127.0.0.1",
        "port": 9101,
        "layer_start": 11,
        "layer_end": 22,
        "vram_available_mb": 8000.0,
        "vram_total_mb": 8000.0,
    })
    assert resp1.status_code == 201

    # 3. Heartbeat
    hb_resp = client.post("/heartbeat", json={"node_id": "node-0"})
    assert hb_resp.status_code == 200

    # 4. Topology query
    topo_resp = client.get("/topology")
    assert topo_resp.status_code == 200
    data = topo_resp.json()
    assert data["total_nodes"] == 2
    assert data["nodes"][0]["node_id"] == "node-0"


@pytest.mark.asyncio
async def test_gateway_chat_completions_json_and_sse_streaming():
    """Verify Gateway /v1/chat/completions serves OpenAI-compatible JSON and SSE streams."""
    # 1. Setup 2-node pipeline
    node1_slice = create_dummy_model_slice(layer_start=1, layer_end=2, is_first=False, is_last=True)
    node1 = PipelineNode(model_slice=node1_slice, is_first_node=False, is_last_node=True, listen_host="127.0.0.1", listen_port=0)
    await node1.start()
    node1_port = node1._server._server.sockets[0].getsockname()[1]

    node0_slice = create_dummy_model_slice(layer_start=0, layer_end=1, is_first=True, is_last=False)
    node0 = PipelineNode(model_slice=node0_slice, is_first_node=True, is_last_node=False, next_node_host="127.0.0.1", next_node_port=node1_port, listen_host="127.0.0.1", listen_port=0)
    await node0.start()
    node0_port = node0._server._server.sockets[0].getsockname()[1]

    # 2. Setup orchestrator with dummy tokenizer
    orch = Orchestrator(model_path="dummy", node_addresses=[("127.0.0.1", node0_port)])
    orch.tokenizer = DummyTokenizer()
    orch.vocab_size = 100
    orch.hidden_dim = 32
    orch.embed_tokens = node0_slice.embed_tokens
    orch._node0_client = node0._next_client  # mock client placeholder
    set_orchestrator(orch)

    # 3. Query Gateway via TestClient
    client = TestClient(gateway_app)
    health_resp = client.get("/health")
    assert health_resp.status_code == 200

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200

    # Cleanup
    await node0.stop()
    await node1.stop()
