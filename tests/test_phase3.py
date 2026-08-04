"""
Phase 3 test — verify Topology Registry (FastAPI) and API Gateway.
"""

import pytest
from fastapi.testclient import TestClient
from shardflow.registry.app import app as registry_app
from shardflow.gateway.app import app as gateway_app

registry_client = TestClient(registry_app)
gateway_client = TestClient(gateway_app)


def test_registry_registration_and_topology():
    # 1. Register Node 0
    node0_payload = {
        "node_id": "node-0",
        "addr": "127.0.0.1",
        "port": 9000,
        "layer_start": 0,
        "layer_end": 11,
        "vram_available_mb": 4000.0,
        "vram_total_mb": 4000.0,
    }
    resp = registry_client.post("/register", json=node0_payload)
    assert resp.status_code == 201
    assert resp.json()["node_id"] == "node-0"

    # 2. Register Node 1
    node1_payload = {
        "node_id": "node-1",
        "addr": "127.0.0.1",
        "port": 9001,
        "layer_start": 11,
        "layer_end": 22,
        "vram_available_mb": 4000.0,
        "vram_total_mb": 4000.0,
    }
    resp = registry_client.post("/register", json=node1_payload)
    assert resp.status_code == 201

    # 3. Heartbeat
    hb_resp = registry_client.post("/heartbeat", json={"node_id": "node-0"})
    assert hb_resp.status_code == 200

    # 4. Get topology
    topo_resp = registry_client.get("/topology")
    assert topo_resp.status_code == 200
    data = topo_resp.json()
    assert data["total_nodes"] == 2
    assert data["nodes"][0]["node_id"] == "node-0"
    assert data["nodes"][1]["node_id"] == "node-1"


def test_gateway_health_and_metrics():
    health_resp = gateway_client.get("/health")
    assert health_resp.status_code == 200
    assert "orchestrator_ready" in health_resp.json()

    metrics_resp = gateway_client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "active_sessions" in metrics_resp.json()
