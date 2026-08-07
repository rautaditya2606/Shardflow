"""
Tests for Dynamic Registry Partitioning & Timeout Fallback in ShardFlow.
"""

import time
import pytest
from fastapi.testclient import TestClient
from shardflow.registry.app import app as registry_app, _reset_registry_state, REGISTRATION_TIMEOUT

client = TestClient(registry_app)


@pytest.fixture(autouse=True)
def reset_state():
    _reset_registry_state()
    yield
    _reset_registry_state()


def test_dynamic_expected_nodes_trigger():
    """Verify registry partitions immediately when expected_nodes=3 is reached."""
    # 1. Register Node 0 (specifies expected_nodes=3)
    p0 = {
        "node_id": "node-0",
        "addr": "127.0.0.1",
        "port": 9000,
        "vram_available_mb": 15000.0,
        "vram_total_mb": 15000.0,
        "model_id": "qwen/qwen2.5-7b-instruct",
        "expected_nodes": 3,
    }
    r0 = client.post("/register", json=p0)
    assert r0.status_code == 201
    assert r0.json()["cluster_ready"] is False

    # Check assignment status for node-0 (should be pending)
    a0 = client.get("/assignment/node-0")
    assert a0.status_code == 202
    assert a0.json()["status"] == "pending"
    assert "Waiting for nodes: 1/3" in a0.json()["message"]

    # 2. Register Node 1
    p1 = {
        "node_id": "node-1",
        "addr": "127.0.0.1",
        "port": 9001,
        "vram_available_mb": 15000.0,
        "vram_total_mb": 15000.0,
        "model_id": "qwen/qwen2.5-7b-instruct",
    }
    r1 = client.post("/register", json=p1)
    assert r1.status_code == 201
    assert r1.json()["cluster_ready"] is False

    # 3. Register Node 2 (3rd node reaches expected count)
    p2 = {
        "node_id": "node-2",
        "addr": "127.0.0.1",
        "port": 9002,
        "vram_available_mb": 15000.0,
        "vram_total_mb": 15000.0,
        "model_id": "qwen/qwen2.5-7b-instruct",
    }
    r2 = client.post("/register", json=p2)
    assert r2.status_code == 201
    assert r2.json()["cluster_ready"] is True

    # Poll assignment for all 3 nodes (all should be assigned)
    for nid in ["node-0", "node-1", "node-2"]:
        res = client.get(f"/assignment/{nid}")
        assert res.status_code == 200
        assert res.json()["status"] == "assigned"
        assert res.json()["cluster_ready"] is True


def test_registration_timeout_fallback(monkeypatch):
    """Verify registry falls back to active nodes when timeout elapses."""
    # Set short timeout window for fast testing
    import shardflow.registry.app as reg_module
    monkeypatch.setattr(reg_module, "REGISTRATION_TIMEOUT", 0.5)

    # 1. Register Node 0 with expected_nodes=3
    p0 = {
        "node_id": "node-0",
        "addr": "127.0.0.1",
        "port": 9000,
        "vram_available_mb": 15000.0,
        "vram_total_mb": 15000.0,
        "model_id": "qwen/qwen2.5-7b-instruct",
        "expected_nodes": 3,
    }
    r0 = client.post("/register", json=p0)
    assert r0.status_code == 201

    # 2. Register Node 1
    p1 = {
        "node_id": "node-1",
        "addr": "127.0.0.1",
        "port": 9001,
        "vram_available_mb": 15000.0,
        "vram_total_mb": 15000.0,
        "model_id": "qwen/qwen2.5-7b-instruct",
    }
    r1 = client.post("/register", json=p1)
    assert r1.status_code == 201

    # Wait for timeout window to expire
    time.sleep(0.6)

    # Poll assignment for node-0 -> timeout forces partition across 2 nodes
    res = client.get("/assignment/node-0")
    assert res.status_code == 200
    assert res.json()["status"] == "assigned"
    assert res.json()["cluster_ready"] is True
