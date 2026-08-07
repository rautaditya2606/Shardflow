"""
Unit tests for Inverted Topology Registry and Partial-Token Fault Tolerance.
"""

import pytest
from fastapi.testclient import TestClient
from shardflow.registry.app import app as registry_app, _nodes
from shardflow.orchestrator.orchestrator import PartialGenerationError

client = TestClient(registry_app)


@pytest.fixture(autouse=True)
def clear_registry_nodes():
    from shardflow.registry.app import _reset_registry_state
    _reset_registry_state()
    yield
    _reset_registry_state()


def test_assignment_pending_until_cluster_ready():
    """Nodes must not receive final assignment until EXPECTED_NODES register."""
    node0 = {
        "node_id": "node-0",
        "addr": "127.0.0.1",
        "port": 9000,
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    client.post("/register", json=node0)

    pending = client.get("/assignment/node-0")
    assert pending.status_code == 202
    assert pending.json()["status"] == "pending"
    assert pending.json()["cluster_ready"] is False

    node1 = {
        "node_id": "node-1",
        "addr": "127.0.0.1",
        "port": 9001,
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    client.post("/register", json=node1)

    assigned = client.get("/assignment/node-0")
    assert assigned.status_code == 200
    data = assigned.json()
    assert data["status"] == "assigned"
    assert data["cluster_ready"] is True
    assert data["layer_start"] == 0
    assert data["layer_end"] == 11
    assert data["next_node_host"] == "127.0.0.1"
    assert data["topology_version"] >= 1


def test_heartbeat_includes_topology_version():
    node0 = {
        "node_id": "node-0",
        "addr": "127.0.0.1",
        "port": 9000,
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    client.post("/register", json=node0)
    hb = client.post("/heartbeat", json={"node_id": "node-0"})
    assert hb.status_code == 200
    assert "topology_version" in hb.json()
    assert hb.json()["cluster_ready"] is False


def test_auto_split_tinyllama_2_nodes():
    """Verify 22-layer TinyLlama splits 11 / 11 across 2 nodes."""
    node0 = {
        "node_id": "node-0",
        "addr": "127.0.0.1",
        "port": 9000,
        "vram_available_mb": 15000.0,
        "vram_total_mb": 15000.0,
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    r0 = client.post("/register", json=node0)
    assert r0.status_code == 201
    d0 = r0.json()
    assert d0["layer_start"] == 0
    assert d0["layer_end"] == 11
    assert d0["is_first_node"] is True
    assert d0["is_last_node"] is False

    node1 = {
        "node_id": "node-1",
        "addr": "127.0.0.1",
        "port": 9001,
        "vram_available_mb": 15000.0,
        "vram_total_mb": 15000.0,
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    r1 = client.post("/register", json=node1)
    assert r1.status_code == 201
    d1 = r1.json()
    assert d1["layer_start"] == 11
    assert d1["layer_end"] == 22
    assert d1["is_first_node"] is False
    assert d1["is_last_node"] is True

    # Re-verify node 0 after rebalance
    topo = client.get("/topology").json()
    assert topo["total_nodes"] == 2
    n0 = topo["nodes"][0]
    n1 = topo["nodes"][1]

    assert n0["layer_start"] == 0 and n0["layer_end"] == 11
    assert n0["is_first_node"] is True and n0["is_last_node"] is False
    assert n0["next_node_host"] == "127.0.0.1" and n0["next_node_port"] == 9001

    assert n1["layer_start"] == 11 and n1["layer_end"] == 22
    assert n1["is_first_node"] is False and n1["is_last_node"] is True
    assert n1["next_node_host"] is None


def test_auto_split_llama3_8b_2_nodes():
    """Verify 32-layer Llama-3 8B splits 16 / 16 across 2 nodes."""
    node0 = {
        "node_id": "node-0",
        "addr": "10.0.0.1",
        "port": 9000,
        "vram_available_mb": 15000.0,
        "model_id": "meta-llama/Meta-Llama-3-8B",
    }
    node1 = {
        "node_id": "node-1",
        "addr": "10.0.0.2",
        "port": 9000,
        "vram_available_mb": 15000.0,
        "model_id": "meta-llama/Meta-Llama-3-8B",
    }
    client.post("/register", json=node0)
    r1 = client.post("/register", json=node1)
    assert r1.status_code == 201

    topo = client.get("/topology").json()
    assert topo["nodes"][0]["layer_start"] == 0
    assert topo["nodes"][0]["layer_end"] == 17
    assert topo["nodes"][1]["layer_start"] == 17
    assert topo["nodes"][1]["layer_end"] == 32


def test_auto_split_llama3_8b_3_nodes():
    """Verify 32-layer Llama-3 8B splits 11 / 11 / 10 across 3 nodes."""
    for i in range(3):
        client.post("/register", json={
            "node_id": f"node-{i}",
            "addr": f"10.0.0.{i+1}",
            "port": 9000,
            "model_id": "meta-llama/Meta-Llama-3-8B",
        })

    topo = client.get("/topology").json()
    assert topo["total_nodes"] == 3
    assert topo["nodes"][0]["layer_start"] == 0 and topo["nodes"][0]["layer_end"] == 11
    assert topo["nodes"][1]["layer_start"] == 11 and topo["nodes"][1]["layer_end"] == 22
    assert topo["nodes"][2]["layer_start"] == 22 and topo["nodes"][2]["layer_end"] == 32


def test_partial_generation_error_exception():
    """Verify PartialGenerationError holds partial text and original error."""
    orig_err = ConnectionResetError("Connection lost to node 1")
    p_err = PartialGenerationError(
        message="Node failure during step 5",
        partial_text="Hello world partial",
        tokens_generated=3,
        original_error=orig_err,
    )

    assert p_err.partial_text == "Hello world partial"
    assert p_err.tokens_generated == 3
    assert p_err.original_error == orig_err
