"""
Comprehensive test suite for ShardFlow v2 Control / Data Plane Separation.

Validates:
1. START_SESSION and STREAM_TOKEN wire encoding/decoding
2. StreamReceiverServer queue dispatching and session management
3. Peer-to-peer decode loop in data plane (Node 0 driving generation)
4. Terminal node direct stream-back over TCP to Gateway
5. Nucleus sampling numerical stability (guaranteed top-1 preservation)
6. Queued session cancellation in RequestScheduler
"""

import asyncio
import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig

from shardflow.transport.protocol import (
    PROTOCOL_VERSION,
    MessageType,
    TensorMessage,
    encode_message,
    decode_message,
)
from shardflow.transport.connection import StreamReceiverServer, NodeServer, NodeClient
from shardflow.node.layer_loader import ModelSlice
from shardflow.node.node import PipelineNode
from shardflow.orchestrator.sampler import sample_next_token
from shardflow.scheduler.scheduler import RequestScheduler
from shardflow.scheduler.session import Session, SessionState


def test_protocol_v2_constants():
    """Verify v2 protocol version and new message types."""
    assert PROTOCOL_VERSION == 2
    assert MessageType.STREAM_TOKEN == 0x06
    assert MessageType.START_SESSION == 0x05


def test_encode_decode_stream_token():
    """Verify STREAM_TOKEN message serialization with finish_reason and is_eos."""
    msg = TensorMessage(
        msg_type=MessageType.STREAM_TOKEN,
        session_id="stream-sess-001",
        token_id=78912,
        is_eos=True,
        finish_reason="stop",
    )
    encoded = encode_message(msg)
    decoded = decode_message(encoded[8:])

    assert decoded.msg_type == MessageType.STREAM_TOKEN
    assert decoded.session_id == "stream-sess-001"
    assert decoded.token_id == 78912
    assert decoded.is_eos is True
    assert decoded.finish_reason == "stop"


def test_encode_decode_start_session_v2():
    """Verify v2 START_SESSION message serialization with eos_token_id."""
    msg = TensorMessage(
        msg_type=MessageType.START_SESSION,
        session_id="start-sess-002",
        prompt_tokens=[101, 2054, 2003, 1037],
        max_tokens=64,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        eos_token_id=102,
        stream_back_host="10.0.0.1",
        stream_back_port=9600,
    )
    encoded = encode_message(msg)
    decoded = decode_message(encoded[8:])

    assert decoded.msg_type == MessageType.START_SESSION
    assert decoded.session_id == "start-sess-002"
    assert decoded.prompt_tokens == [101, 2054, 2003, 1037]
    assert decoded.max_tokens == 64
    assert pytest.approx(decoded.temperature, abs=1e-3) == 0.7
    assert decoded.top_k == 40
    assert pytest.approx(decoded.top_p, abs=1e-3) == 0.9
    assert decoded.eos_token_id == 102
    assert decoded.stream_back_host == "10.0.0.1"
    assert decoded.stream_back_port == 9600


@pytest.mark.asyncio
async def test_stream_receiver_server_dispatch():
    """Verify Gateway StreamReceiverServer dispatches incoming STREAM_TOKENs to session queues."""
    receiver = StreamReceiverServer(host="127.0.0.1", port=0)
    bound_port = await receiver.start()
    assert bound_port > 0

    session_id = "test-stream-dispatch"
    q = receiver.register_session(session_id)

    client = NodeClient("127.0.0.1", bound_port)
    await client.connect()

    # Send token 1
    await client.send(TensorMessage(
        msg_type=MessageType.STREAM_TOKEN,
        session_id=session_id,
        token_id=42,
        is_eos=False,
    ))

    # Send token 2 with EOS
    await client.send(TensorMessage(
        msg_type=MessageType.STREAM_TOKEN,
        session_id=session_id,
        token_id=43,
        is_eos=True,
        finish_reason="stop",
    ))

    msg1 = await asyncio.wait_for(q.get(), timeout=2.0)
    assert msg1.token_id == 42
    assert msg1.is_eos is False

    msg2 = await asyncio.wait_for(q.get(), timeout=2.0)
    assert msg2.token_id == 43
    assert msg2.is_eos is True
    assert msg2.finish_reason == "stop"

    receiver.unregister_session(session_id)
    await client.close()
    await receiver.stop()


def test_sampler_nucleus_zero_probability_safety():
    """Verify sample_next_token never crashes on extreme top_p or peaked distributions."""
    logits = torch.tensor([-100.0, -100.0, -100.0, 50.0, -100.0])
    token = sample_next_token(logits, temperature=0.7, top_p=0.01)
    assert token == 3

    # All identical logits with tiny top_p
    flat_logits = torch.ones(100)
    token2 = sample_next_token(flat_logits, temperature=1.0, top_p=0.001)
    assert 0 <= token2 < 100


@pytest.mark.asyncio
async def test_request_scheduler_cancel_queued_session():
    """Verify RequestScheduler cancels sessions still waiting in the pending queue."""
    scheduler = RequestScheduler(max_concurrent_sessions=1)
    s1 = Session(prompt="Session 1", max_tokens=5)
    s2 = Session(prompt="Session 2", max_tokens=5)

    await scheduler.submit_request(s1)
    await scheduler.submit_request(s2)

    # Cancel s2 while it is in the queue
    assert scheduler.cancel_session(s2.session_id) is True
    assert s2.state == SessionState.CANCELLED

    # Dequeue s1 and verify s2 is cancelled
    deq1 = await scheduler.pending_queue.get()
    assert deq1.session_id == s1.session_id

    deq2 = await scheduler.pending_queue.get()
    assert deq2.session_id == s2.session_id
    assert deq2.state == SessionState.CANCELLED


@pytest.mark.asyncio
async def test_v2_peer_to_peer_data_plane_cluster():
    """
    Full integration test of v2 Control / Data plane:
    - Node 0 owns embedding and decode loop
    - Node 1 owns LM head and direct stream-back to Gateway
    - Gateway initiates via START_SESSION and receives streamed tokens over TCP
    """
    hidden_dim = 32
    vocab_size = 100

    class DummyEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(vocab_size, hidden_dim))
        def forward(self, x):
            return torch.nn.functional.embedding(x, self.weight)

    class DummyLayer(nn.Module):
        def forward(self, x, **kwargs):
            return x

    class DummyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, vocab_size)
        def forward(self, x):
            return self.linear(x)

    # Start Gateway StreamReceiverServer
    stream_receiver = StreamReceiverServer(host="127.0.0.1", port=0)
    gw_stream_port = await stream_receiver.start()

    # Create Node 1 (Terminal Node: layers 1..2, LM Head)
    node1_slice = ModelSlice(
        layers=nn.ModuleList([DummyLayer()]),
        config=None,
        layer_start=1,
        layer_end=2,
        lm_head=DummyHead(),
    )
    node1 = PipelineNode(
        model_slice=node1_slice,
        is_first_node=False,
        is_last_node=True,
        listen_host="127.0.0.1",
        listen_port=0,
    )
    await node1.start()
    node1_port = node1._server._server.sockets[0].getsockname()[1]

    # Create Node 0 (Data Plane Controller: layers 0..1, Embed tokens)
    node0_slice = ModelSlice(
        layers=nn.ModuleList([DummyLayer()]),
        config=None,
        layer_start=0,
        layer_end=1,
        embed_tokens=DummyEmbedding(),
    )
    node0 = PipelineNode(
        model_slice=node0_slice,
        is_first_node=True,
        is_last_node=False,
        next_node_host="127.0.0.1",
        next_node_port=node1_port,
        listen_host="127.0.0.1",
        listen_port=0,
    )
    await node0.start()
    node0_port = node0._server._server.sockets[0].getsockname()[1]

    # Gateway connects to Node 0 and starts session
    session_id = "v2-e2e-session-test"
    stream_q = stream_receiver.register_session(session_id)

    gw_to_node0 = NodeClient("127.0.0.1", node0_port)
    await gw_to_node0.connect()

    start_msg = TensorMessage(
        msg_type=MessageType.START_SESSION,
        session_id=session_id,
        prompt_tokens=[1, 2, 3],
        max_tokens=5,
        temperature=0.0,  # greedy
        stream_back_host="127.0.0.1",
        stream_back_port=gw_stream_port,
    )

    # Gateway sends START_SESSION once (exits per-token critical path)
    await gw_to_node0.send(start_msg)

    # Collect streamed tokens from StreamReceiverServer
    received_tokens = []
    while True:
        token_msg = await asyncio.wait_for(stream_q.get(), timeout=5.0)
        if token_msg.is_eos or (token_msg.finish_reason is not None and token_msg.finish_reason != ""):
            break
        received_tokens.append(token_msg.token_id)
        if len(received_tokens) >= 5:
            break

    assert len(received_tokens) > 0
    assert all(isinstance(t, int) for t in received_tokens)

    # Cleanup
    stream_receiver.unregister_session(session_id)
    await gw_to_node0.close()
    await node0.stop()
    await node1.stop()
    await stream_receiver.stop()


@pytest.mark.asyncio
async def test_gateway_v2_chat_completions_json_and_sse():
    """Verify FastAPI Gateway /v1/chat/completions endpoint works for JSON and SSE."""
    from fastapi.testclient import TestClient
    from shardflow.gateway.app import app as gateway_app, set_orchestrator
    from shardflow.registry.app import _reset_registry_state
    from shardflow.orchestrator.orchestrator import Orchestrator

    _reset_registry_state()

    hidden_dim = 32
    vocab_size = 100

    class DummyTokenizer:
        vocab_size = 100
        eos_token_id = 99
        def encode(self, text, **kwargs):
            return [1, 2, 3]
        def decode(self, token_ids, **kwargs):
            return " hello"
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    class DummyEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(vocab_size, hidden_dim))
        def forward(self, x):
            return torch.nn.functional.embedding(x, self.weight)

    class DummyLayer(nn.Module):
        def forward(self, x, **kwargs):
            return x

    class DummyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(hidden_dim, vocab_size)
        def forward(self, x):
            return self.linear(x)

    node0_slice = ModelSlice(
        layers=nn.ModuleList([DummyLayer()]),
        config=None,
        layer_start=0,
        layer_end=1,
        embed_tokens=DummyEmbedding(),
        lm_head=DummyHead(),
    )
    node0 = PipelineNode(
        model_slice=node0_slice,
        is_first_node=True,
        is_last_node=True,
        listen_host="127.0.0.1",
        listen_port=0,
    )
    await node0.start()
    node0_port = node0._server._server.sockets[0].getsockname()[1]

    orch = Orchestrator(model_path="dummy", node_addresses=[("127.0.0.1", node0_port)])
    orch.tokenizer = DummyTokenizer()
    orch._node0_client = NodeClient("127.0.0.1", node0_port)
    await orch._node0_client.connect()
    set_orchestrator(orch)

    http_client = TestClient(gateway_app)

    # Test non-streaming JSON response
    resp = http_client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 3,
            "stream": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]

    # Test SSE streaming response
    resp_sse = http_client.post(
        "/v1/chat/completions",
        json={
            "model": "dummy",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 3,
            "stream": True,
        },
    )
    assert resp_sse.status_code == 200
    assert "data:" in resp_sse.text
    assert "[DONE]" in resp_sse.text

    await orch._node0_client.close()
    await node0.stop()
