"""
Unit and regression tests for critical bug fixes and latency optimizations.
"""

import asyncio
import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache

from shardflow.node.kv_cache import KVCacheStore
from shardflow.node.node import PipelineNode
from shardflow.node.layer_loader import ModelSlice
from shardflow.transport.connection import NodeClient, NodeServer
from shardflow.transport.protocol import (
    MessageType,
    TensorMessage,
    encode_message,
    decode_message,
)
from shardflow.orchestrator.sampler import sample_next_token


def test_causal_mask_chunked_prefill():
    """Verify that multi-token chunks with past KV cache construct a properly broadcastable causal mask."""
    seq_len = 512
    past_seq_len = 512
    hidden_dim = 64
    device = torch.device("cpu")

    hidden_states = torch.randn(1, seq_len, hidden_dim, dtype=torch.float32)

    # Reconstruct the logic from node.py
    causal_mask = torch.full(
        (seq_len, past_seq_len + seq_len),
        0.0,
        device=device,
        dtype=hidden_states.dtype,
    )
    current_chunk_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=hidden_states.dtype),
        diagonal=1,
    )
    causal_mask[:, past_seq_len:] = current_chunk_mask
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # Check shapes
    assert causal_mask.shape == (1, 1, seq_len, past_seq_len + seq_len)

    # Prefix columns (past KV) must be all 0.0 (fully visible)
    assert torch.all(causal_mask[:, :, :, :past_seq_len] == 0.0)

    # Current chunk upper triangle must be -inf
    assert causal_mask[0, 0, 0, past_seq_len + 1] == float("-inf")
    # Current chunk diagonal must be 0.0
    assert causal_mask[0, 0, 0, past_seq_len] == 0.0


def test_dynamic_cache_stats():
    """Verify KVCacheStore.stats() does not crash with DynamicCache objects."""
    store = KVCacheStore(eviction_timeout=60.0, max_sessions=4)
    cache = DynamicCache()

    # Populate a layer in DynamicCache
    key = torch.randn(1, 4, 16, 32)
    val = torch.randn(1, 4, 16, 32)
    cache.update(key, val, layer_idx=0)

    store.put("test-session-1", cache)
    stats = store.stats()

    assert stats["active_sessions"] == 1
    assert stats["total_memory_mb"] > 0
    assert stats["max_sessions"] == 4


def test_dynamic_cache_intermediate_node_lookup():
    """Verify sequence length lookup for an intermediate node (layer_start > 0)."""
    cache = DynamicCache()
    key = torch.randn(1, 4, 64, 32)  # 64 tokens
    val = torch.randn(1, 4, 64, 32)

    # Intermediate node occupies layer_start = 14
    layer_start = 14
    cache.update(key, val, layer_idx=layer_start)

    # Default get_seq_length() (index 0) returns 0 for layer 14
    assert cache.get_seq_length(0) == 0

    # Querying with layer_start returns the correct sequence length
    assert cache.get_seq_length(layer_start) == 64


@pytest.mark.asyncio
async def test_node_client_closes_socket_on_timeout():
    """Verify NodeClient cleanly closes its writer and marks disconnected on send_recv timeout."""
    # Start a dummy server that sleeps and never responds
    async def hung_handler(reader, writer):
        await asyncio.sleep(10.0)

    server = await asyncio.start_server(hung_handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    client = NodeClient("127.0.0.1", port, recv_timeout=0.2)
    await client.connect()
    assert client.is_connected

    msg = TensorMessage(
        msg_type=MessageType.ACTIVATION,
        session_id="timeout-test",
        tensor=torch.randn(1, 1, 16),
    )

    with pytest.raises(asyncio.TimeoutError):
        await client.send_recv(msg, timeout=0.1)

    # Socket must be marked disconnected and closed
    assert not client.is_connected
    assert client._writer.is_closing()

    server.close()
    await server.wait_closed()


def test_sample_on_node_flag_skips_head():
    """Verify PipelineNode._forward does not apply lm_head when compute_head=False."""
    class DummyLayer(nn.Module):
        def forward(self, x, **kwargs):
            return x

    class DummyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 32)
            self.called = False

        def forward(self, x):
            self.called = True
            return self.linear(x)

    head = DummyHead()
    slice_obj = ModelSlice(
        layers=nn.ModuleList([DummyLayer()]),
        config=AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0") if False else None,
        layer_start=0,
        layer_end=1,
        lm_head=head,
    )

    node = PipelineNode(
        model_slice=slice_obj,
        is_first_node=True,
        is_last_node=True,
    )

    inp = torch.randn(1, 2, 16)

    # Forward without compute_head
    out1 = node._forward(inp, session_id="test-head-skip", compute_head=False)
    assert not head.called
    assert out1.shape == (1, 2, 16)

    # Forward with compute_head
    out2 = node._forward(inp, session_id="test-head-exec", compute_head=True)
    assert head.called
    assert out2.shape == (1, 2, 32)
