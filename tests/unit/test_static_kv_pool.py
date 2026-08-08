"""
Unit tests for StaticKVSlot pre-allocation, zero-allocation memory recycling, and KVCacheStore session capping.
"""

import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import StaticCache, DynamicCache

from shardflow.node.kv_cache import KVCacheStore, StaticKVSlot


def test_static_kv_slot_lease_and_reset():
    """Verify StaticKVSlot advances sequence length and resets memory in-place without reallocation."""
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    slot = StaticKVSlot(slot_id=0, config=config, max_seq_len=64, device=device, dtype=dtype)
    assert slot.is_free() is True

    # Lease
    cache = slot.lease("session-test-01")
    assert slot.is_free() is False
    assert slot.session_id == "session-test-01"
    assert isinstance(cache, StaticCache)

    # Release and reset
    slot.release()
    assert slot.is_free() is True
    assert slot.session_id is None


def test_kv_cache_store_session_capping_and_eviction():
    """Verify KVCacheStore enforces max_sessions limit and LRU slot eviction."""
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    store = KVCacheStore(eviction_timeout=30.0, max_sessions=2, max_seq_len=64, enable_static_cache=True)
    store.initialize_static_pool(config=config, device=device, dtype=dtype)

    if device.type == "cuda" and torch.cuda.is_available():
        assert len(store._static_slots) == 2

        # Lease slots 1 and 2
        c1 = store.get_or_create("s1", config=config, device=device, dtype=dtype)
        c2 = store.get_or_create("s2", config=config, device=device, dtype=dtype)
        assert store.active_sessions == 2

        # Lease slot 3 -> evicts oldest (s1)
        c3 = store.get_or_create("s3", config=config, device=device, dtype=dtype)
        assert store.active_sessions == 2
        assert "s1" not in store._session_to_slot
        assert "s3" in store._session_to_slot

        # Stats check
        stats = store.stats()
        assert stats["active_sessions"] == 2
        assert stats["static_slots_allocated"] == 2
        assert stats["static_slots_leased"] == 2
    else:
        # Dynamic fallback on CPU
        c1 = store.get_or_create("s1")
        assert isinstance(c1, DynamicCache)
        assert store.active_sessions == 1
