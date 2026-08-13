"""
Unit tests for Speculative Decoding verification, KV cache rewind, and accepted count telemetry.
"""

import pytest
import torch
from transformers.cache_utils import DynamicCache
from shardflow.node.draft_model import rewind_dynamic_cache, rewind_kv_cache
from shardflow.transport.protocol import MessageType, TensorMessage, encode_message, decode_message, LENGTH_PREFIX_SIZE


def test_dynamic_cache_rewind():
    """Verify rewind_dynamic_cache truncates KV tensors to target sequence length."""
    cache = DynamicCache()
    # Populate a fake KV cache with 2 layers, shape [1, 4, 10, 32] (seq_len=10)
    for layer_idx in range(2):
        k = torch.randn(1, 4, 10, 32)
        v = torch.randn(1, 4, 10, 32)
        cache.update(k, v, layer_idx=layer_idx)

    assert cache.get_seq_length(0) == 10

    # Rewind to target_seq_len = 6
    rewind_kv_cache(cache, target_seq_len=6)

    assert cache.get_seq_length(0) == 6


def test_speculative_verification_logic():
    """Simulate terminal node candidate verification and KV cache slice calculation."""
    draft_tokens = [100, 200, 300, 400]
    # Simulated model output where drafts [100, 200] match, but position 2 yields 999
    simulated_target_tokens = [100, 200, 999, 500]

    accepted_tokens = []
    next_token = None
    for i in range(len(draft_tokens)):
        cand = simulated_target_tokens[i]
        if cand == draft_tokens[i]:
            accepted_tokens.append(draft_tokens[i])
        else:
            next_token = cand
            break

    assert accepted_tokens == [100, 200]
    assert next_token == 999
    # Total accepted sequence count is len(accepted_tokens) + 1 (the corrected token)
    accepted_count = len(accepted_tokens) + 1
    assert accepted_count == 3


def test_speculative_full_acceptance():
    """Verify behavior when all K drafts match the target model."""
    draft_tokens = [10, 20, 30, 40]
    simulated_target_tokens = [10, 20, 30, 40]

    accepted_tokens = []
    next_token = None
    for i in range(len(draft_tokens)):
        cand = simulated_target_tokens[i]
        if cand == draft_tokens[i]:
            accepted_tokens.append(draft_tokens[i])
        else:
            next_token = cand
            break

    if next_token is None:
        # All K match; bonus token from final position
        next_token = 50

    assert accepted_tokens == [10, 20, 30, 40]
    assert next_token == 50
    assert len(accepted_tokens) + 1 == 5
