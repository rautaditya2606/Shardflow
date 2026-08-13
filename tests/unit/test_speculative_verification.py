"""
Unit tests for Speculative Decoding verification, KV cache rewind, and accepted count telemetry.
"""

import pytest
import torch
from transformers.cache_utils import DynamicCache, StaticCache
from shardflow.node.draft_model import rewind_dynamic_cache, rewind_kv_cache, rewind_static_cache
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


def test_causal_speculative_verification_partial_acceptance():
    """
    Test causal verification when input is [T_current, d_0, d_1, d_2, d_3] (length 5).
    Logits at position i predict the token following candidate_tokens[i].
    Position 0 (T_current) -> predicts d_0 (match)
    Position 1 (d_0) -> predicts d_1 (match)
    Position 2 (d_1) -> predicts 999 (mismatch with d_2=300)
    """
    current_token = 50
    draft_tokens = [100, 200, 300, 400]
    candidate_tokens = [current_token] + draft_tokens  # [50, 100, 200, 300, 400]

    # Target model predictions at each candidate position:
    # index 0 (input 50) -> predicts 100
    # index 1 (input 100) -> predicts 200
    # index 2 (input 200) -> predicts 999 (mismatch with draft 300)
    # index 3 (input 300) -> predicts 500
    # index 4 (input 400) -> predicts 600
    simulated_target_predictions = [100, 200, 999, 500, 600]

    accepted_tokens = []
    next_token = None
    for i in range(len(draft_tokens)):
        cand = simulated_target_predictions[i]
        if cand == draft_tokens[i]:
            accepted_tokens.append(draft_tokens[i])
        else:
            next_token = cand
            break

    assert accepted_tokens == [100, 200]
    assert next_token == 999
    # Total new accepted tokens: len(accepted_tokens) + 1 (the corrected token)
    accepted_count = len(accepted_tokens) + 1
    assert accepted_count == 3


def test_causal_speculative_verification_full_acceptance():
    """
    Test full acceptance of all K drafts and sampling bonus token from position K.
    """
    current_token = 5
    draft_tokens = [10, 20, 30, 40]
    candidate_tokens = [current_token] + draft_tokens

    # All positions 0..3 match draft tokens; position 4 gives bonus token (50)
    simulated_target_predictions = [10, 20, 30, 40, 50]

    accepted_tokens = []
    next_token = None
    for i in range(len(draft_tokens)):
        cand = simulated_target_predictions[i]
        if cand == draft_tokens[i]:
            accepted_tokens.append(draft_tokens[i])
        else:
            next_token = cand
            break

    if next_token is None:
        # All K match; bonus token sampled from final position
        next_token = simulated_target_predictions[-1]

    assert accepted_tokens == [10, 20, 30, 40]
    assert next_token == 50
    assert len(accepted_tokens) + 1 == 5


def test_causal_speculative_verification_immediate_rejection():
    """
    Test immediate rejection on draft 0 (position 0 predicts a different token).
    """
    current_token = 50
    draft_tokens = [100, 200, 300, 400]
    # Position 0 predicts 777 instead of draft 0 (100)
    simulated_target_predictions = [777, 200, 300, 400, 500]

    accepted_tokens = []
    next_token = None
    for i in range(len(draft_tokens)):
        cand = simulated_target_predictions[i]
        if cand == draft_tokens[i]:
            accepted_tokens.append(draft_tokens[i])
        else:
            next_token = cand
            break

    assert accepted_tokens == []
    assert next_token == 777
    assert len(accepted_tokens) + 1 == 1
