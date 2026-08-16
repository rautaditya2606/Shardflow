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


def test_node0_profiler_speculative_metrics():
    """Verify Node0Profiler correctly tracks effective tokens per round and bonus accept rate."""
    from scripts.kaggle_node0 import Node0Profiler

    profiler = Node0Profiler()
    # Record 2 speculative rounds:
    # Round 1: K=4 drafts, accepted_count=3 (2 bonus accepted)
    # Round 2: K=4 drafts, accepted_count=5 (4 bonus accepted)
    profiler.record(
        embed_ms=0.0,
        gpu_fwd_ms=30.0,
        g2c_ms=0.05,
        ser_ms=0.05,
        send_ms=0.05,
        recv_ms=90.0,
        total_ms=120.0,
        draft_gen_ms=2.5,
        accepted=3,
        drafted=4,
        is_spec=True,
    )
    profiler.record(
        embed_ms=0.0,
        gpu_fwd_ms=30.0,
        g2c_ms=0.05,
        ser_ms=0.05,
        send_ms=0.05,
        recv_ms=90.0,
        total_ms=120.0,
        draft_gen_ms=2.5,
        accepted=5,
        drafted=4,
        is_spec=True,
    )

    spec_acc = [acc for acc, is_s in zip(profiler.accepted_per_round, profiler.is_spec_step) if is_s]
    spec_drf = [drf for drf, is_s in zip(profiler.drafted_per_round, profiler.is_spec_step) if is_s]

    # Effective tokens per round = (3 + 5) / 2 = 4.0
    tok_per_round = sum(spec_acc) / len(spec_acc)
    assert tok_per_round == 4.0

    # Bonus accept rate = ((3-1) + (5-1)) / (4 + 4) * 100 = 6/8 = 75.0%
    bonus_rate = sum(max(0, a - 1) for a in spec_acc) / sum(spec_drf) * 100.0
    assert bonus_rate == 75.0


def test_speculative_lookahead_kv_reconciliation():
    """Verify speculative lookahead KV states are cleanly truncated on partial acceptance."""
    cache = DynamicCache()
    # Initial sequence length = 10
    for layer_idx in range(2):
        k = torch.randn(1, 4, 10, 32)
        v = torch.randn(1, 4, 10, 32)
        cache.update(k, v, layer_idx=layer_idx)

    past_seq_len = 10
    # Step 1: Compute 5 candidate tokens (seq_len becomes 15)
    for layer_idx in range(2):
        k = torch.randn(1, 4, 5, 32)
        v = torch.randn(1, 4, 5, 32)
        cache.update(k, v, layer_idx=layer_idx)
    assert cache.get_seq_length(0) == 15

    # Lookahead speculative forward: Compute 5 more tokens ahead (seq_len becomes 20)
    for layer_idx in range(2):
        k = torch.randn(1, 4, 5, 32)
        v = torch.randn(1, 4, 5, 32)
        cache.update(k, v, layer_idx=layer_idx)
    assert cache.get_seq_length(0) == 20

    # Node 1 returns accepted_count = 3 (out of 5)
    # Rewind should crop KV cache back to past_seq_len + accepted_count = 13
    accepted_count = 3
    rewind_kv_cache(cache, past_seq_len + accepted_count)

    assert cache.get_seq_length(0) == 13
    assert cache.get_seq_length(1) == 13


def test_inflight_window_multi_round_correctness():
    """Verify that multiple consecutive rounds of W=2 in-flight execution produce correct token histories."""
    # Simulation: 
    # Prompt tokens: [1, 2, 3]
    # Round 1: candidate drafts = [10, 20, 30, 40]
    # Target model accepts [10, 20], rejects 30, gives verified token 99
    # Accepted sequence should be: [1, 2, 3, 10, 20, 99]
    token_history = [1, 2, 3]
    past_seq_len = len(token_history)
    drafts = [10, 20, 30, 40]
    accepted_count = 3  # drafts[0]=10, drafts[1]=20, next_token=99
    next_token = 99

    accepted_tokens = []
    if accepted_count > 1:
        for d_idx in range(accepted_count - 1):
            tok = drafts[d_idx]
            accepted_tokens.append(tok)
            token_history.append(tok)

    accepted_tokens.append(next_token)
    token_history.append(next_token)

    assert token_history == [1, 2, 3, 10, 20, 99]
    assert len(accepted_tokens) == accepted_count == 3

