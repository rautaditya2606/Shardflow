"""
Unit tests for GPU sampling logic: greedy, top-k, nucleus top-p, and zero-probability safety.
"""

import torch
import pytest
from shardflow.orchestrator.sampler import sample_next_token


def test_greedy_sampling():
    """Verify greedy temperature=0.0 selects the exact argmax token."""
    logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
    token = sample_next_token(logits, temperature=0.0)
    assert token == 1


def test_top_k_filtering():
    """Verify top-k limits sampling to the top k highest probability tokens."""
    logits = torch.tensor([0.1, 10.0, 9.0, 0.05, 0.01])
    tokens = set()
    for _ in range(20):
        token = sample_next_token(logits, temperature=1.0, top_k=2)
        tokens.add(token)
    # Only tokens 1 and 2 should ever be chosen
    assert tokens.issubset({1, 2})


def test_nucleus_zero_probability_safety():
    """Verify nucleus top-p filtering handles extreme or zero probability distributions safely."""
    # Degenerate uniform logits
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
    token = sample_next_token(logits, temperature=1.0, top_p=0.9)
    assert token in [0, 1, 2, 3]

    # Single dominant logit
    logits = torch.tensor([-100.0, 100.0, -100.0])
    token = sample_next_token(logits, temperature=0.8, top_p=0.5)
    assert token == 1
