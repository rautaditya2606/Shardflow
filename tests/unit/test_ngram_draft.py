"""
Unit tests for NGramDraftSampler (Prompt-Lookup Speculation).
"""

from shardflow.node.ngram_draft import NGramDraftSampler


def test_ngram_draft_finds_continuation():
    sampler = NGramDraftSampler(max_ngram_size=3, min_ngram_size=1, spec_k=4)
    # History containing "Fibonacci numbers using dynamic programming"
    history = [10, 20, 30, 40, 50, 60, 70, 80, 10, 20]
    
    # Current suffix is [10, 20] which matches the earlier [10, 20] at index 0..1
    candidates = sampler.find_candidates(history, k=4)
    assert candidates == [30, 40, 50, 60]


def test_ngram_draft_empty_when_no_match():
    sampler = NGramDraftSampler(max_ngram_size=3, min_ngram_size=1, spec_k=4)
    # Unique tokens, no repeated n-grams
    history = [1, 2, 3, 4, 5, 6, 7]
    candidates = sampler.find_candidates(history, k=4)
    assert candidates == []


def test_ngram_draft_short_history_safety():
    sampler = NGramDraftSampler(max_ngram_size=3, min_ngram_size=1, spec_k=4)
    assert sampler.find_candidates([], k=4) == []
    assert sampler.find_candidates([1], k=4) == []
