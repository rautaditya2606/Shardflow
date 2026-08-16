"""
ShardFlow v2 — Prompt-Lookup / N-gram Speculative Sampler.

Finds candidate tokens from prompt and generated context in <0.1 ms with zero GPU compute.
Replaces heavy neural draft models to maximize accepted tokens per network round-trip.
"""

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class NGramDraftSampler:
    """
    Ultra-fast N-gram / Prompt-Lookup candidate generator.
    Scans past token history for matching n-gram prefixes to predict candidate continuation.
    """

    def __init__(
        self,
        max_ngram_size: int = 3,
        min_ngram_size: int = 1,
        spec_k: int = 4,
    ):
        self.max_ngram_size = max_ngram_size
        self.min_ngram_size = min_ngram_size
        self.spec_k = spec_k

    def find_candidates(self, token_history: List[int], k: Optional[int] = None) -> List[int]:
        """
        Find up to k candidate continuation tokens matching the latest suffix in token_history.

        Args:
            token_history: Full list of tokens (prompt + generated so far)
            k: Maximum number of candidate tokens to return

        Returns:
            List of candidate token IDs (0 to k tokens)
        """
        k = k or self.spec_k
        if not token_history or len(token_history) < 2 or k <= 0:
            return []

        hist_len = len(token_history)

        # Try longest n-gram match down to min_ngram_size
        for n in range(min(self.max_ngram_size, hist_len - 1), self.min_ngram_size - 1, -1):
            target_ngram = token_history[-n:]

            # Scan history from right to left (most recent occurrences first)
            # Exclude the current suffix at the very end
            for i in range(hist_len - n - 1, -1, -1):
                if token_history[i : i + n] == target_ngram:
                    # Match found at index i! Subsequent tokens start at i + n
                    candidates = token_history[i + n : i + n + k]
                    if candidates:
                        return candidates

        return []
