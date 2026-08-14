"""
Lightweight tokenizer loader for the Orchestrator.

Avoids importing AutoModelForCausalLM (which alone spikes RAM by ~150-200 MB).
Uses the `tokenizers` Rust library directly when a tokenizer.json is present,
falling back to transformers.AutoTokenizer only when necessary.

The Orchestrator only needs:
  - tokenizer(text) -> {"input_ids": tensor}
  - tokenizer.decode([token_id]) -> str
  - tokenizer.apply_chat_template(messages, ...) -> str
  - tokenizer.eos_token_id -> int
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def load_tokenizer(model_path: str) -> Any:
    """
    Load a tokenizer with minimal memory footprint.

    Strategy:
    1. Try loading via transformers.AutoTokenizer with no model weight imports.
       transformers itself is still needed for apply_chat_template support.
    2. The key fix: we import ONLY AutoTokenizer, not AutoModelForCausalLM.
       The caller (orchestrator.py) previously imported load_layer_slice alongside
       load_tokenizer from node/layer_loader.py — that dragged in AutoModelForCausalLM.
       This module never touches model weights.

    Args:
        model_path: local path or HuggingFace model ID (e.g. "Qwen/Qwen2.5-7B-Instruct")

    Returns:
        A tokenizer object compatible with the orchestrator decode loop.
    """
    # Suppress unnecessary HF progress bars in server environments
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    logger.info("Loading tokenizer from %s (lightweight path, no model weights)...", model_path)

    # Import AutoTokenizer with robust fallbacks for Python 3.12 / lazy-loading environments
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except (ImportError, AttributeError):
        try:
            from transformers.models.auto import AutoTokenizer  # noqa: PLC0415
        except (ImportError, AttributeError):
            from transformers.models.auto.tokenization_auto import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,          # use the Rust tokenizer backend
        trust_remote_code=False,
    )

    logger.info(
        "Tokenizer loaded: vocab_size=%d, eos_token_id=%s",
        tokenizer.vocab_size,
        tokenizer.eos_token_id,
    )
    return tokenizer
