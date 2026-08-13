"""
Speculative Decoding Draft Sampler and KV Cache Rewind Utilities.
Enables generating K candidate tokens locally on Node 0 in a single network roundtrip.
"""

import logging
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache, StaticCache, Cache
from shardflow.orchestrator.sampler import sample_next_token

logger = logging.getLogger(__name__)


def rewind_dynamic_cache(cache: DynamicCache, target_seq_len: int) -> None:
    """Rewind DynamicCache to target_seq_len by slicing key/value tensors in place."""
    # ponytail: native cache.crop() truncates dynamic cache in a single standard call
    if hasattr(cache, "crop"):
        cache.crop(target_seq_len)
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for layer_idx in range(len(cache.key_cache)):
            if cache.key_cache[layer_idx] is not None:
                cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, :target_seq_len, :]
            if cache.value_cache[layer_idx] is not None:
                cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, :target_seq_len, :]


def rewind_static_cache(cache: StaticCache, target_seq_len: int) -> None:
    """Rewind StaticCache to target_seq_len by updating seen_tokens pointer."""
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = target_seq_len


def rewind_kv_cache(cache: Cache, target_seq_len: int) -> None:
    """Universal KV cache rewind for either DynamicCache or StaticCache."""
    if isinstance(cache, StaticCache):
        rewind_static_cache(cache, target_seq_len)
    elif isinstance(cache, DynamicCache):
        rewind_dynamic_cache(cache, target_seq_len)


class DraftSampler:
    """
    Lightweight draft model sampler running on Node 0.
    Generates K candidate tokens locally at high throughput (~80+ TPS).
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        spec_k: int = 4,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.spec_k = spec_k
        self.cache = DynamicCache()

        logger.info("Loading draft model %s on %s (dtype=%s, K=%d)...", model_path, device, dtype, spec_k)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=str(device),
        )
        self.model.eval()

    def reset(self) -> None:
        """Reset draft model KV cache for a new session."""
        self.cache = DynamicCache()

    def rewind(self, target_seq_len: int) -> None:
        """Rewind draft model KV cache after speculative rejection."""
        rewind_dynamic_cache(self.cache, target_seq_len)

    @torch.inference_mode()
    def generate_drafts(
        self,
        current_token: int,
        k: Optional[int] = None,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> List[int]:
        """
        Generate K candidate draft tokens locally.
        Returns: list of K token IDs.
        """
        k = k or self.spec_k
        draft_tokens: List[int] = []
        next_tok = current_token

        for _ in range(k):
            input_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=self.device)
            outputs = self.model(
                input_ids=input_tensor,
                past_key_values=self.cache,
                use_cache=True,
            )
            logits = outputs.logits[0, -1, :]
            sampled_id = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            draft_tokens.append(sampled_id)
            next_tok = sampled_id

        return draft_tokens
