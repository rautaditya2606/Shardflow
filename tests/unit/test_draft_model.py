"""
Unit tests for DraftSampler in ShardFlow.
Tests DynamicCache / StaticCache rewinding, prefill, and greedy draft generation.
"""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache, StaticCache
from shardflow.node.draft_model import (
    DraftSampler,
    rewind_dynamic_cache,
    rewind_static_cache,
    rewind_kv_cache,
)


def test_rewind_dynamic_cache():
    """Verify rewind_dynamic_cache truncates key and value caches properly."""
    cache = DynamicCache()
    # Populate with dummy layer
    k = torch.randn(1, 4, 20, 64)
    v = torch.randn(1, 4, 20, 64)
    cache.update(k, v, layer_idx=0)
    
    assert cache.get_seq_length(0) == 20
    rewind_dynamic_cache(cache, target_seq_len=12)
    assert cache.get_seq_length(0) == 12


def test_rewind_static_cache():
    """Verify rewind_static_cache updates _seen_tokens and zeroes uncommitted slots."""
    config = AutoConfig.for_model("llama")
    config.hidden_size = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.num_hidden_layers = 2
    
    device = torch.device("cpu")
    cache = StaticCache(config=config, max_batch_size=1, max_cache_len=32, device=device, dtype=torch.float32)
    
    # Initialize cache layers via update
    k = torch.ones(1, 4, 20, 16)
    v = torch.ones(1, 4, 20, 16)
    for i in range(len(cache.layers)):
        cache.update(k, v, i, cache_kwargs={"cache_position": torch.arange(20)})
    
    cache._seen_tokens = 20
    rewind_static_cache(cache, target_seq_len=10)
    assert cache._seen_tokens == 10
    
    # Check slots >= 10 are zeroed
    for layer in cache.layers:
        if layer.keys is not None:
            assert torch.all(layer.keys[:, :, 10:, :] == 0.0)
            assert torch.all(layer.values[:, :, 10:, :] == 0.0)
            assert torch.all(layer.keys[:, :, :10, :] == 1.0)


def test_universal_rewind_kv_cache():
    """Verify universal rewind_kv_cache handles both cache types."""
    dynamic_c = DynamicCache()
    dynamic_c.update(torch.randn(1, 2, 15, 32), torch.randn(1, 2, 15, 32), layer_idx=0)
    rewind_kv_cache(dynamic_c, target_seq_len=8)
    assert dynamic_c.get_seq_length(0) == 8


def test_draft_sampler_cpu_fallback(monkeypatch):
    """Test DraftSampler initialization and generation on CPU (eager fallback)."""
    # Create a tiny dummy causal LM
    config = AutoConfig.for_model("qwen2")
    config.vocab_size = 100
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    
    dummy_model = AutoModelForCausalLM.from_config(config)
    dummy_model.eval()
    
    # Monkeypatch AutoModelForCausalLM.from_pretrained to return dummy model
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: dummy_model)
    
    sampler = DraftSampler(
        model_path="dummy/qwen2-0.5B",
        device=torch.device("cpu"),
        dtype=torch.float32,
        spec_k=4,
    )
    
    assert sampler.seq_len == 0
    prompt = [10, 20, 30, 40]
    sampler.prefill(prompt)
    assert sampler.seq_len == 4
    
    drafts = sampler.generate_drafts(current_token=40, k=4)
    assert len(drafts) == 4
    assert sampler.seq_len == 8
    
    sampler.rewind(target_seq_len=6)
    assert sampler.seq_len == 6
