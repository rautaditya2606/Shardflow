"""
Unit tests for 4-bit in-place meta device layer loader in ShardFlow.
"""

import pytest
import torch
import torch.nn as nn
from accelerate import init_empty_weights
import bitsandbytes as bnb

from shardflow.node.layer_loader import (
    _replace_linear_with_4bit_meta,
    _load_state_dict_into_4bit_slice,
    load_layer_slice,
)


class DummySelfAttention(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(q + k + v)


class DummyMLP(nn.Module):
    def __init__(self, hidden_dim: int = 64, intermediate_dim: int = 128):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class DummyDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_dim)
        self.self_attn = DummySelfAttention(hidden_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)
        self.mlp = DummyMLP(hidden_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x + self.self_attn(self.input_layernorm(x))
        out = h + self.mlp(self.post_attention_layernorm(h))
        return (out,)


def test_replace_linear_with_4bit_meta_zero_ram():
    """Verify that _replace_linear_with_4bit_meta converts all Linear layers to Linear4bit on meta device."""
    with init_empty_weights():
        layer = DummyDecoderLayer(hidden_dim=64)

    # Initial state should be meta-device nn.Linear
    assert isinstance(layer.self_attn.q_proj, nn.Linear)
    assert not isinstance(layer.self_attn.q_proj, bnb.nn.Linear4bit)
    assert layer.self_attn.q_proj.weight.device.type == "meta"

    # Convert on meta device
    _replace_linear_with_4bit_meta(layer, compute_dtype=torch.float16)

    # All linear layers must now be bnb.nn.Linear4bit on meta device
    assert isinstance(layer.self_attn.q_proj, bnb.nn.Linear4bit)
    assert isinstance(layer.self_attn.k_proj, bnb.nn.Linear4bit)
    assert isinstance(layer.self_attn.v_proj, bnb.nn.Linear4bit)
    assert isinstance(layer.self_attn.o_proj, bnb.nn.Linear4bit)
    assert isinstance(layer.mlp.gate_proj, bnb.nn.Linear4bit)
    assert isinstance(layer.mlp.up_proj, bnb.nn.Linear4bit)
    assert isinstance(layer.mlp.down_proj, bnb.nn.Linear4bit)

    # Weights must still be meta-device Params4bit
    assert layer.self_attn.q_proj.weight.device.type == "meta"
    assert isinstance(layer.self_attn.q_proj.weight, bnb.nn.Params4bit)


def test_load_state_dict_into_4bit_slice_forward_pass():
    """Verify that in-place state_dict loading quantizes weights and runs a valid forward pass."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 64

    # 1. Instantiate meta layer
    with init_empty_weights():
        layer = DummyDecoderLayer(hidden_dim=hidden_dim)

    _replace_linear_with_4bit_meta(layer, compute_dtype=torch.float16)
    extracted_layers = nn.ModuleList([layer])

    # 2. Synthetic state dict matching layer 0
    state_dict = {
        "model.layers.0.input_layernorm.weight": torch.ones(hidden_dim, dtype=torch.float16),
        "model.layers.0.input_layernorm.bias": torch.zeros(hidden_dim, dtype=torch.float16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden_dim, hidden_dim, dtype=torch.float16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(hidden_dim, hidden_dim, dtype=torch.float16),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(hidden_dim, hidden_dim, dtype=torch.float16),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden_dim, hidden_dim, dtype=torch.float16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(hidden_dim, dtype=torch.float16),
        "model.layers.0.post_attention_layernorm.bias": torch.zeros(hidden_dim, dtype=torch.float16),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(128, hidden_dim, dtype=torch.float16),
        "model.layers.0.mlp.up_proj.weight": torch.randn(128, hidden_dim, dtype=torch.float16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(hidden_dim, 128, dtype=torch.float16),
    }

    # 3. Load state dict into 4-bit slice
    _load_state_dict_into_4bit_slice(
        extracted_layers=extracted_layers,
        state_dict=state_dict,
        layer_start=0,
        layer_end=1,
        device=device,
        compute_dtype=torch.float16,
    )

    # 4. Verify weight placement & Params4bit quant_state
    q_proj = extracted_layers[0].self_attn.q_proj
    assert q_proj.weight.device.type == device.type
    assert isinstance(q_proj.weight, bnb.nn.Params4bit)
    assert q_proj.weight.quant_state is not None

    # 5. Run forward pass
    x = torch.randn(1, 4, hidden_dim, dtype=torch.float16, device=device)
    out = extracted_layers[0](x)[0]

    assert out.shape == (1, 4, hidden_dim)
    assert out.device.type == device.type
    assert not torch.isnan(out).any()


def test_load_layer_slice_local_tinyllama():
    """Verify load_layer_slice end-to-end with 4-bit on local TinyLlama model."""
    import os
    model_path = "./models/TinyLlama-1.1B-Chat-v1.0"
    if not os.path.exists(model_path):
        pytest.skip("Local TinyLlama model not found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    slice_4bit = load_layer_slice(
        model_path=model_path,
        layer_start=0,
        layer_end=2,
        include_norm=False,
        include_lm_head=False,
        device=device,
        load_in_4bit=True,
    )

    assert len(slice_4bit.layers) == 2
    assert slice_4bit.layer_start == 0
    assert slice_4bit.layer_end == 2
    assert slice_4bit.embed_tokens is not None

    # Verify first projection is 4-bit
    first_layer = slice_4bit.layers[0]
    assert isinstance(first_layer.self_attn.q_proj, bnb.nn.Linear4bit)
    assert isinstance(first_layer.self_attn.q_proj.weight, bnb.nn.Params4bit)
