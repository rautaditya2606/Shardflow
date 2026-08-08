"""
Standardized mock model layers and tokenizer fixtures for testing ShardFlow.
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig
from shardflow.node.layer_loader import ModelSlice


class DummyTokenizer:
    """Mock tokenizer providing encode/decode for testing."""
    def __init__(self, vocab_size: int = 100, eos_token_id: int = 99):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    def encode(self, text: str, **kwargs) -> list[int]:
        return [1, 2, 3]

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return " hello"

    def __call__(self, text: str, **kwargs) -> dict:
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}


class DummyEmbedding(nn.Module):
    """Mock token embedding layer."""
    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(x, self.weight)


class DummyLayer(nn.Module):
    """Mock transformer layer with parameter tracking."""
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.linear(x)


class DummyHead(nn.Module):
    """Mock LM head projection."""
    def __init__(self, hidden_dim: int = 32, vocab_size: int = 100):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def create_dummy_model_slice(
    layer_start: int = 0,
    layer_end: int = 1,
    hidden_dim: int = 32,
    vocab_size: int = 100,
    is_first: bool = False,
    is_last: bool = False,
    device: str = "cpu",
) -> ModelSlice:
    """Create a standardized ModelSlice instance for unit and integration testing."""
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_dim,
        intermediate_size=hidden_dim * 4,
        num_hidden_layers=layer_end - layer_start,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    layers = nn.ModuleList([DummyLayer(hidden_dim) for _ in range(layer_end - layer_start)]).to(device)
    embed = DummyEmbedding(vocab_size, hidden_dim).to(device) if is_first else None
    head = DummyHead(hidden_dim, vocab_size).to(device) if is_last else None

    return ModelSlice(
        layers=layers,
        config=config,
        layer_start=layer_start,
        layer_end=layer_end,
        embed_tokens=embed,
        lm_head=head,
        device=torch.device(device),
    )
