"""
Partial model loading — load only a contiguous slice of transformer layers.

Given a model path and a layer range [start, end), loads:
- Only the specified transformer layers
- Optionally the embedding layer (for orchestrator)
- Optionally the LM head + final norm (for final node)

Uses safetensors for direct weight loading without instantiating the full model.
"""

import gc
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ModelSlice:
    """A slice of transformer layers with optional head/embedding."""
    layers: nn.ModuleList
    config: AutoConfig
    layer_start: int
    layer_end: int  # exclusive
    norm: Optional[nn.Module] = None       # final RMSNorm (for last node)
    lm_head: Optional[nn.Module] = None    # LM head (for last node)
    embed_tokens: Optional[nn.Module] = None  # token embedding (for orchestrator)
    rotary_emb: Optional[nn.Module] = None  # RoPE embedding (every node needs this)
    device: torch.device = torch.device("cpu")

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start

    def to(self, device: torch.device) -> "ModelSlice":
        """Move all components to a device."""
        self.device = device
        self.layers = self.layers.to(device)
        if self.norm is not None:
            self.norm = self.norm.to(device)
        if self.lm_head is not None:
            self.lm_head = self.lm_head.to(device)
        if self.embed_tokens is not None:
            self.embed_tokens = self.embed_tokens.to(device)
        if self.rotary_emb is not None:
            self.rotary_emb = self.rotary_emb.to(device)
        return self


def load_layer_slice(
    model_path: str,
    layer_start: int,
    layer_end: int,
    include_norm: bool = False,
    include_lm_head: bool = False,
    include_embed: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: str = "cpu",
) -> ModelSlice:
    """
    Load a contiguous slice of transformer layers from a model.

    Strategy:
    1. Load the full model to CPU with low memory footprint
    2. Extract the needed layers
    3. Delete the rest and free memory
    4. Move the slice to the target device

    Args:
        model_path: local path or HF model ID
        layer_start: first layer index (inclusive)
        layer_end: last layer index (exclusive)
        include_norm: if True, include the final RMSNorm (for final node)
        include_lm_head: if True, include the LM head (for final node)
        include_embed: if True, include the token embedding (for orchestrator)
        dtype: cast weights to this dtype (default: model's native dtype)
        device: target device for the slice

    Returns:
        ModelSlice with the requested components
    """
    logger.info(
        "Loading layers [%d, %d) from %s (norm=%s, lm_head=%s, embed=%s)",
        layer_start, layer_end, model_path,
        include_norm, include_lm_head, include_embed,
    )

    # Load config to get layer count
    config = AutoConfig.from_pretrained(model_path)
    total_layers = config.num_hidden_layers

    if layer_start < 0 or layer_end > total_layers or layer_start >= layer_end:
        raise ValueError(
            f"Invalid layer range [{layer_start}, {layer_end}) "
            f"for model with {total_layers} layers"
        )

    target_dtype = dtype or getattr(config, "torch_dtype", torch.float16)
    if isinstance(target_dtype, str):
        target_dtype = getattr(torch, target_dtype)

    # Fast path: Zero-RAM meta device shell instantiation
    try:
        from accelerate import init_empty_weights
        has_accelerate = True
    except ImportError:
        has_accelerate = False

    if has_accelerate:
        logger.info("Initializing meta device model shell (0 MB RAM footprint)...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=target_dtype)
        
        # Extract requested layers and allocate memory ONLY for the slice
        extracted_layers = nn.ModuleList([
            model.model.layers[i] for i in range(layer_start, layer_end)
        ]).to_empty(device=device)

        norm = None
        if include_norm:
            norm = model.model.norm.to_empty(device=device)
            logger.info("Extracted final norm layer")

        lm_head = None
        if include_lm_head:
            lm_head = model.lm_head.to_empty(device=device)
            logger.info("Extracted LM head")

        embed_tokens = None
        if include_embed:
            embed_tokens = model.model.embed_tokens.to_empty(device=device)
            logger.info("Extracted embedding layer")

        rotary_emb = None
        if hasattr(model.model, "rotary_emb"):
            rotary_emb = model.model.rotary_emb

        # Now load actual weights into the allocated slice
        logger.info("Loading weights directly into slice layers [%d, %d)...", layer_start, layer_end)
        full_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=target_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        for idx, orig_idx in enumerate(range(layer_start, layer_end)):
            extracted_layers[idx].load_state_dict(full_model.model.layers[orig_idx].state_dict())
        if norm is not None:
            norm.load_state_dict(full_model.model.norm.state_dict())
        if lm_head is not None:
            lm_head.load_state_dict(full_model.lm_head.state_dict())
        if embed_tokens is not None:
            embed_tokens.load_state_dict(full_model.model.embed_tokens.state_dict())

        del full_model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info("Loading full model to CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=target_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        extracted_layers = nn.ModuleList([
            model.model.layers[i] for i in range(layer_start, layer_end)
        ])
        norm = model.model.norm if include_norm else None
        lm_head = model.lm_head if include_lm_head else None
        embed_tokens = model.model.embed_tokens if include_embed else None
        rotary_emb = getattr(model.model, "rotary_emb", None)
        del model
        gc.collect()

    # Build the slice
    model_slice = ModelSlice(
        layers=extracted_layers,
        config=config,
        layer_start=layer_start,
        layer_end=layer_end,
        norm=norm,
        lm_head=lm_head,
        embed_tokens=embed_tokens,
        rotary_emb=rotary_emb,
    )

    # Move to target device
    target_device = torch.device(device)
    if target_device.type != "cpu":
        logger.info("Moving slice to %s...", target_device)
        model_slice.to(target_device)

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(target_device) / 1e9
            logger.info("GPU memory after loading: %.2f GB", allocated)

    return model_slice


def load_tokenizer(model_path: str) -> AutoTokenizer:
    """Load the tokenizer for a model."""
    return AutoTokenizer.from_pretrained(model_path)


def get_model_info(model_path: str) -> dict:
    """Get model metadata without loading weights."""
    config = AutoConfig.from_pretrained(model_path)
    return {
        "model_type": config.model_type,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "torch_dtype": str(config.torch_dtype),
    }
