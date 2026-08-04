"""
Phase 0 — int4 and 8-bit partial model loader.

Provides utilities for:
1. Loading 4-bit (bitsandbytes NF4/FP4) or 8-bit layer slices
2. Direct safetensors shard weight extraction without full model instantiation
"""

import gc
import logging
from typing import Optional, List, Dict
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def load_int4_layer_slice(
    model_path: str,
    layer_start: int,
    layer_end: int,
    include_norm: bool = False,
    include_lm_head: bool = False,
    include_embed: bool = False,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Load a contiguous slice of quantized transformer layers (4-bit or 8-bit).

    Uses `bitsandbytes` quantization via `transformers.BitsAndBytesConfig`.
    Extracts the specified layer range `[layer_start, layer_end)` onto the target device.
    """
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError("bitsandbytes package required for int4 loading. Install with `pip install bitsandbytes`.")

    config = AutoConfig.from_pretrained(model_path)
    total_layers = config.num_hidden_layers

    logger.info("Loading model %s in %s for layers [%d, %d)...",
                model_path, "4-bit" if load_in_4bit else "8-bit", layer_start, layer_end)

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        load_in_8bit=load_in_8bit if not load_in_4bit else False,
        device_map=device if device != "cpu" else "cpu",
        low_cpu_mem_usage=True,
    )

    # Extract required layers
    extracted_layers = nn.ModuleList([
        model.model.layers[i] for i in range(layer_start, layer_end)
    ])

    norm = model.model.norm if include_norm else None
    lm_head = model.lm_head if include_lm_head else None
    embed_tokens = model.model.embed_tokens if include_embed else None
    rotary_emb = getattr(model.model, "rotary_emb", None)

    logger.info("Successfully extracted %d quantized layers", len(extracted_layers))

    return {
        "layers": extracted_layers,
        "config": config,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "norm": norm,
        "lm_head": lm_head,
        "embed_tokens": embed_tokens,
        "rotary_emb": rotary_emb,
    }
