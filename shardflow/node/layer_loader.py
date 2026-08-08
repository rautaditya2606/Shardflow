"""
Partial model loading — load only a contiguous slice of transformer layers.

Given a model path and a layer range [start, end), loads:
- Only the specified transformer layers
- Optionally the embedding layer (for orchestrator)
- Optionally the LM head + final norm (for final node)

Uses safetensors for direct weight loading without instantiating the full model.
"""

import gc
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _resolve_local_dir(model_path: str) -> Optional[Path]:
    """Return a local directory path for model_path, or None if unavailable."""
    if os.path.isdir(model_path):
        return Path(model_path)
    try:
        from huggingface_hub import snapshot_download
        local = snapshot_download(model_path, local_files_only=True)
        return Path(local)
    except Exception:
        return None


def _load_weights_from_safetensors(
    model_path: str,
    target_keys: Set[str],
    target_dtype: torch.dtype,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load only the weights in target_keys from safetensors files.

    Downloads ONLY the specific safetensors shard files containing target_keys,
    eliminating full-model downloads and 14 GB RAM spikes.

    Returns a state_dict (key → tensor) or None if safetensors unavailable.
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        return None

    # Check local directory first
    if os.path.isdir(model_path):
        local_dir = Path(model_path)
        index_path = local_dir / "model.safetensors.index.json"
        single_path = local_dir / "model.safetensors"

        if index_path.exists():
            with open(index_path) as f:
                weight_map: Dict[str, str] = json.load(f)["weight_map"]
            needed_shards = {weight_map[k] for k in target_keys if k in weight_map}
            if not needed_shards:
                logger.warning("No safetensors shards matched target keys in local index")
                return None
            state_dict: Dict[str, torch.Tensor] = {}
            for shard in sorted(needed_shards):
                logger.info("Loading local shard %s ...", shard)
                data = load_file(local_dir / shard, device="cpu")
                state_dict.update({k: v.to(target_dtype) for k, v in data.items() if k in target_keys})
            return state_dict

        if single_path.exists():
            logger.info("Loading single local safetensors file...")
            data = load_file(single_path, device="cpu")
            return {k: v.to(target_dtype) for k, v in data.items() if k in target_keys}

    # Remote HuggingFace Hub loading — download ONLY required index & shards
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    # 1. Try downloading index file
    try:
        logger.info("Fetching model.safetensors.index.json for targeted shard mapping (%s)...", model_path)
        index_file = hf_hub_download(repo_id=model_path, filename="model.safetensors.index.json")
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
        needed_shards = {weight_map[k] for k in target_keys if k in weight_map}
        if not needed_shards:
            logger.warning("No safetensors shards matched target keys in remote index")
            return None

        state_dict = {}
        for shard_filename in sorted(needed_shards):
            logger.info("Downloading targeted shard %s from HuggingFace Hub...", shard_filename)
            shard_path = hf_hub_download(repo_id=model_path, filename=shard_filename)
            logger.info("Loading shard %s ...", shard_filename)
            data = load_file(shard_path, device="cpu")
            state_dict.update({k: v.to(target_dtype) for k, v in data.items() if k in target_keys})
        return state_dict
    except Exception as e:
        logger.debug("Failed to load via remote index file: %s", e)

    # 2. Try single safetensors file
    try:
        logger.info("Attempting single model.safetensors download from HuggingFace Hub (%s)...", model_path)
        single_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")
        data = load_file(single_path, device="cpu")
        return {k: v.to(target_dtype) for k, v in data.items() if k in target_keys}
    except Exception as e:
        logger.warning("Failed single safetensors download: %s", e)

    return None



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
            try:
                self.rotary_emb = self.rotary_emb.to(device)
            except Exception as e:
                logger.debug("Could not move rotary_emb to %s: %s", device, e)
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
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
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

    # Node 0 (layer_start == 0) always owns token embedding matrix
    if layer_start == 0:
        include_embed = True

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
            model = AutoModelForCausalLM.from_config(config, dtype=target_dtype)
        
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
        if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
            try:
                rotary_cls = type(model.model.rotary_emb)
                rotary_emb = rotary_cls(config).to(device)
            except Exception:
                try:
                    rotary_emb = model.model.rotary_emb.to_empty(device=device)
                except Exception:
                    rotary_emb = None
        elif hasattr(model, "rotary_emb") and model.rotary_emb is not None:
            try:
                rotary_cls = type(model.rotary_emb)
                rotary_emb = rotary_cls(config).to(device)
            except Exception:
                rotary_emb = None

        # Collect the exact safetensors keys we need from the allocated modules.
        # This is model-agnostic: we ask the module itself what its parameters are.
        target_keys: Set[str] = set()
        for idx, orig_idx in enumerate(range(layer_start, layer_end)):
            prefix = f"model.layers.{orig_idx}"
            for param_name in dict(extracted_layers[idx].named_parameters()).keys():
                target_keys.add(f"{prefix}.{param_name}")
        if norm is not None:
            for param_name in dict(norm.named_parameters()).keys():
                target_keys.add(f"model.norm.{param_name}")
        if lm_head is not None:
            for param_name in dict(lm_head.named_parameters()).keys():
                target_keys.add(f"lm_head.{param_name}")
        if embed_tokens is not None:
            for param_name in dict(embed_tokens.named_parameters()).keys():
                target_keys.add(f"model.embed_tokens.{param_name}")

        logger.info(
            "Need %d weight tensors for layers [%d, %d)",
            len(target_keys), layer_start, layer_end,
        )

        # Try targeted safetensors load first (only loads slice shards)
        state_dict = _load_weights_from_safetensors(model_path, target_keys, target_dtype)

        if state_dict is not None:
            # Apply directly into the already-allocated tensors
            for idx, orig_idx in enumerate(range(layer_start, layer_end)):
                prefix = f"model.layers.{orig_idx}"
                layer_sd = {
                    k[len(prefix) + 1:]: v
                    for k, v in state_dict.items()
                    if k.startswith(prefix + ".")
                }
                extracted_layers[idx].load_state_dict(layer_sd)
            if norm is not None:
                norm_sd = {k[len("model.norm."):]: v for k, v in state_dict.items() if k.startswith("model.norm.")}
                norm.load_state_dict(norm_sd)
            if lm_head is not None:
                head_sd = {k[len("lm_head."):]: v for k, v in state_dict.items() if k.startswith("lm_head.")}
                lm_head.load_state_dict(head_sd)
            if embed_tokens is not None:
                embed_sd = {k[len("model.embed_tokens."):]: v for k, v in state_dict.items() if k.startswith("model.embed_tokens.")}
                embed_tokens.load_state_dict(embed_sd)
            del state_dict
        else:
            # Fallback: load full model to CPU and copy slice weights
            logger.warning(
                "Targeted safetensors load unavailable — loading full model to CPU (higher RAM usage)"
            )
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

    if load_in_4bit:
        logger.info("Quantizing layer slice to 4-bit NF4 via bitsandbytes...")
        quantize_module_4bit(model_slice.layers, target_device)
        if model_slice.lm_head is not None:
            quantize_module_4bit(model_slice.lm_head, target_device)

    if target_device.type != "cpu":
        logger.info("Moving slice to %s...", target_device)
        model_slice.to(target_device)

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(target_device) / 1e9
            logger.info("GPU memory after loading: %.2f GB", allocated)

    return model_slice


def quantize_module_4bit(module: nn.Module, device: torch.device) -> nn.Module:
    """Replace linear layers in module with bitsandbytes 4-bit NF4 linear layers."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        logger.warning("bitsandbytes not installed — skipping 4-bit quantization")
        return module

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            has_bias = child.bias is not None
            compute_dtype = child.weight.dtype if child.weight.dtype in (torch.float16, torch.bfloat16) else torch.float16
            qlinear = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=has_bias,
                compute_dtype=compute_dtype,
                quant_type="nf4",
                device=device,
            )
            qlinear.weight = bnb.nn.Params4bit(
                child.weight.data,
                requires_grad=False,
                quant_type="nf4",
            ).to(device)
            if has_bias and child.bias is not None:
                qlinear.bias = child.bias.to(device)
            setattr(module, name, qlinear)
        else:
            quantize_module_4bit(child, device)
    return module


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
