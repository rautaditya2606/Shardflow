"""
Partial model loading — load only a contiguous slice of transformer layers.

Given a model path and a layer range [start, end), loads:
- Only the specified transformer layers
- Optionally the embedding layer (for orchestrator / Node 0)
- Optionally the LM head + final norm (for final node)

Supports:
1. Targeted safetensors shard loading without instantiating full model
2. In-place 4-bit (bitsandbytes NF4) meta-device quantization for 70B/72B scale models
"""

import os

# Disable Hugging Face Xet transfer backend which causes memory leaks and OOM in notebook environments (Kaggle/Colab)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# Force direct assignment to 20GB /kaggle/working storage if running in Kaggle to avoid root / exhaustion
if os.path.exists("/kaggle"):
    os.environ["HF_HOME"] = "/kaggle/working/hf_home"
    os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/hf_home"
    os.environ["HF_HUB_CACHE"] = "/kaggle/working/hf_home"

import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def _replace_linear_with_4bit_meta(
    module: nn.Module,
    compute_dtype: torch.dtype = torch.float16,
    quant_type: str = "nf4",
    use_double_quant: bool = True,
) -> nn.Module:
    """
    Recursively replaces all nn.Linear modules with bnb.nn.Linear4bit on the 'meta' device.
    Zero real memory is allocated during this conversion.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError("bitsandbytes required for 4-bit loading. Run `pip install bitsandbytes`.")

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            has_bias = child.bias is not None

            qlinear = bnb.nn.Linear4bit(
                input_features=child.in_features,
                output_features=child.out_features,
                bias=has_bias,
                compute_dtype=compute_dtype,
                compress_statistics=use_double_quant,
                quant_type=quant_type,
                device="meta",
            )
            qlinear.weight = bnb.nn.Params4bit(
                data=torch.empty(child.out_features, child.in_features, device="meta"),
                requires_grad=False,
                quant_type=quant_type,
            )
            if has_bias:
                qlinear.bias = nn.Parameter(
                    torch.empty(child.out_features, device="meta"),
                    requires_grad=False,
                )
            setattr(module, name, qlinear)
        else:
            _replace_linear_with_4bit_meta(
                child,
                compute_dtype=compute_dtype,
                quant_type=quant_type,
                use_double_quant=use_double_quant,
            )
    return module


def _load_state_dict_into_4bit_slice(
    extracted_layers: nn.ModuleList,
    state_dict: Dict[str, torch.Tensor],
    layer_start: int,
    layer_end: int,
    device: torch.device,
    compute_dtype: torch.dtype = torch.float16,
    quant_type: str = "nf4",
    use_double_quant: bool = True,
    norm: Optional[nn.Module] = None,
    lm_head: Optional[nn.Module] = None,
    embed_tokens: Optional[nn.Module] = None,
) -> None:
    """
    Quantizes and loads FP16/BF16 state_dict tensors directly into 4-bit module slice in-place.
    Peak memory per tensor is strictly bounded to a single weight matrix.
    """
    try:
        import bitsandbytes as bnb
        import bitsandbytes.functional as bnb_F
    except ImportError:
        raise ImportError("bitsandbytes required for 4-bit loading.")

    for key, tensor in list(state_dict.items()):
        # Transformer layers
        if key.startswith("model.layers."):
            parts = key.split(".")
            orig_idx = int(parts[2])
            if orig_idx < layer_start or orig_idx >= layer_end:
                continue

            local_idx = orig_idx - layer_start
            layer = extracted_layers[local_idx]
            subpath = ".".join(parts[3:])

            submod = layer
            path_parts = subpath.split(".")
            for part in path_parts[:-1]:
                submod = getattr(submod, part)
            param_name = path_parts[-1]

            if isinstance(submod, bnb.nn.Linear4bit) and param_name == "weight":
                raw_weight = tensor.to(device=device, dtype=compute_dtype, non_blocking=True)
                q_weight, q_state = bnb_F.quantize_4bit(
                    raw_weight,
                    quant_type=quant_type,
                    blocksize=64,
                    compress_statistics=use_double_quant,
                )
                param = bnb.nn.Params4bit(
                    data=q_weight,
                    requires_grad=False,
                    quant_type=quant_type,
                )
                param.quant_state = q_state
                setattr(submod, "weight", param)
                del raw_weight

            elif isinstance(submod, bnb.nn.Linear4bit) and param_name == "bias":
                bias_param = nn.Parameter(
                    tensor.to(device=device, dtype=compute_dtype),
                    requires_grad=False,
                )
                setattr(submod, "bias", bias_param)

            else:
                norm_param = nn.Parameter(
                    tensor.to(device=device, dtype=compute_dtype),
                    requires_grad=False,
                )
                setattr(submod, param_name, norm_param)

        # Final RMSNorm
        elif key.startswith("model.norm.") and norm is not None:
            norm.weight = nn.Parameter(
                tensor.to(device=device, dtype=compute_dtype),
                requires_grad=False,
            )

        # LM Head
        elif key.startswith("lm_head.") and lm_head is not None:
            if isinstance(lm_head, bnb.nn.Linear4bit):
                raw_weight = tensor.to(device=device, dtype=compute_dtype, non_blocking=True)
                q_weight, q_state = bnb_F.quantize_4bit(
                    raw_weight,
                    quant_type=quant_type,
                    blocksize=64,
                    compress_statistics=use_double_quant,
                )
                param = bnb.nn.Params4bit(
                    data=q_weight,
                    requires_grad=False,
                    quant_type=quant_type,
                )
                param.quant_state = q_state
                lm_head.weight = param
                del raw_weight
            else:
                lm_head.weight = nn.Parameter(
                    tensor.to(device=device, dtype=compute_dtype),
                    requires_grad=False,
                )

        # Token Embedding
        elif key.startswith("model.embed_tokens.") and embed_tokens is not None:
            embed_tokens.weight = nn.Parameter(
                tensor.to(device=device, dtype=compute_dtype),
                requires_grad=False,
            )

    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _get_safetensors_shards_map(model_path: str) -> tuple[Optional[Dict[str, str]], Optional[Path]]:
    """
    Resolve safetensors weight_map and base directory if available.
    Returns (weight_map, local_dir_path).
    """
    local_dir = _resolve_local_dir(model_path)
    if local_dir is not None:
        index_path = local_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                return json.load(f)["weight_map"], local_dir
        single_path = local_dir / "model.safetensors"
        if single_path.exists():
            return None, local_dir

    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        target_cache_dir = "/kaggle/working/hf_home" if os.path.exists("/kaggle") else None
        hf_tok = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        index_file = try_to_load_from_cache(repo_id=model_path, filename="model.safetensors.index.json", cache_dir=target_cache_dir, token=hf_tok)
        if not (isinstance(index_file, (str, Path)) and os.path.exists(index_file)):
            index_file = hf_hub_download(repo_id=model_path, filename="model.safetensors.index.json", cache_dir=target_cache_dir, token=hf_tok)
        with open(index_file) as f:
            return json.load(f)["weight_map"], local_dir
    except Exception:
        return None, local_dir


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

    Args:
        model_path: local path or HF model ID
        layer_start: first layer index (inclusive)
        layer_end: last layer index (exclusive)
        include_norm: if True, include the final RMSNorm (for final node)
        include_lm_head: if True, include the LM head (for final node)
        include_embed: if True, include the token embedding (for orchestrator/Node 0)
        dtype: cast weights to this dtype (default: model's native dtype)
        device: target device for the slice
        load_in_4bit: if True, quantize linear layers in-place to NF4 via bitsandbytes

    Returns:
        ModelSlice with the requested components
    """
    logger.info(
        "Loading layers [%d, %d) from %s (norm=%s, lm_head=%s, embed=%s, 4bit=%s)",
        layer_start, layer_end, model_path,
        include_norm, include_lm_head, include_embed, load_in_4bit,
    )

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

    target_device = torch.device(device)

    try:
        from accelerate import init_empty_weights
        has_accelerate = True
    except ImportError:
        has_accelerate = False

    if not has_accelerate:
        raise RuntimeError("The 'accelerate' package is required for zero-RAM layer slicing. Run `pip install accelerate`.")

    # 1. Instantiate meta shell (0 MB RAM footprint)
    logger.info("Initializing meta device model shell (0 MB RAM footprint)...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=target_dtype)

    # 2. Extract requested layers on meta device
    extracted_layers = nn.ModuleList([
        model.model.layers[i] for i in range(layer_start, layer_end)
    ])
    norm = model.model.norm if include_norm else None
    lm_head = model.lm_head if include_lm_head else None
    embed_tokens = model.model.embed_tokens if include_embed else None

    rotary_emb = None
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        try:
            rotary_cls = type(model.model.rotary_emb)
            rotary_emb = rotary_cls(config).to(target_device)
        except Exception:
            rotary_emb = None
    elif hasattr(model, "rotary_emb") and model.rotary_emb is not None:
        try:
            rotary_cls = type(model.rotary_emb)
            rotary_emb = rotary_cls(config).to(target_device)
        except Exception:
            rotary_emb = None

    # 3. Handle 4-Bit In-Place Quantization vs FP16/BF16 Allocation
    if load_in_4bit:
        logger.info("Converting Linear layers to bnb.nn.Linear4bit on meta device (0 RAM)...")
        for layer in extracted_layers:
            _replace_linear_with_4bit_meta(layer, compute_dtype=target_dtype)
        if lm_head is not None:
            _replace_linear_with_4bit_meta(lm_head, compute_dtype=target_dtype)

        # Allocate non-quantized 1D modules on target device
        if norm is not None:
            norm = norm.to_empty(device=target_device)
        if embed_tokens is not None:
            embed_tokens = embed_tokens.to_empty(device=target_device)

    else:
        # FP16 / BF16 unquantized path
        extracted_layers = extracted_layers.to_empty(device=target_device)
        if norm is not None:
            norm = norm.to_empty(device=target_device)
        if lm_head is not None:
            lm_head = lm_head.to_empty(device=target_device)
        if embed_tokens is not None:
            embed_tokens = embed_tokens.to_empty(device=target_device)

    # 4. Resolve targeted safetensors shards
    from safetensors.torch import load_file

    weight_map, local_dir = _get_safetensors_shards_map(model_path)
    target_prefixes = [f"model.layers.{i}." for i in range(layer_start, layer_end)]
    if include_norm:
        target_prefixes.append("model.norm.")
    if include_lm_head:
        target_prefixes.append("lm_head.")
    if include_embed:
        target_prefixes.append("model.embed_tokens.")

    if weight_map is not None:
        needed_shards = sorted({
            shard for key, shard in weight_map.items()
            if any(key.startswith(p) for p in target_prefixes)
        })
        logger.info("Matched %d targeted safetensors shards: %s", len(needed_shards), needed_shards)

        for shard_name in needed_shards:
            if local_dir is not None and (local_dir / shard_name).exists():
                shard_path = str(local_dir / shard_name)
            else:
                from huggingface_hub import hf_hub_download, try_to_load_from_cache
                target_cache_dir = "/kaggle/working/hf_home" if os.path.exists("/kaggle") else None
                hf_tok = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
                cached = try_to_load_from_cache(repo_id=model_path, filename=shard_name, cache_dir=target_cache_dir, token=hf_tok)
                if isinstance(cached, (str, Path)) and os.path.exists(cached):
                    shard_path = str(cached)
                else:
                    logger.info("Downloading targeted shard %s from HuggingFace Hub (cache_dir=%s)...", shard_name, target_cache_dir)
                    shard_path = hf_hub_download(repo_id=model_path, filename=shard_name, cache_dir=target_cache_dir, token=hf_tok)

            logger.info("Streaming and loading weights from shard %s ...", shard_name)
            from safetensors import safe_open
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if not any(key.startswith(p) for p in target_prefixes):
                        continue
                    tensor = f.get_tensor(key)

                    if load_in_4bit:
                        _load_state_dict_into_4bit_slice(
                            extracted_layers=extracted_layers,
                            state_dict={key: tensor},
                            layer_start=layer_start,
                            layer_end=layer_end,
                            device=target_device,
                            compute_dtype=target_dtype,
                            norm=norm,
                            lm_head=lm_head,
                            embed_tokens=embed_tokens,
                        )
                    else:
                        # Direct in-place FP16/BF16 loading to avoid doubling VRAM allocation
                        if key.startswith("model.layers."):
                            parts = key.split(".")
                            orig_idx = int(parts[2])
                            if layer_start <= orig_idx < layer_end:
                                local_idx = orig_idx - layer_start
                                subpath = ".".join(parts[3:])
                                submod = extracted_layers[local_idx]
                                path_parts = subpath.split(".")
                                for part in path_parts[:-1]:
                                    submod = getattr(submod, part)
                                param_name = path_parts[-1]
                                current_param = getattr(submod, param_name, None)
                                if current_param is not None and current_param.device == target_device:
                                    current_param.data.copy_(tensor)
                                else:
                                    param = nn.Parameter(
                                        tensor.to(device=target_device, dtype=target_dtype, non_blocking=True),
                                        requires_grad=False,
                                    )
                                    setattr(submod, param_name, param)

                        elif key.startswith("model.norm.") and norm is not None:
                            if hasattr(norm, "weight") and norm.weight is not None and norm.weight.device == target_device:
                                norm.weight.data.copy_(tensor)
                            else:
                                norm.weight = nn.Parameter(
                                    tensor.to(device=target_device, dtype=target_dtype, non_blocking=True),
                                    requires_grad=False,
                                )
                        elif key.startswith("lm_head.") and lm_head is not None:
                            if hasattr(lm_head, "weight") and lm_head.weight is not None and lm_head.weight.device == target_device:
                                lm_head.weight.data.copy_(tensor)
                            else:
                                lm_head.weight = nn.Parameter(
                                    tensor.to(device=target_device, dtype=target_dtype, non_blocking=True),
                                    requires_grad=False,
                                )
                        elif key.startswith("model.embed_tokens.") and embed_tokens is not None:
                            if hasattr(embed_tokens, "weight") and embed_tokens.weight is not None and embed_tokens.weight.device == target_device:
                                embed_tokens.weight.data.copy_(tensor)
                            else:
                                embed_tokens.weight = nn.Parameter(
                                    tensor.to(device=target_device, dtype=target_dtype, non_blocking=True),
                                    requires_grad=False,
                                )

                    del tensor

            logger.info("Shard %s loaded successfully into GPU device %s", shard_name, target_device)
            gc.collect()
            if target_device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    else:
        # Single safetensors file or local directory
        single_path = None
        if local_dir is not None and (local_dir / "model.safetensors").exists():
            single_path = local_dir / "model.safetensors"
        else:
            try:
                from huggingface_hub import hf_hub_download, try_to_load_from_cache
                cached = try_to_load_from_cache(repo_id=model_path, filename="model.safetensors")
                if isinstance(cached, (str, Path)) and os.path.exists(cached):
                    single_path = str(cached)
                else:
                    single_path = hf_hub_download(repo_id=model_path, filename="model.safetensors")
            except Exception:
                single_path = None

        if single_path is not None:
            logger.info("Loading single safetensors file %s...", single_path)
            state_dict = load_file(single_path, device="cpu")
            if load_in_4bit:
                _load_state_dict_into_4bit_slice(
                    extracted_layers=extracted_layers,
                    state_dict=state_dict,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    device=target_device,
                    compute_dtype=target_dtype,
                    norm=norm,
                    lm_head=lm_head,
                    embed_tokens=embed_tokens,
                )
            else:
                for idx, orig_idx in enumerate(range(layer_start, layer_end)):
                    prefix = f"model.layers.{orig_idx}."
                    layer_sd = {
                        k[len(prefix):]: v.to(target_device, dtype=target_dtype)
                        for k, v in state_dict.items() if k.startswith(prefix)
                    }
                    extracted_layers[idx].load_state_dict(layer_sd, strict=False)
                if norm is not None:
                    norm.load_state_dict({k[len("model.norm."):]: v.to(target_device, dtype=target_dtype) for k, v in state_dict.items() if k.startswith("model.norm.")}, strict=False)
                if lm_head is not None:
                    lm_head.load_state_dict({k[len("lm_head."):]: v.to(target_device, dtype=target_dtype) for k, v in state_dict.items() if k.startswith("lm_head.")}, strict=False)
                if embed_tokens is not None:
                    embed_tokens.load_state_dict({k[len("model.embed_tokens."):]: v.to(target_device, dtype=target_dtype) for k, v in state_dict.items() if k.startswith("model.embed_tokens.")}, strict=False)
            del state_dict

    del model
    gc.collect()
    if target_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(target_device) / 1e9
        logger.info("GPU VRAM after slice load: %.2f GB", allocated)

    return ModelSlice(
        layers=extracted_layers,
        config=config,
        layer_start=layer_start,
        layer_end=layer_end,
        norm=norm,
        lm_head=lm_head,
        embed_tokens=embed_tokens,
        rotary_emb=rotary_emb,
        device=target_device,
    )


def quantize_module_4bit(module: nn.Module, device: torch.device) -> nn.Module:
    """Helper to replace linear layers in module with bitsandbytes 4-bit NF4 linear layers."""
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
