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


def get_num_hidden_layers(config: Any, default: int = 64) -> int:
    """Safely extract the number of transformer layers across all model families and architectures."""
    if config is None:
        return default
    try:
        if hasattr(config, "text_config") and config.text_config is not None:
            tc = config.text_config
            if hasattr(tc, "num_hidden_layers") and tc.num_hidden_layers is not None:
                return int(tc.num_hidden_layers)
    except Exception:
        pass

    try:
        val = getattr(config, "num_hidden_layers", None)
        if val is not None:
            return int(val)
    except Exception:
        pass

    try:
        val = getattr(config, "num_layers", None)
        if val is not None:
            return int(val)
    except Exception:
        pass

    return default


def _download_shard_direct(model_path: str, shard_name: str, dest_path: str) -> str:
    """
    Downloads a single safetensors shard directly to dest_path without HF Hub blob duplication.
    Guarantees zero hidden cache build-up.
    """
    import urllib.request
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo_id=model_path, filename=shard_name)
    hf_tok = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    headers = {"User-Agent": "ShardFlow/2.1"}
    if hf_tok:
        headers["Authorization"] = f"Bearer {hf_tok}"

    req = urllib.request.Request(url, headers=headers)
    logger.info("Downloading shard %s directly (clean single-file stream)...", shard_name)

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 16 * 1024 * 1024  # 16 MB chunks for max throughput
        last_log = time.perf_counter()

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if time.perf_counter() - last_log > 3.0:
                mb = downloaded / (1024 * 1024)
                tot_mb = total_size / (1024 * 1024)
                pct = (downloaded / total_size * 100.0) if total_size > 0 else 0.0
                speed = mb / max(0.1, time.perf_counter() - t0)
                logger.info("  [%s] %.1f / %.1f MB (%.1f%%) - %.1f MB/s", shard_name, mb, tot_mb, pct, speed)
                last_log = time.perf_counter()

    logger.info("[OK] Shard %s downloaded (%.2f GB) in %.1f s", shard_name, downloaded / 1024**3, time.perf_counter() - t0)
    return dest_path


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
    Loads state_dict tensors directly into 4-bit module slice in-place.
    Handles both pre-quantized bitsandbytes NF4 checkpoints and on-the-fly quantization of FP16 weights.
    """
    try:
        import bitsandbytes as bnb
        import bitsandbytes.functional as bnb_F
    except ImportError:
        raise ImportError("bitsandbytes required for 4-bit loading.")

    # Group tensors per target module
    grouped_modules: Dict[Any, Dict[str, torch.Tensor]] = {}

    for key, tensor in list(state_dict.items()):
        norm_key = key
        for prefix in ["model.language_model.", "language_model.model.", "language_model."]:
            if norm_key.startswith(prefix):
                norm_key = "model." + norm_key[len(prefix):]
                break

        # Final RMSNorm
        if norm_key.startswith("model.norm.") and norm is not None:
            norm.weight = nn.Parameter(
                tensor.to(device=device, dtype=compute_dtype),
                requires_grad=False,
            )
            continue

        # Token Embedding
        if norm_key.startswith("model.embed_tokens.") and embed_tokens is not None:
            embed_tokens.weight = nn.Parameter(
                tensor.to(device=device, dtype=compute_dtype),
                requires_grad=False,
            )
            continue

        # LM Head
        if norm_key.startswith("lm_head.") and lm_head is not None:
            subpath = norm_key[len("lm_head."):]
            if lm_head not in grouped_modules:
                grouped_modules[lm_head] = {}
            grouped_modules[lm_head][subpath] = tensor
            continue

        # Transformer layers
        if norm_key.startswith("model.layers."):
            parts = norm_key.split(".")
            orig_idx = int(parts[2])
            if orig_idx < layer_start or orig_idx >= layer_end:
                continue

            local_idx = orig_idx - layer_start
            layer = extracted_layers[local_idx]

            subpath_parts = parts[3:]
            submod = layer
            param_parts = []

            for idx, part in enumerate(subpath_parts):
                if hasattr(submod, part) and isinstance(getattr(submod, part), nn.Module):
                    submod = getattr(submod, part)
                else:
                    param_parts = subpath_parts[idx:]
                    break

            param_key = ".".join(param_parts) if param_parts else "weight"
            if submod not in grouped_modules:
                grouped_modules[submod] = {}
            grouped_modules[submod][param_key] = tensor

    # Assign parameters to each submodule
    for submod, tensors in grouped_modules.items():
        if isinstance(submod, bnb.nn.Linear4bit):
            # Check if pre-quantized (uint8/int8) or raw FP16/BF16
            weight_tensor = tensors.get("weight")
            if weight_tensor is None:
                for k, v in tensors.items():
                    if k.startswith("weight") and v.dtype in (torch.uint8, torch.int8):
                        weight_tensor = v
                        break

            if weight_tensor is not None and weight_tensor.dtype in (torch.uint8, torch.int8):
                # Pre-quantized safetensors
                quant_stats = {}
                for k, v in tensors.items():
                    clean_k = k[7:] if k.startswith("weight.") else k
                    if clean_k not in ("weight", "bias"):
                        quant_stats[clean_k] = v.to(device=device) if hasattr(v, "to") else v
                try:
                    param = bnb.nn.Params4bit.from_prequantized(
                        data=weight_tensor,
                        quantized_stats=quant_stats,
                        requires_grad=False,
                        device=device,
                    )
                    submod.weight = param
                except Exception as e:
                    logger.debug("from_prequantized fallback for %s: %s", submod, e)
                    param = bnb.nn.Params4bit(weight_tensor.to(device=device), requires_grad=False, quant_type=quant_type)
                    submod.weight = param

            elif weight_tensor is not None:
                # Raw unquantized FP16/BF16 -> quantize in-place
                raw_weight = weight_tensor.to(device=device, dtype=compute_dtype, non_blocking=True)
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
                submod.weight = param
                del raw_weight

            # Bias if present
            if "bias" in tensors and tensors["bias"] is not None:
                submod.bias = nn.Parameter(
                    tensors["bias"].to(device=device, dtype=compute_dtype),
                    requires_grad=False,
                )

        else:
            # Regular LayerNorm, RMSNorm, Linear, or embedding parameter
            for p_name, p_tensor in tensors.items():
                clean_name = p_name.split(".")[-1]
                param = nn.Parameter(
                    p_tensor.to(device=device, dtype=compute_dtype),
                    requires_grad=False,
                )
                try:
                    setattr(submod, clean_name, param)
                except Exception as e:
                    logger.debug("Parameter set on %s.%s: %s", submod, clean_name, e)

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
        target_cache_dir = None
        if os.path.exists("/kaggle"):
            target_cache_dir = "/kaggle/working/hf_home"
        elif os.path.exists("/content"):
            target_cache_dir = "/content/hf_home"

        hf_tok = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        index_file = try_to_load_from_cache(repo_id=model_path, filename="model.safetensors.index.json", cache_dir=target_cache_dir)
        if not (isinstance(index_file, (str, Path)) and os.path.exists(index_file)):
            index_file = hf_hub_download(repo_id=model_path, filename="model.safetensors.index.json", cache_dir=target_cache_dir, token=hf_tok)
        with open(index_file) as f:
            return json.load(f)["weight_map"], local_dir
    except Exception as e:
        logger.warning("Could not resolve model.safetensors.index.json for %s: %s", model_path, e)
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

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    total_layers = get_num_hidden_layers(config, default=64)

    # Node 0 (layer_start == 0) always owns token embedding matrix
    if layer_start == 0:
        include_embed = True

    # Terminal node (layer_end == total_layers) owns final norm and LM head
    if layer_end == total_layers:
        include_norm = True
        include_lm_head = True

    if layer_start < 0 or layer_end > total_layers or layer_start >= layer_end:
        raise ValueError(
            f"Invalid layer range [{layer_start}, {layer_end}) "
            f"for model with {total_layers} layers"
        )

    target_dtype = dtype or getattr(config, "torch_dtype", torch.float16)
    if isinstance(target_dtype, str):
        target_dtype = getattr(torch, target_dtype)

    if isinstance(device, str) and "," in device:
        device_list = [torch.device(d.strip()) for d in device.split(",") if d.strip()]
    elif isinstance(device, (list, tuple)):
        device_list = [torch.device(d) for d in device]
    else:
        device_list = [torch.device(device)]

    target_device = device_list[0]
    num_slice_layers = layer_end - layer_start
    per_dev = (num_slice_layers + len(device_list) - 1) // len(device_list)

    def get_layer_device(local_idx: int) -> torch.device:
        if len(device_list) <= 1:
            return device_list[0]
        dev_idx = min(local_idx // per_dev, len(device_list) - 1)
        return device_list[dev_idx]

    # Force float16 if bfloat16 is requested but target GPU does not support hardware BF16 (e.g. Tesla P100 sm_60)
    if target_dtype == torch.bfloat16 and target_device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            logger.info("Target GPU %s does not support native BF16 — converting model and KV cache to torch.float16", target_device)
            target_dtype = torch.float16

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
        try:
            model = AutoModelForCausalLM.from_config(config, dtype=target_dtype, trust_remote_code=True)
        except Exception:
            try:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_config(config, dtype=target_dtype, trust_remote_code=True)
            except Exception:
                from transformers import AutoModel
                model = AutoModel.from_config(config, dtype=target_dtype, trust_remote_code=True)

    logger.info(
        "AFTER MODEL SHELL: allocated=%.2f GB | reserved=%.2f GB",
        torch.cuda.memory_allocated(target_device) / 1024**3 if target_device.type == "cuda" else 0.0,
        torch.cuda.memory_reserved(target_device) / 1024**3 if target_device.type == "cuda" else 0.0,
    )

    # 2. Extract requested layers on meta device
    base_model = getattr(model, "model", getattr(model, "language_model", getattr(model, "transformer", model)))
    if hasattr(base_model, "model"):
        base_model = base_model.model
    if hasattr(base_model, "language_model"):
        base_model = base_model.language_model

    layers_attr = getattr(base_model, "layers", getattr(model, "layers", None))
    if layers_attr is None:
        raise AttributeError(f"Could not locate 'layers' attribute in {type(model)} or {type(base_model)}")

    extracted_layers = nn.ModuleList([
        layers_attr[i] for i in range(layer_start, layer_end)
    ])
    norm = getattr(base_model, "norm", getattr(model, "norm", None)) if include_norm else None
    lm_head = getattr(model, "lm_head", None) if include_lm_head else None
    embed_tokens = getattr(base_model, "embed_tokens", getattr(model, "embed_tokens", None)) if include_embed else None

    rotary_emb = None
    if hasattr(base_model, "rotary_emb") and base_model.rotary_emb is not None:
        try:
            rotary_cls = type(base_model.rotary_emb)
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
            norm = norm.to_empty(device=device_list[-1])
        if embed_tokens is not None:
            embed_tokens = embed_tokens.to_empty(device=device_list[0])

    else:
        # FP16 / BF16 unquantized path
        for local_idx in range(len(extracted_layers)):
            layer_dev = get_layer_device(local_idx)
            extracted_layers[local_idx] = extracted_layers[local_idx].to_empty(device=layer_dev)
        if norm is not None:
            norm = norm.to_empty(device=device_list[-1])
        if lm_head is not None:
            lm_head = lm_head.to_empty(device=device_list[-1])
        if embed_tokens is not None:
            embed_tokens = embed_tokens.to_empty(device=device_list[0])

    logger.info(
        "AFTER DEVICE PLACEMENT: allocated=%.2f GB | reserved=%.2f GB",
        torch.cuda.memory_allocated(target_device) / 1024**3 if target_device.type == "cuda" else 0.0,
        torch.cuda.memory_reserved(target_device) / 1024**3 if target_device.type == "cuda" else 0.0,
    )

    # 4. Resolve targeted safetensors shards
    from safetensors.torch import load_file

    weight_map, local_dir = _get_safetensors_shards_map(model_path)
    target_prefixes = [f"model.layers.{i}." for i in range(layer_start, layer_end)]
    target_prefixes.extend([f"model.language_model.layers.{i}." for i in range(layer_start, layer_end)])
    target_prefixes.extend([f"language_model.model.layers.{i}." for i in range(layer_start, layer_end)])
    if include_norm:
        target_prefixes.extend(["model.norm.", "model.language_model.norm.", "language_model.norm."])
    if include_lm_head:
        target_prefixes.append("lm_head.")
    if include_embed:
        target_prefixes.extend(["model.embed_tokens.", "model.language_model.embed_tokens.", "language_model.embed_tokens."])

    if weight_map is not None:
        needed_shards = sorted({
            shard for key, shard in weight_map.items()
            if any(key.startswith(p) for p in target_prefixes)
        })
        logger.info("Matched %d targeted safetensors shards: %s", len(needed_shards), needed_shards)

        for shard_name in needed_shards:
            is_temp_download = False
            if local_dir is not None and (local_dir / shard_name).exists():
                shard_path = str(local_dir / shard_name)
            else:
                if os.path.exists("/kaggle"):
                    dest_file = f"/kaggle/working/{shard_name}"
                elif os.path.exists("/content"):
                    dest_file = f"/content/{shard_name}"
                else:
                    dest_file = f"/tmp/{shard_name}"

                shard_path = _download_shard_direct(model_path, shard_name, dest_file)
                is_temp_download = True

            logger.info("Streaming and loading weights from shard %s ...", shard_name)
            import psutil
            proc = psutil.Process()
            rss = proc.memory_info().rss / 1024**3
            avail = psutil.virtual_memory().available / 1024**3
            gpu_alloc = torch.cuda.memory_allocated(target_device) / 1024**3 if target_device.type == "cuda" else 0.0
            gpu_res = torch.cuda.memory_reserved(target_device) / 1024**3 if target_device.type == "cuda" else 0.0
            logger.info("SHARD START %s | RSS=%.2f GB | available=%.2f GB | GPU alloc=%.2f GB | GPU reserved=%.2f GB", shard_name, rss, avail, gpu_alloc, gpu_res)

            from safetensors.torch import load_file
            shard_dict = load_file(shard_path, device="cpu")
            for key, tensor in shard_dict.items():
                norm_key = key
                for prefix in ["model.language_model.", "language_model.model.", "language_model."]:
                    if norm_key.startswith(prefix):
                        norm_key = "model." + norm_key[len(prefix):]
                        break

                if not any(norm_key.startswith(p) or key.startswith(p) for p in target_prefixes):
                    continue

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
                    if norm_key.startswith("model.layers."):
                        parts = norm_key.split(".")
                        orig_idx = int(parts[2])
                        if layer_start <= orig_idx < layer_end:
                            local_idx = orig_idx - layer_start
                            layer_dev = get_layer_device(local_idx)
                            subpath = ".".join(parts[3:])
                            submod = extracted_layers[local_idx]
                            path_parts = subpath.split(".")
                            for part in path_parts[:-1]:
                                submod = getattr(submod, part)
                            param_name = path_parts[-1]
                            current_param = getattr(submod, param_name, None)
                            if current_param is not None and current_param.device == layer_dev:
                                current_param.data.copy_(tensor, non_blocking=False)
                            else:
                                param = nn.Parameter(
                                    tensor.to(device=layer_dev, dtype=target_dtype, non_blocking=False),
                                    requires_grad=False,
                                )
                                setattr(submod, param_name, param)

                    elif norm_key.startswith("model.norm.") and norm is not None:
                        norm_dev = device_list[-1]
                        if hasattr(norm, "weight") and norm.weight is not None and norm.weight.device == norm_dev:
                            norm.weight.data.copy_(tensor, non_blocking=False)
                        else:
                            norm.weight = nn.Parameter(
                                tensor.to(device=norm_dev, dtype=target_dtype, non_blocking=False),
                                requires_grad=False,
                            )
                    elif norm_key.startswith("lm_head.") and lm_head is not None:
                        head_dev = device_list[-1]
                        if hasattr(lm_head, "weight") and lm_head.weight is not None and lm_head.weight.device == head_dev:
                            lm_head.weight.data.copy_(tensor, non_blocking=False)
                        else:
                            lm_head.weight = nn.Parameter(
                                tensor.to(device=head_dev, dtype=target_dtype, non_blocking=False),
                                requires_grad=False,
                            )
                    elif norm_key.startswith("model.embed_tokens.") and embed_tokens is not None:
                        embed_dev = device_list[0]
                        if hasattr(embed_tokens, "weight") and embed_tokens.weight is not None and embed_tokens.weight.device == embed_dev:
                            embed_tokens.weight.data.copy_(tensor, non_blocking=False)
                        else:
                            embed_tokens.weight = nn.Parameter(
                                tensor.to(device=embed_dev, dtype=target_dtype, non_blocking=False),
                                requires_grad=False,
                            )

            del shard_dict
            if target_device.type == "cuda" and torch.cuda.is_available():
                for dev in device_list:
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                        torch.cuda.empty_cache()

            rss = proc.memory_info().rss / 1024**3
            avail = psutil.virtual_memory().available / 1024**3
            gpu_alloc = torch.cuda.memory_allocated(target_device) / 1024**3 if target_device.type == "cuda" else 0.0
            gpu_res = torch.cuda.memory_reserved(target_device) / 1024**3 if target_device.type == "cuda" else 0.0
            logger.info("SHARD END %s | RSS=%.2f GB | available=%.2f GB | GPU alloc=%.2f GB | GPU reserved=%.2f GB", shard_name, rss, avail, gpu_alloc, gpu_res)
            gc.collect()

            # Prune downloaded shard from disk on Kaggle/Colab
            if is_temp_download:
                try:
                    if os.path.exists(shard_path):
                        os.remove(shard_path)
                        logger.info("Pruned temporary shard %s from disk (disk freed, 0 cache residual)", shard_name)
                except Exception as e:
                    logger.warning("Could not prune temporary shard %s: %s", shard_name, e)

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

        if single_path is None:
            raise RuntimeError(
                f"Failed to load weights for {model_path}! "
                "Neither safetensors index (model.safetensors.index.json) nor model.safetensors could be found. "
                "Please verify model ID or ensure internet access to Hugging Face."
            )

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
                layer_dev = get_layer_device(idx)
                layer_sd = {
                    k[len(prefix):]: v.to(layer_dev, dtype=target_dtype)
                    for k, v in state_dict.items() if k.startswith(prefix)
                }
                extracted_layers[idx].load_state_dict(layer_sd, strict=False)
            if norm is not None:
                norm.load_state_dict({k[len("model.norm."):]: v.to(device_list[-1], dtype=target_dtype) for k, v in state_dict.items() if k.startswith("model.norm.")}, strict=False)
            if lm_head is not None:
                lm_head.load_state_dict({k[len("lm_head."):]: v.to(device_list[-1], dtype=target_dtype) for k, v in state_dict.items() if k.startswith("lm_head.")}, strict=False)
            if embed_tokens is not None:
                embed_tokens.load_state_dict({k[len("model.embed_tokens."):]: v.to(device_list[0], dtype=target_dtype) for k, v in state_dict.items() if k.startswith("model.embed_tokens.")}, strict=False)
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
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tc = getattr(config, "text_config", config)
    return {
        "model_type": getattr(config, "model_type", "unknown"),
        "num_layers": get_num_hidden_layers(config),
        "hidden_size": getattr(tc, "hidden_size", getattr(config, "hidden_size", 4096)),
        "num_attention_heads": getattr(tc, "num_attention_heads", getattr(config, "num_attention_heads", 32)),
        "num_kv_heads": getattr(tc, "num_key_value_heads", getattr(config, "num_key_value_heads", getattr(tc, "num_attention_heads", 32))),
        "vocab_size": getattr(tc, "vocab_size", getattr(config, "vocab_size", 152064)),
        "max_position_embeddings": getattr(tc, "max_position_embeddings", getattr(config, "max_position_embeddings", 32768)),
        "torch_dtype": str(getattr(config, "torch_dtype", torch.float16)),
    }
