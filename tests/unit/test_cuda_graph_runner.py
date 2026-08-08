import os
import pytest
import torch
from transformers.cache_utils import StaticCache
from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.cuda_graph import CUDAGraphRunner

MODEL_EXISTS = os.path.exists("./models/TinyLlama-1.1B-Chat-v1.0")


@pytest.mark.skipif(not (torch.cuda.is_available() and MODEL_EXISTS), reason="CUDA and local weights required")
def test_cuda_graph_runner_decode_and_verify_capture():
    """Verify CUDAGraphRunner captures both single-token decode and multi-token verify graphs."""
    model_path = "./models/TinyLlama-1.1B-Chat-v1.0"
    device = torch.device("cuda")

    model_slice = load_layer_slice(model_path, 0, 2, device="cuda")
    dtype = next(model_slice.layers[0].parameters()).dtype
    config = model_slice.config

    static_cache = StaticCache(
        config=config,
        max_batch_size=1,
        max_cache_len=128,
        device=device,
        dtype=dtype,
    )

    runner = CUDAGraphRunner(
        layers=model_slice.layers,
        hidden_size=config.hidden_size,
        device=device,
        dtype=dtype,
        spec_k=4,
        rotary_emb=model_slice.rotary_emb,
        enabled=True,
    )

    captured = runner.capture(static_cache)
    assert captured is True
    assert runner.is_captured is True
    assert runner.can_use_graph(1) is True
    assert runner.can_use_graph(4) is True
    assert runner.can_use_graph(5) is False

    # Eager vs Graph replay parity
    test_input = torch.randn((1, 1, config.hidden_size), device=device, dtype=dtype)
    pos_ids = torch.tensor([[0]], device=device, dtype=torch.long)
    pos_emb = model_slice.rotary_emb(test_input, pos_ids) if model_slice.rotary_emb else None

    eager_cache = StaticCache(config=config, max_batch_size=1, max_cache_len=128, device=device, dtype=dtype)
    h_eager = test_input.clone()
    for layer in model_slice.layers:
        kwargs = {"position_ids": pos_ids, "past_key_values": eager_cache, "use_cache": True}
        if pos_emb is not None:
            kwargs["position_embeddings"] = pos_emb
        out = layer(h_eager, **kwargs)
        h_eager = out[0] if isinstance(out, tuple) else out

    h_graph = runner.replay_decode(test_input, position=0)
    diff = torch.max(torch.abs(h_eager - h_graph)).item()
    assert diff < 1e-2, f"Eager vs CUDA Graph output discrepancy too high: {diff}"
