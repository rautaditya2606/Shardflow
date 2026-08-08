import os
import asyncio
import pytest
import torch

from shardflow.node.layer_loader import load_layer_slice
from shardflow.node.node import PipelineNode
from shardflow.transport.protocol import MessageType, TensorMessage

MODEL_EXISTS = os.path.exists("./models/TinyLlama-1.1B-Chat-v1.0")


@pytest.mark.skipif(not MODEL_EXISTS, reason="Local TinyLlama-1.1B weights not found on disk")
@pytest.mark.asyncio
async def test_tinyllama_e2e_node_forward_generation():
    """Verify that a PipelineNode running real TinyLlama weights computes valid forward hidden states."""
    model_path = "./models/TinyLlama-1.1B-Chat-v1.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    slice0 = load_layer_slice(model_path, 0, 2, device=device)
    node = PipelineNode(
        model_slice=slice0,
        is_first_node=True,
        is_last_node=True,
        listen_host="127.0.0.1",
        listen_port=0,
        max_sessions=2,
        enable_cuda_graphs=torch.cuda.is_available(),
    )
    await node.start()

    session_id = "tinyllama-e2e-sess"

    # Step 1: Token embedding
    token_tensor = torch.tensor([[42]], device=node.model_slice.device, dtype=torch.long)
    hidden_states = node.model_slice.embed_tokens(token_tensor)
    assert hidden_states.shape == (1, 1, node.model_slice.config.hidden_size)

    # Step 2: Forward pass
    out = node._forward(hidden_states, session_id=session_id, compute_head=False)
    assert out.shape == (1, 1, node.model_slice.config.hidden_size)
    assert not torch.isnan(out).any()

    await node.stop()
