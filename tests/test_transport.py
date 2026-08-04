"""
Unit tests for ShardFlow length-prefix protocol serialization and framing.
"""

import pytest
import torch
from shardflow.transport.protocol import (
    MessageType,
    TensorMessage,
    encode_message,
    decode_message,
)


def test_encode_decode_activation():
    tensor = torch.randn(1, 1, 2048, dtype=torch.bfloat16)
    msg = TensorMessage(
        msg_type=MessageType.ACTIVATION,
        session_id="test-session-12345",
        tensor=tensor,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        sample_on_node=True,
    )

    encoded = encode_message(msg)

    # Skip 8-byte length prefix
    decoded = decode_message(encoded[8:])

    assert decoded.msg_type == MessageType.ACTIVATION
    assert decoded.session_id == "test-session-12345"
    assert decoded.sample_on_node is True
    assert pytest.approx(decoded.temperature, abs=1e-3) == 0.7
    assert decoded.top_k == 40
    assert pytest.approx(decoded.top_p, abs=1e-3) == 0.9
    assert decoded.tensor is not None
    assert decoded.tensor.shape == (1, 1, 2048)
    assert decoded.tensor.dtype == torch.bfloat16
    assert torch.equal(decoded.tensor, tensor)


def test_encode_decode_token_id():
    msg = TensorMessage(
        msg_type=MessageType.TOKEN_ID,
        session_id="token-session-9999",
        token_id=12845,
    )

    encoded = encode_message(msg)
    decoded = decode_message(encoded[8:])

    assert decoded.msg_type == MessageType.TOKEN_ID
    assert decoded.session_id == "token-session-9999"
    assert decoded.token_id == 12845


def test_encode_decode_clear():
    msg = TensorMessage(
        msg_type=MessageType.CLEAR,
        session_id="clear-session-777",
    )

    encoded = encode_message(msg)
    decoded = decode_message(encoded[8:])

    assert decoded.msg_type == MessageType.CLEAR
    assert decoded.session_id == "clear-session-777"
    assert decoded.tensor is None
