"""
Unit tests for wire protocol framing, binary packing, microsecond timestamps, and zero-copy deserialization.
"""

import time
import torch
import pytest

from shardflow.transport.protocol import (
    MessageType,
    TensorMessage,
    encode_message,
    decode_message,
    LENGTH_PREFIX_SIZE,
    HEADER_SIZE,
)


def test_protocol_constants_and_sizes():
    """Verify wire constants match protocol spec."""
    assert LENGTH_PREFIX_SIZE == 8
    assert HEADER_SIZE == 45  # 1 byte type + 36 bytes session_id + 8 bytes send_ts_us


def test_timestamp_preservation_and_uint64_range():
    """Verify 8-byte uint64 microsecond timestamp is bit-preserved without clock assumptions."""
    explicit_ts = 1700000000123456  # Arbitrary 64-bit microsecond integer
    msg = TensorMessage(
        msg_type=MessageType.TOKEN_ID,
        session_id="ts-test-sess",
        token_id=55,
        send_ts_us=explicit_ts,
    )

    raw_bytes = encode_message(msg)
    decoded = decode_message(raw_bytes[LENGTH_PREFIX_SIZE:])

    assert decoded.send_ts_us == explicit_ts
    assert decoded.token_id == 55


def test_clear_message_framing():
    """Verify CLEAR message binary serialization."""
    session_id = "test-session-1234"
    msg = TensorMessage(msg_type=MessageType.CLEAR, session_id=session_id)

    raw_bytes = encode_message(msg)
    assert len(raw_bytes) == LENGTH_PREFIX_SIZE + HEADER_SIZE

    decoded = decode_message(raw_bytes[LENGTH_PREFIX_SIZE:])
    assert decoded.msg_type == MessageType.CLEAR
    assert decoded.session_id == session_id
    assert decoded.send_ts_us > 0


def test_token_id_message_framing():
    """Verify TOKEN_ID message binary serialization."""
    session_id = "token-session-5678"
    token_id = 4294967295
    msg = TensorMessage(msg_type=MessageType.TOKEN_ID, session_id=session_id, token_id=token_id)

    raw_bytes = encode_message(msg)
    decoded = decode_message(raw_bytes[LENGTH_PREFIX_SIZE:])

    assert decoded.msg_type == MessageType.TOKEN_ID
    assert decoded.session_id == session_id
    assert decoded.token_id == token_id
    assert decoded.send_ts_us > 0


def test_stream_token_message_framing():
    """Verify STREAM_TOKEN message with finish reason."""
    session_id = "stream-session-9999"
    msg = TensorMessage(
        msg_type=MessageType.STREAM_TOKEN,
        session_id=session_id,
        token_id=1024,
        is_eos=True,
        finish_reason="stop",
    )

    raw_bytes = encode_message(msg)
    decoded = decode_message(raw_bytes[LENGTH_PREFIX_SIZE:])

    assert decoded.msg_type == MessageType.STREAM_TOKEN
    assert decoded.session_id == session_id
    assert decoded.token_id == 1024
    assert decoded.is_eos is True
    assert decoded.finish_reason == "stop"


def test_activation_tensor_bit_exact_framing():
    """Verify ACTIVATION message preserves exact tensor floating-point values and shape."""
    session_id = "tensor-session-001"
    tensor = torch.randn(1, 4, 32, dtype=torch.float16)

    msg = TensorMessage(
        msg_type=MessageType.ACTIVATION,
        session_id=session_id,
        tensor=tensor,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        sample_on_node=True,
        stream_back_host="127.0.0.1",
        stream_back_port=9600,
    )

    raw_bytes = encode_message(msg)
    decoded = decode_message(raw_bytes[LENGTH_PREFIX_SIZE:])

    assert decoded.msg_type == MessageType.ACTIVATION
    assert decoded.session_id == session_id
    assert decoded.temperature == pytest.approx(0.7, abs=1e-4)
    assert decoded.top_k == 50
    assert decoded.top_p == pytest.approx(0.9, abs=1e-4)
    assert decoded.sample_on_node is True
    assert decoded.stream_back_host == "127.0.0.1"
    assert decoded.stream_back_port == 9600

    assert decoded.tensor is not None
    assert decoded.tensor.shape == (1, 4, 32)
    assert decoded.tensor.dtype == torch.float16
    assert torch.equal(decoded.tensor, tensor)


def test_start_session_prompt_tokens_framing():
    """Verify START_SESSION payload encodes prompt tokens and generation parameters."""
    session_id = "start-session-abc"
    tokens = [101, 2054, 2003, 1037, 102]

    msg = TensorMessage(
        msg_type=MessageType.START_SESSION,
        session_id=session_id,
        prompt_tokens=tokens,
        max_tokens=64,
        eos_token_id=2,
        temperature=0.0,
        stream_back_host="10.0.0.5",
        stream_back_port=9500,
    )

    raw_bytes = encode_message(msg)
    decoded = decode_message(raw_bytes[LENGTH_PREFIX_SIZE:])

    assert decoded.msg_type == MessageType.START_SESSION
    assert decoded.session_id == session_id
    assert decoded.prompt_tokens == tokens
    assert decoded.max_tokens == 64
    assert decoded.eos_token_id == 2
    assert decoded.temperature == 0.0
    assert decoded.stream_back_host == "10.0.0.5"
    assert decoded.stream_back_port == 9500


def test_activation_draft_tokens_framing():
    """Verify ACTIVATION message encodes and decodes candidate draft tokens."""
    session_id = "draft-session-001"
    tensor = torch.randn(1, 4, 32, dtype=torch.float16)
    draft_tokens = [1024, 2048, 3072, 4096]

    msg = TensorMessage(
        msg_type=MessageType.ACTIVATION,
        session_id=session_id,
        tensor=tensor,
        draft_tokens=draft_tokens,
        sample_on_node=True,
    )

    raw_bytes = encode_message(msg)
    decoded = decode_message(raw_bytes[LENGTH_PREFIX_SIZE:])

    assert decoded.msg_type == MessageType.ACTIVATION
    assert decoded.session_id == session_id
    assert decoded.draft_tokens == draft_tokens
    assert decoded.tensor is not None
    assert torch.equal(decoded.tensor, tensor)


def test_token_id_accepted_count_framing():
    """Verify TOKEN_ID message encodes and decodes speculative accepted_count."""
    session_id = "spec-token-sess"
    msg = TensorMessage(
        msg_type=MessageType.TOKEN_ID,
        session_id=session_id,
        token_id=50256,
        accepted_count=4,
    )

    raw_bytes = encode_message(msg)
    decoded = decode_message(raw_bytes[LENGTH_PREFIX_SIZE:])

    assert decoded.msg_type == MessageType.TOKEN_ID
    assert decoded.session_id == session_id
    assert decoded.token_id == 50256
    assert decoded.accepted_count == 4

