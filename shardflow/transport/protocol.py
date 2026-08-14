"""
Length-prefix TCP framing for tensor transfer.

Wire format:
    ┌──────────────┬─────────────────────────────────┐
    │ 8 bytes      │ N bytes                         │
    │ uint64 LE    │ raw tensor data (bf16/fp16)     │
    │ (payload len)│                                 │
    └──────────────┴─────────────────────────────────┘

Messages are structured as:
    [1 byte type] [payload-specific header] [tensor bytes]

Message types:
    0x01 ACTIVATION  — hidden states for a session
    0x02 CLEAR       — evict KV cache for a session
    0x03 LOGITS      — final logits from last node back to orchestrator
    
"""

import asyncio
import struct
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# 8-byte length prefix, little-endian unsigned 64-bit
LENGTH_PREFIX_FMT = "<Q"
LENGTH_PREFIX_SIZE = 8

# Header: 1 byte msg_type + 36 bytes session_id (UUID as string) + 8 bytes send_ts_us (uint64 microsecond timestamp)
HEADER_FMT = "<B36sQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# Tensor metadata: shape dims as int32 values, dtype as 1 byte
# For hidden_states [1, seq_len, hidden_dim]: 3 dims × 4 bytes + 1 byte dtype = 13 bytes
TENSOR_META_FMT = "<B"  # num_dims as uint8
DIM_FMT = "<I"  # each dim as uint32


PROTOCOL_VERSION: int = 2


class MessageType(IntEnum):
    """Wire message types."""
    ACTIVATION = 0x01
    CLEAR = 0x02
    LOGITS = 0x03
    TOKEN_ID = 0x04
    START_SESSION = 0x05
    STREAM_TOKEN = 0x06


DTYPE_MAP = {
    0: torch.float16,
    1: torch.bfloat16,
    2: torch.float32,
    3: torch.int64,
    4: torch.int32,
}
DTYPE_REVERSE = {v: k for k, v in DTYPE_MAP.items()}


@dataclass
class TensorMessage:
    """A framed message containing a tensor and metadata with wire telemetry timestamp."""
    msg_type: MessageType
    session_id: str
    send_ts_us: int = 0                   # Monotonic microsecond timestamp when sent
    tensor: Optional[torch.Tensor] = None  # None for CLEAR, TOKEN_ID, START_SESSION, STREAM_TOKEN
    token_id: Optional[int] = None        # Set for TOKEN_ID and STREAM_TOKEN messages
    temperature: float = 0.0              # Sampling temperature
    top_k: int = 0                        # Sampling top_k
    top_p: float = 1.0                    # Sampling top_p
    sample_on_node: bool = True           # Request node to perform GPU sampling
    prompt_tokens: Optional[list[int]] = None  # Token IDs for START_SESSION
    max_tokens: int = 128                 # Max tokens for START_SESSION
    eos_token_id: Optional[int] = None    # EOS token ID for data-plane decode loop
    is_eos: bool = False                  # Flag indicating EOS reached in STREAM_TOKEN
    finish_reason: Optional[str] = None   # Finish reason: "stop", "length", None
    stream_back_host: Optional[str] = None     # Host address to stream generated tokens to
    stream_back_port: Optional[int] = None     # Port address to stream generated tokens to
    accepted_count: int = 1                    # Number of accepted speculative tokens
    draft_tokens: Optional[list[int]] = None   # Speculative candidate token IDs



def _encode_tensor_meta(tensor: torch.Tensor) -> bytes:
    """Encode tensor shape and dtype as bytes."""
    dtype_code = DTYPE_REVERSE.get(tensor.dtype)
    if dtype_code is None:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}. Use fp16, bf16, or fp32.")

    parts = []
    ndim = len(tensor.shape)
    parts.append(struct.pack(TENSOR_META_FMT, ndim))
    for dim in tensor.shape:
        parts.append(struct.pack(DIM_FMT, dim))
    parts.append(struct.pack("<B", dtype_code))
    return b"".join(parts)


def _decode_tensor_meta(data: bytes, offset: int) -> tuple[list[int], torch.dtype, int]:
    """Decode tensor shape and dtype from bytes. Returns (shape, dtype, new_offset)."""
    ndim = struct.unpack_from(TENSOR_META_FMT, data, offset)[0]
    offset += struct.calcsize(TENSOR_META_FMT)

    shape = []
    for _ in range(ndim):
        dim = struct.unpack_from(DIM_FMT, data, offset)[0]
        shape.append(dim)
        offset += struct.calcsize(DIM_FMT)

    dtype_code = struct.unpack_from("<B", data, offset)[0]
    offset += 1

    dtype = DTYPE_MAP.get(dtype_code)
    if dtype is None:
        raise ValueError(f"Unknown dtype code: {dtype_code}")

    return shape, dtype, offset


SAMPLING_FMT = "<fHfB"  # temp (float32), top_k (uint16), top_p (float32), sample_flag (uint8)
SAMPLING_SIZE = struct.calcsize(SAMPLING_FMT)


def encode_message(msg: TensorMessage) -> bytes:
    """
    Encode a TensorMessage into wire format using single-buffer allocation.

    Returns: length-prefixed bytes ready to send.
    """
    import time
    session_id_bytes = msg.session_id.encode("ascii")
    send_ts = msg.send_ts_us if msg.send_ts_us > 0 else int(time.perf_counter() * 1_000_000)

    if msg.msg_type == MessageType.CLEAR:
        payload_len = HEADER_SIZE
        buf = bytearray(LENGTH_PREFIX_SIZE + payload_len)
        struct.pack_into(LENGTH_PREFIX_FMT, buf, 0, payload_len)
        struct.pack_into(HEADER_FMT, buf, LENGTH_PREFIX_SIZE, msg.msg_type, session_id_bytes, send_ts)
        return bytes(buf)

    elif msg.msg_type == MessageType.TOKEN_ID:
        payload_len = HEADER_SIZE + 8 + 4
        buf = bytearray(LENGTH_PREFIX_SIZE + payload_len)
        struct.pack_into(LENGTH_PREFIX_FMT, buf, 0, payload_len)
        struct.pack_into(HEADER_FMT, buf, LENGTH_PREFIX_SIZE, msg.msg_type, session_id_bytes, send_ts)
        struct.pack_into("<qI", buf, LENGTH_PREFIX_SIZE + HEADER_SIZE, msg.token_id or 0, msg.accepted_count or 1)
        return bytes(buf)

    elif msg.msg_type == MessageType.STREAM_TOKEN:
        # STREAM_TOKEN: session_id + send_ts + token_id (q) + is_eos (B) + finish_reason (16s)
        reason_bytes = (msg.finish_reason or "").encode("ascii")[:16]
        payload_len = HEADER_SIZE + 8 + 1 + 16
        buf = bytearray(LENGTH_PREFIX_SIZE + payload_len)
        struct.pack_into(LENGTH_PREFIX_FMT, buf, 0, payload_len)
        offset = LENGTH_PREFIX_SIZE
        struct.pack_into(HEADER_FMT, buf, offset, msg.msg_type, session_id_bytes, send_ts)
        offset += HEADER_SIZE
        struct.pack_into("<qB16s", buf, offset, msg.token_id or 0, 1 if msg.is_eos else 0, reason_bytes)
        return bytes(buf)

    elif msg.msg_type == MessageType.START_SESSION:
        host_bytes = (msg.stream_back_host or "").encode("utf-8")
        tokens = msg.prompt_tokens or []

        # SAMPLING_FMT (temp, top_k, top_p, sample_flag) + max_tokens (I) + host_len (H) + host_bytes + port (H) + eos_token_id (q) + num_tokens (I) + tokens (q*N)
        extra_len = SAMPLING_SIZE + 4 + 2 + len(host_bytes) + 2 + 8 + 4 + (8 * len(tokens))
        payload_len = HEADER_SIZE + extra_len
        buf = bytearray(LENGTH_PREFIX_SIZE + payload_len)

        struct.pack_into(LENGTH_PREFIX_FMT, buf, 0, payload_len)
        offset = LENGTH_PREFIX_SIZE

        struct.pack_into(HEADER_FMT, buf, offset, msg.msg_type, session_id_bytes, send_ts)
        offset += HEADER_SIZE

        struct.pack_into(
            SAMPLING_FMT,
            buf,
            offset,
            float(msg.temperature),
            int(msg.top_k),
            float(msg.top_p),
            1 if msg.sample_on_node else 0,
        )
        offset += SAMPLING_SIZE

        struct.pack_into("<IH", buf, offset, int(msg.max_tokens), len(host_bytes))
        offset += 6

        if host_bytes:
            buf[offset:offset + len(host_bytes)] = host_bytes
            offset += len(host_bytes)

        eos_val = msg.eos_token_id if msg.eos_token_id is not None else -1
        struct.pack_into("<HqI", buf, offset, int(msg.stream_back_port or 0), int(eos_val), len(tokens))
        offset += 14

        for token in tokens:
            struct.pack_into("<q", buf, offset, int(token))
            offset += 8

        return bytes(buf)

    else:
        # ACTIVATION / LOGITS
        tensor = msg.tensor
        if tensor is None:
            raise ValueError(f"Tensor required for message type {msg.msg_type}")

        tensor = tensor.contiguous()
        tensor_meta = _encode_tensor_meta(tensor)
        # ponytail: view(uint8).cpu().numpy().tobytes() is 700x faster than bytes(tensor.untyped_storage())
        tensor_bytes = tensor.view(torch.uint8).cpu().numpy().tobytes()

        draft_tokens = msg.draft_tokens or []
        draft_meta_len = 2 + (8 * len(draft_tokens))
        host_bytes = (msg.stream_back_host or "").encode("utf-8")
        stream_port = int(msg.stream_back_port or 0)
        stream_meta_len = 6 + len(host_bytes) + draft_meta_len

        payload_len = HEADER_SIZE + SAMPLING_SIZE + stream_meta_len + len(tensor_meta) + len(tensor_bytes)
        buf = bytearray(LENGTH_PREFIX_SIZE + payload_len)

        struct.pack_into(LENGTH_PREFIX_FMT, buf, 0, payload_len)
        offset = LENGTH_PREFIX_SIZE

        struct.pack_into(HEADER_FMT, buf, offset, msg.msg_type, session_id_bytes, send_ts)
        offset += HEADER_SIZE

        struct.pack_into(
            SAMPLING_FMT,
            buf,
            offset,
            float(msg.temperature),
            int(msg.top_k),
            float(msg.top_p),
            1 if msg.sample_on_node else 0,
        )
        offset += SAMPLING_SIZE

        struct.pack_into("<HHH", buf, offset, stream_port, len(host_bytes), len(draft_tokens))
        offset += 6

        if host_bytes:
            buf[offset:offset + len(host_bytes)] = host_bytes
            offset += len(host_bytes)

        for d_tok in draft_tokens:
            struct.pack_into("<q", buf, offset, int(d_tok))
            offset += 8

        buf[offset:offset + len(tensor_meta)] = tensor_meta
        offset += len(tensor_meta)

        buf[offset:offset + len(tensor_bytes)] = tensor_bytes
        return bytes(buf)


def decode_message(data: bytes) -> TensorMessage:
    """
    Decode a wire-format payload (after length prefix has been read) into a TensorMessage.
    """
    msg_type_raw, session_id_raw, send_ts_us = struct.unpack_from(HEADER_FMT, data, 0)
    msg_type = MessageType(msg_type_raw)
    session_id = session_id_raw.decode("ascii").strip("\x00")
    offset = HEADER_SIZE

    if msg_type == MessageType.CLEAR:
        return TensorMessage(msg_type=msg_type, session_id=session_id, send_ts_us=send_ts_us)

    if msg_type == MessageType.TOKEN_ID:
        token_id = struct.unpack_from("<q", data, offset)[0]
        accepted_count = 1
        if len(data) >= offset + 12:
            accepted_count = struct.unpack_from("<I", data, offset + 8)[0]
        return TensorMessage(
            msg_type=msg_type,
            session_id=session_id,
            send_ts_us=send_ts_us,
            token_id=token_id,
            accepted_count=accepted_count,
        )

    if msg_type == MessageType.STREAM_TOKEN:
        token_id, is_eos_val, reason_raw = struct.unpack_from("<qB16s", data, offset)
        reason = reason_raw.decode("ascii").strip("\x00")
        return TensorMessage(
            msg_type=msg_type,
            session_id=session_id,
            send_ts_us=send_ts_us,
            token_id=token_id,
            is_eos=bool(is_eos_val),
            finish_reason=reason if reason else None,
        )

    if msg_type == MessageType.START_SESSION:
        temp, top_k, top_p, sample_flag = struct.unpack_from(SAMPLING_FMT, data, offset)
        offset += SAMPLING_SIZE

        max_tokens, host_len = struct.unpack_from("<IH", data, offset)
        offset += 6

        stream_host = ""
        if host_len > 0:
            stream_host = data[offset:offset + host_len].decode("utf-8")
            offset += host_len

        # Check if extended v2 START_SESSION with eos_token_id (14 bytes) or v1 (6 bytes)
        eos_token_id = None
        remaining_header = len(data) - offset
        if remaining_header >= 14:
            stream_port, eos_val, num_tokens = struct.unpack_from("<HqI", data, offset)
            offset += 14
            eos_token_id = eos_val if eos_val >= 0 else None
        else:
            stream_port, num_tokens = struct.unpack_from("<HI", data, offset)
            offset += 6

        prompt_tokens = []
        for _ in range(num_tokens):
            t_id = struct.unpack_from("<q", data, offset)[0]
            prompt_tokens.append(t_id)
            offset += 8

        return TensorMessage(
            msg_type=msg_type,
            session_id=session_id,
            send_ts_us=send_ts_us,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            sample_on_node=bool(sample_flag),
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            eos_token_id=eos_token_id,
            stream_back_host=stream_host if stream_host else None,
            stream_back_port=stream_port if stream_port > 0 else None,
        )

    # Parse sampling options
    temp, top_k, top_p, sample_flag = struct.unpack_from(SAMPLING_FMT, data, offset)
    offset += SAMPLING_SIZE

    stream_port, host_len, num_drafts = struct.unpack_from("<HHH", data, offset)
    offset += 6

    stream_host = ""
    if host_len > 0:
        stream_host = data[offset:offset + host_len].decode("utf-8")
        offset += host_len

    draft_tokens = None
    if num_drafts > 0:
        draft_tokens = []
        for _ in range(num_drafts):
            d_tok = struct.unpack_from("<q", data, offset)[0]
            draft_tokens.append(d_tok)
            offset += 8

    # Parse tensor metadata
    shape, dtype, offset = _decode_tensor_meta(data, offset)

    # Parse tensor bytes — zero-copy from writable buffer (ponytail: no redundant clone)
    numel = 1
    for dim in shape:
        numel *= dim
    mutable_data = bytearray(data)
    tensor = torch.frombuffer(mutable_data, dtype=dtype, count=numel, offset=offset).reshape(shape)

    return TensorMessage(
        msg_type=msg_type,
        session_id=session_id,
        send_ts_us=send_ts_us,
        tensor=tensor,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        sample_on_node=bool(sample_flag),
        stream_back_host=stream_host if stream_host else None,
        stream_back_port=stream_port if stream_port > 0 else None,
        draft_tokens=draft_tokens,
    )


async def send_message(
    writer: asyncio.StreamWriter,
    msg: TensorMessage,
) -> None:
    """
    Send a length-prefixed tensor message over an async TCP connection.
    Unconditionally drains to flush bytes to the OS kernel socket immediately.
    """
    import time
    if msg.send_ts_us <= 0:
        msg.send_ts_us = int(time.perf_counter() * 1_000_000)
    data = encode_message(msg)
    writer.write(data)
    await writer.drain()
    logger.debug(
        "Sent %s for session %s (%d bytes)",
        msg.msg_type.name, msg.session_id, len(data)
    )


async def recv_message(
    reader: asyncio.StreamReader,
    timeout: float = 60.0,
) -> TensorMessage:
    """
    Receive a length-prefixed tensor message from an async TCP connection.
    """
    # Read length prefix
    length_bytes = await asyncio.wait_for(
        reader.readexactly(LENGTH_PREFIX_SIZE),
        timeout=timeout,
    )
    payload_length = struct.unpack(LENGTH_PREFIX_FMT, length_bytes)[0]

    # Read payload
    payload = await asyncio.wait_for(
        reader.readexactly(payload_length),
        timeout=timeout,
    )

    msg = decode_message(payload)
    logger.debug(
        "Received %s for session %s (%d bytes payload)",
        msg.msg_type.name, msg.session_id, payload_length
    )
    return msg
