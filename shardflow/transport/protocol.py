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

# Header: 1 byte msg_type + 36 bytes session_id (UUID as string)
HEADER_FMT = "<B36s"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# Tensor metadata: shape dims as int32 values, dtype as 1 byte
# For hidden_states [1, seq_len, hidden_dim]: 3 dims × 4 bytes + 1 byte dtype = 13 bytes
TENSOR_META_FMT = "<B"  # num_dims as uint8
DIM_FMT = "<I"  # each dim as uint32


class MessageType(IntEnum):
    """Wire message types."""
    ACTIVATION = 0x01
    CLEAR = 0x02
    LOGITS = 0x03
    TOKEN_ID = 0x04
    START_SESSION = 0x05


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
    """A framed message containing a tensor and metadata."""
    msg_type: MessageType
    session_id: str
    tensor: Optional[torch.Tensor] = None  # None for CLEAR, TOKEN_ID, or START_SESSION messages
    token_id: Optional[int] = None        # Set for TOKEN_ID messages
    temperature: float = 0.0              # Sampling temperature
    top_k: int = 0                        # Sampling top_k
    top_p: float = 1.0                    # Sampling top_p
    sample_on_node: bool = True           # Request node to perform GPU sampling
    prompt_tokens: Optional[list[int]] = None  # Token IDs for START_SESSION
    max_tokens: int = 128                 # Max tokens for START_SESSION
    stream_back_host: Optional[str] = None     # Host address to stream generated tokens to
    stream_back_port: Optional[int] = None     # Port address to stream generated tokens to



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
    session_id_bytes = msg.session_id.encode("ascii")

    if msg.msg_type == MessageType.CLEAR:
        payload_len = HEADER_SIZE
        buf = bytearray(LENGTH_PREFIX_SIZE + payload_len)
        struct.pack_into(LENGTH_PREFIX_FMT, buf, 0, payload_len)
        struct.pack_into(HEADER_FMT, buf, LENGTH_PREFIX_SIZE, msg.msg_type, session_id_bytes)
        return bytes(buf)

    elif msg.msg_type == MessageType.TOKEN_ID:
        payload_len = HEADER_SIZE + 8
        buf = bytearray(LENGTH_PREFIX_SIZE + payload_len)
        struct.pack_into(LENGTH_PREFIX_FMT, buf, 0, payload_len)
        struct.pack_into(HEADER_FMT, buf, LENGTH_PREFIX_SIZE, msg.msg_type, session_id_bytes)
        struct.pack_into("<q", buf, LENGTH_PREFIX_SIZE + HEADER_SIZE, msg.token_id or 0)
        return bytes(buf)

    elif msg.msg_type == MessageType.START_SESSION:
        host_bytes = (msg.stream_back_host or "").encode("utf-8")
        tokens = msg.prompt_tokens or []

        # SAMPLING_FMT (temp, top_k, top_p, sample_flag) + max_tokens (I) + host_len (H) + host_bytes + port (H) + num_tokens (I) + tokens (q*N)
        extra_len = SAMPLING_SIZE + 4 + 2 + len(host_bytes) + 2 + 4 + (8 * len(tokens))
        payload_len = HEADER_SIZE + extra_len
        buf = bytearray(LENGTH_PREFIX_SIZE + payload_len)

        struct.pack_into(LENGTH_PREFIX_FMT, buf, 0, payload_len)
        offset = LENGTH_PREFIX_SIZE

        struct.pack_into(HEADER_FMT, buf, offset, msg.msg_type, session_id_bytes)
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

        struct.pack_into("<HI", buf, offset, int(msg.stream_back_port or 0), len(tokens))
        offset += 6

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

        payload_len = HEADER_SIZE + SAMPLING_SIZE + len(tensor_meta) + len(tensor_bytes)
        buf = bytearray(LENGTH_PREFIX_SIZE + payload_len)

        struct.pack_into(LENGTH_PREFIX_FMT, buf, 0, payload_len)
        offset = LENGTH_PREFIX_SIZE

        struct.pack_into(HEADER_FMT, buf, offset, msg.msg_type, session_id_bytes)
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

        buf[offset:offset + len(tensor_meta)] = tensor_meta
        offset += len(tensor_meta)

        buf[offset:offset + len(tensor_bytes)] = tensor_bytes
        return bytes(buf)


def decode_message(data: bytes) -> TensorMessage:
    """
    Decode a wire-format payload (after length prefix has been read) into a TensorMessage.
    """
    msg_type_raw, session_id_raw = struct.unpack_from(HEADER_FMT, data, 0)
    msg_type = MessageType(msg_type_raw)
    session_id = session_id_raw.decode("ascii").strip("\x00")
    offset = HEADER_SIZE

    if msg_type == MessageType.CLEAR:
        return TensorMessage(msg_type=msg_type, session_id=session_id)

    if msg_type == MessageType.TOKEN_ID:
        token_id = struct.unpack_from("<q", data, offset)[0]
        return TensorMessage(msg_type=msg_type, session_id=session_id, token_id=token_id)

    if msg_type == MessageType.START_SESSION:
        temp, top_k, top_p, sample_flag = struct.unpack_from(SAMPLING_FMT, data, offset)
        offset += SAMPLING_SIZE

        max_tokens, host_len = struct.unpack_from("<IH", data, offset)
        offset += 6

        stream_host = ""
        if host_len > 0:
            stream_host = data[offset:offset + host_len].decode("utf-8")
            offset += host_len

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
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            sample_on_node=bool(sample_flag),
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            stream_back_host=stream_host if stream_host else None,
            stream_back_port=stream_port if stream_port > 0 else None,
        )

    # Parse sampling options
    temp, top_k, top_p, sample_flag = struct.unpack_from(SAMPLING_FMT, data, offset)
    offset += SAMPLING_SIZE

    # Parse tensor metadata
    shape, dtype, offset = _decode_tensor_meta(data, offset)

    # Parse tensor bytes — zero-copy from buffer
    numel = 1
    for dim in shape:
        numel *= dim
    tensor = torch.frombuffer(data, dtype=dtype, count=numel, offset=offset).reshape(shape).clone()

    return TensorMessage(
        msg_type=msg_type,
        session_id=session_id,
        tensor=tensor,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        sample_on_node=bool(sample_flag),
    )


async def send_message(
    writer: asyncio.StreamWriter,
    msg: TensorMessage,
) -> None:
    """Send a length-prefixed tensor message over an async TCP connection."""
    data = encode_message(msg)
    writer.write(data)
    # ponytail: only drain under backpressure — avoids a kernel RTT every token
    if writer.transport.get_write_buffer_size() > 65536:
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

    Args:
        reader: asyncio StreamReader
        timeout: seconds before raising TimeoutError (default 5s)

    Raises:
        asyncio.TimeoutError: if no data received within timeout
        ConnectionError: if connection is closed
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
