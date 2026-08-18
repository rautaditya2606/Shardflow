"""
ShardFlow v2 — TCP Relay Transport Layer.

Provides direct TCP transport through the Rust relay running on EC2 (or private host).
Implements length-prefixed tensor framing, symmetric READY handshakes, socket buffer optimizations,
socket timeout guards, and zero-copy tensor serialization.
"""

import socket
import struct
import time
import logging
from typing import Optional, Tuple, List, Union
import torch

import os

logger = logging.getLogger(__name__)

# Default Relay Configuration (overridable via SHARDFLOW_RELAY_HOST or RELAY_HOST env vars)
RELAY_HOST = os.getenv("SHARDFLOW_RELAY_HOST", os.getenv("RELAY_HOST", "127.0.0.1"))
RELAY_PORT = int(os.getenv("SHARDFLOW_RELAY_PORT", os.getenv("RELAY_PORT", "9500")))
AUTH_BYTE = bytes([0xAD])

# Default Socket Timeout in seconds (prevents hanging if a Kaggle session dies)
SOCKET_DATA_TIMEOUT = 120.0

# Wire Framing Constants (8-byte big-endian unsigned 64-bit length prefix)
LENGTH_PREFIX_FMT = ">Q"
LENGTH_PREFIX_SIZE = 8

# Message Types
MSG_TENSOR = 0x01
MSG_TOKEN = 0x02
MSG_CONTROL = 0x03

# Supported Dtypes Map
DTYPE_MAP = {
    0: torch.float16,
    1: torch.bfloat16,
    2: torch.float32,
    3: torch.int64,
    4: torch.int32,
}
DTYPE_REVERSE = {v: k for k, v in DTYPE_MAP.items()}


def configure_socket(sock: socket.socket, data_timeout: float = SOCKET_DATA_TIMEOUT) -> None:
    """Apply high-performance low-latency TCP socket configurations and data timeout."""
    # Disable Nagle's algorithm for immediate transmission of ~10KB activation frames
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as e:
        logger.debug("Failed to set TCP_NODELAY: %s", e)

    # Enable TCP Keepalive
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except Exception as e:
        logger.debug("Failed to set SO_KEEPALIVE: %s", e)

    # Linux-specific keepalive intervals
    if hasattr(socket, "TCP_KEEPIDLE"):
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
        except Exception:
            pass

    # Linux QUICKACK for eliminating delayed ACK pauses
    if hasattr(socket, "TCP_QUICKACK"):
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
        except Exception:
            pass

    # Expand OS socket buffer sizes to 4MB to fit large prompt prefill activations in one write
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    except Exception as e:
        logger.debug("Failed to expand socket buffers: %s", e)

    # Apply socket timeout
    sock.settimeout(data_timeout)


def recvall(sock: socket.socket, n: int) -> bytes:
    """
    Read exactly n bytes from a TCP socket.
    Raises TimeoutError with actionable diagnostics if socket times out.
    """
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    try:
        while pos < n:
            nbytes = sock.recv_into(view[pos:], n - pos)
            if nbytes == 0:
                raise ConnectionError(
                    f"Socket connection closed unexpectedly after reading {pos}/{n} bytes. "
                    "The remote node or relay connection was terminated."
                )
            pos += nbytes
        return bytes(buf)
    except socket.timeout as e:
        raise TimeoutError(
            f"Socket timed out while waiting for data ({pos}/{n} bytes read). "
            f"The peer Kaggle session or relay may have died, timed out, or disconnected."
        ) from e


def connect_to_relay(
    host: str = RELAY_HOST,
    port: int = RELAY_PORT,
    auth_byte: bytes = AUTH_BYTE,
    connect_timeout: float = 30.0,
    data_timeout: float = SOCKET_DATA_TIMEOUT,
) -> socket.socket:
    """
    Create TCP socket, configure TCP_NODELAY, connect to relay, and send auth byte.

    Args:
        host: Relay IP address or hostname (default: RELAY_HOST)
        port: Relay port (default: 9500)
        auth_byte: 1-byte authorization token (default: 0xAD)
        connect_timeout: Timeout for initial socket connection
        data_timeout: Timeout for subsequent send/recv operations (default: 30s)

    Returns:
        Connected and configured socket.socket
    """
    logger.info("Connecting to TCP relay at %s:%d ...", host, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(connect_timeout)

    try:
        sock.connect((host, port))
        # Send auth byte immediately to claim/pair slot on relay
        sock.sendall(auth_byte)
        # Apply full socket tuning + data timeout
        configure_socket(sock, data_timeout=data_timeout)
        logger.info("[OK] Connected to relay at %s:%d (auth byte sent, timeout=%0.1fs)", host, port, data_timeout)
        return sock
    except Exception as e:
        sock.close()
        raise ConnectionError(f"Failed to connect to relay at {host}:{port}: {e}") from e


HANDSHAKE_MAGIC = b"SF_READY"  # Exactly 8 bytes


def handshake(sock: socket.socket, is_initiator: bool = True, timeout: float = 300.0) -> None:
    """
    Execute clean two-way READY handshake through the EC2 relay.

    Node 0 (is_initiator=True): sends SF_READY -> waits for SF_READY response.
    Node 1 (is_initiator=False): waits for SF_READY -> sends SF_READY response.
    """
    orig_timeout = sock.gettimeout()
    sock.settimeout(timeout)
    try:
        if is_initiator:
            logger.info("Sending READY handshake to peer through relay (waiting up to %0.0fs for Node 1)...", timeout)
            sock.sendall(HANDSHAKE_MAGIC)
            try:
                ack = recvall(sock, len(HANDSHAKE_MAGIC))
            except TimeoutError as e:
                raise TimeoutError(
                    f"Handshake timed out after {timeout}s. Node 1 has not connected to the relay yet. "
                    "Make sure Node 1 (scripts/colab_node1.py) is started and connected before running Node 0."
                ) from e
            if ack != HANDSHAKE_MAGIC:
                raise ConnectionError(f"Handshake failed: expected {HANDSHAKE_MAGIC!r}, received {ack!r}")
            logger.info("[OK] Handshake successful: Peer node is READY!")
        else:
            logger.info("Waiting up to %0.0fs for READY handshake from Node 0 through relay...", timeout)
            try:
                ping = recvall(sock, len(HANDSHAKE_MAGIC))
            except TimeoutError as e:
                raise TimeoutError(
                    f"Handshake timed out after {timeout}s waiting for Node 0 to start. "
                    "Make sure Node 0 (scripts/colab_node0.py) is launched."
                ) from e
            if ping != HANDSHAKE_MAGIC:
                raise ConnectionError(f"Handshake failed: expected {HANDSHAKE_MAGIC!r}, received {ping!r}")
            sock.sendall(HANDSHAKE_MAGIC)
            logger.info("[OK] Handshake successful: Peer node is READY!")
    finally:
        sock.settimeout(orig_timeout)


def send_tensor(
    sock: socket.socket,
    tensor: torch.Tensor,
    draft_tokens: Optional[List[int]] = None,
    round_id: int = 0,
    parent_round_id: int = 0,
) -> None:
    """
    Serialize and send a tensor with 8-byte big-endian length prefix framing.
    """
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
        cpu_tensor = tensor.contiguous().cpu()
    else:
        cpu_tensor = tensor.contiguous()

    tensor_bytes = cpu_tensor.view(torch.uint8).numpy().tobytes()

    dtype_code = DTYPE_REVERSE.get(cpu_tensor.dtype, 0)
    shape = list(cpu_tensor.shape)
    ndim = len(shape)

    drafts = draft_tokens or []
    num_drafts = len(drafts)

    payload_len = 1 + 4 + 4 + 2 + (8 * num_drafts) + 1 + (4 * ndim) + 1 + len(tensor_bytes)

    buf = bytearray(LENGTH_PREFIX_SIZE + payload_len)
    struct.pack_into(LENGTH_PREFIX_FMT, buf, 0, payload_len)
    offset = LENGTH_PREFIX_SIZE

    struct.pack_into(">BIIH", buf, offset, MSG_TENSOR, int(round_id), int(parent_round_id), num_drafts)
    offset += 1 + 4 + 4 + 2
    for d in drafts:
        struct.pack_into(">q", buf, offset, int(d))
        offset += 8

    struct.pack_into(">B", buf, offset, ndim)
    offset += 1
    for dim in shape:
        struct.pack_into(">I", buf, offset, int(dim))
        offset += 4

    struct.pack_into(">B", buf, offset, dtype_code)
    offset += 1

    buf[offset : offset + len(tensor_bytes)] = tensor_bytes
    sock.sendall(buf)


def send_tensor_timed(
    sock: socket.socket,
    tensor: torch.Tensor,
    draft_tokens: Optional[List[int]] = None,
    round_id: int = 0,
    parent_round_id: int = 0,
) -> dict:
    """
    Send a length-prefixed tensor with speculative drafts, round_id, and timing metrics.
    """
    t_start = time.perf_counter()
    if tensor.is_cuda:
        t_sync_0 = time.perf_counter()
        torch.cuda.synchronize(tensor.device)
        t_sync_1 = time.perf_counter()

        t_g2c_0 = time.perf_counter()
        cpu_tensor = tensor.contiguous().cpu()
        t_g2c_1 = time.perf_counter()
    else:
        t_sync_0 = t_sync_1 = t_g2c_0 = t_g2c_1 = time.perf_counter()
        cpu_tensor = tensor.contiguous()

    t_ser_0 = time.perf_counter()
    tensor_bytes = cpu_tensor.view(torch.uint8).numpy().tobytes()
    dtype_code = DTYPE_REVERSE.get(cpu_tensor.dtype, 0)
    shape = list(cpu_tensor.shape)
    ndim = len(shape)
    drafts = draft_tokens or []
    num_drafts = len(drafts)
    payload_len = 1 + 4 + 4 + 2 + (8 * num_drafts) + 1 + (4 * ndim) + 1 + len(tensor_bytes)

    buf = bytearray(LENGTH_PREFIX_SIZE + payload_len)
    struct.pack_into(LENGTH_PREFIX_FMT, buf, 0, payload_len)
    offset = LENGTH_PREFIX_SIZE
    struct.pack_into(">BIIH", buf, offset, MSG_TENSOR, int(round_id), int(parent_round_id), num_drafts)
    offset += 1 + 4 + 4 + 2
    for d in drafts:
        struct.pack_into(">q", buf, offset, int(d))
        offset += 8
    struct.pack_into(">B", buf, offset, ndim)
    offset += 1
    for dim in shape:
        struct.pack_into(">I", buf, offset, int(dim))
        offset += 4
    struct.pack_into(">B", buf, offset, dtype_code)
    offset += 1
    buf[offset : offset + len(tensor_bytes)] = tensor_bytes
    t_ser_1 = time.perf_counter()

    t_send_0 = time.perf_counter()
    sock.sendall(buf)
    t_send_1 = time.perf_counter()

    return {
        "cuda_sync_ms": (t_sync_1 - t_sync_0) * 1000.0,
        "gpu_to_cpu_ms": (t_g2c_1 - t_g2c_0) * 1000.0,
        "serialize_ms": (t_ser_1 - t_ser_0) * 1000.0,
        "tcp_send_ms": (t_send_1 - t_send_0) * 1000.0,
        "total_send_ms": (t_send_1 - t_start) * 1000.0,
        "round_id": round_id,
        "parent_round_id": parent_round_id,
    }


def recv_tensor(
    sock: socket.socket,
    shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Receive a length-prefixed tensor and any accompanying draft tokens.
    """
    tensor, drafts, _ = recv_tensor_timed(sock, shape, dtype)
    return tensor, drafts


def recv_tensor_timed(
    sock: socket.socket,
    shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, List[int], dict]:
    """
    Receive a length-prefixed tensor while measuring exact TCP recv and deserialization times.
    """
    t_start = time.perf_counter()
    len_bytes = recvall(sock, LENGTH_PREFIX_SIZE)
    payload_len = struct.unpack(LENGTH_PREFIX_FMT, len_bytes)[0]
    if payload_len > 500_000_000 or payload_len <= 0:
        raise ConnectionError(
            f"Invalid tensor frame length ({payload_len} bytes). TCP stream desynchronized."
        )
    payload = recvall(sock, payload_len)
    t_recv_1 = time.perf_counter()

    t_deser_0 = time.perf_counter()
    offset = 0
    msg_type = struct.unpack_from(">B", payload, offset)[0]
    offset += 1
    if msg_type != MSG_TENSOR:
        raise ValueError(f"Expected MSG_TENSOR (0x01), got msg_type={msg_type}")

    round_id = 0
    parent_round_id = 0
    # Check if payload has round_id headers (1 + 4 + 4 + 2 = 11 header bytes before drafts)
    if len(payload) >= 11:
        round_id, parent_round_id, num_drafts = struct.unpack_from(">IIH", payload, offset)
        offset += 4 + 4 + 2
    else:
        num_drafts = struct.unpack_from(">H", payload, offset)[0]
        offset += 2

    draft_tokens: List[int] = []
    for _ in range(num_drafts):
        d = struct.unpack_from(">q", payload, offset)[0]
        draft_tokens.append(d)
        offset += 8

    ndim = struct.unpack_from(">B", payload, offset)[0]
    offset += 1
    shape_dims = []
    for _ in range(ndim):
        d = struct.unpack_from(">I", payload, offset)[0]
        shape_dims.append(d)
        offset += 4

    dtype_code = struct.unpack_from(">B", payload, offset)[0]
    offset += 1
    torch_dtype = dtype or DTYPE_MAP.get(dtype_code, torch.float16)

    tensor_data = payload[offset:]
    tensor = torch.frombuffer(tensor_data, dtype=torch_dtype).reshape(shape_dims).clone()
    t_deser_1 = time.perf_counter()

    timings = {
        "tcp_recv_ms": (t_recv_1 - t_start) * 1000.0,
        "deserialize_ms": (t_deser_1 - t_deser_0) * 1000.0,
        "total_recv_ms": (t_deser_1 - t_start) * 1000.0,
        "round_id": round_id,
        "parent_round_id": parent_round_id,
    }
    return tensor, draft_tokens, timings


def send_token(
    sock: socket.socket,
    token_id: int,
    accepted_count: int = 1,
    is_eos: bool = False,
    compute_ms: float = 0.0,
    round_id: int = 0,
    is_stale_discard: bool = False,
) -> None:
    """
    Send a token response (Node 1 -> Node 0) with 8-byte length prefix framing.
    """
    payload_len = 1 + 8 + 4 + 1 + 4 + 4 + 1
    buf = bytearray(LENGTH_PREFIX_SIZE + payload_len)
    struct.pack_into(LENGTH_PREFIX_FMT, buf, 0, payload_len)
    struct.pack_into(
        ">BqIBfIB",
        buf,
        LENGTH_PREFIX_SIZE,
        MSG_TOKEN,
        int(token_id),
        int(accepted_count),
        1 if is_eos else 0,
        float(compute_ms),
        int(round_id),
        1 if is_stale_discard else 0,
    )
    sock.sendall(buf)


def send_token_timed(
    sock: socket.socket,
    token_id: int,
    accepted_count: int = 1,
    is_eos: bool = False,
    compute_ms: float = 0.0,
    round_id: int = 0,
    is_stale_discard: bool = False,
) -> dict:
    """
    Send a token response (Node 1 -> Node 0) with timing measurement.
    """
    t_start = time.perf_counter()
    payload_len = 1 + 8 + 4 + 1 + 4 + 4 + 1
    buf = bytearray(LENGTH_PREFIX_SIZE + payload_len)
    struct.pack_into(LENGTH_PREFIX_FMT, buf, 0, payload_len)
    struct.pack_into(
        ">BqIBfIB",
        buf,
        LENGTH_PREFIX_SIZE,
        MSG_TOKEN,
        int(token_id),
        int(accepted_count),
        1 if is_eos else 0,
        float(compute_ms),
        int(round_id),
        1 if is_stale_discard else 0,
    )
    t_ser_end = time.perf_counter()
    sock.sendall(buf)
    t_send_end = time.perf_counter()
    return {
        "serialize_ms": (t_ser_end - t_start) * 1000.0,
        "tcp_send_ms": (t_send_end - t_ser_end) * 1000.0,
        "total_ms": (t_send_end - t_start) * 1000.0,
        "round_id": round_id,
        "is_stale_discard": is_stale_discard,
    }


def recv_token(sock: socket.socket) -> Tuple[int, int, bool]:
    """
    Receive a token response from peer.
    """
    token_id, accepted_count, is_eos, _ = recv_token_timed(sock)
    return token_id, accepted_count, is_eos


def recv_token_timed(sock: socket.socket) -> Tuple[int, int, bool, dict]:
    """
    Receive a token response from peer with timing measurement.
    """
    t_start = time.perf_counter()
    len_bytes = recvall(sock, LENGTH_PREFIX_SIZE)
    payload_len = struct.unpack(LENGTH_PREFIX_FMT, len_bytes)[0]
    if payload_len > 100_000 or payload_len <= 0:
        raise ConnectionError(
            f"Invalid token response frame length ({payload_len} bytes). TCP stream desynchronized."
        )
    payload = recvall(sock, payload_len)
    t_recv_end = time.perf_counter()

    t_deser_0 = time.perf_counter()
    msg_type = struct.unpack_from(">B", payload, 0)[0]
    if msg_type != MSG_TOKEN:
        raise ValueError(f"Expected MSG_TOKEN (0x02), got msg_type={msg_type}")

    compute_ms = 0.0
    round_id = 0
    is_stale_discard = False
    if len(payload) >= 23:
        token_id, accepted_count, is_eos_val, compute_ms, round_id, stale_val = struct.unpack_from(">qIBfIB", payload, 1)
        is_stale_discard = bool(stale_val)
    elif len(payload) >= 18:
        token_id, accepted_count, is_eos_val, compute_ms = struct.unpack_from(">qIBf", payload, 1)
    else:
        token_id, accepted_count, is_eos_val = struct.unpack_from(">qIB", payload, 1)
    t_deser_1 = time.perf_counter()

    raw_recv_ms = (t_recv_end - t_start) * 1000.0
    net_rtt = max(0.0, raw_recv_ms - compute_ms) if compute_ms > 0 else raw_recv_ms
    timings = {
        "tcp_recv_ms": raw_recv_ms,
        "deserialize_ms": (t_deser_1 - t_deser_0) * 1000.0,
        "node1_compute_ms": compute_ms,
        "network_rtt_ms": net_rtt,
        "one_way_flight_ms": net_rtt / 2.0,
        "round_id": round_id,
        "is_stale_discard": is_stale_discard,
        "total_recv_ms": (t_deser_1 - t_start) * 1000.0,
    }
    return token_id, accepted_count, bool(is_eos_val), timings
