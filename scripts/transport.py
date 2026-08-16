"""
ShardFlow v2 — Standalone TCP Relay Transport.

Can be run standalone or imported directly in Kaggle/Colab notebooks.
Connects to the Rust relay on AWS EC2 Ohio and pipes raw tensors/tokens bidirectionally.
"""

import socket
import struct
import logging
from typing import Optional, Tuple, List, Union
import torch

logger = logging.getLogger("transport")

RELAY_HOST = "3.23.174.207"
RELAY_PORT = 9500
AUTH_BYTE = bytes([0xAD])
SOCKET_DATA_TIMEOUT = 30.0

LENGTH_PREFIX_FMT = ">Q"
LENGTH_PREFIX_SIZE = 8

MSG_TENSOR = 0x01
MSG_TOKEN = 0x02
MSG_CONTROL = 0x03

DTYPE_MAP = {
    0: torch.float16,
    1: torch.bfloat16,
    2: torch.float32,
    3: torch.int64,
    4: torch.int32,
}
DTYPE_REVERSE = {v: k for k, v in DTYPE_MAP.items()}


def configure_socket(sock: socket.socket, data_timeout: float = SOCKET_DATA_TIMEOUT) -> None:
    """Configure TCP_NODELAY, SO_KEEPALIVE, large socket buffers, and data timeout."""
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except Exception:
        pass
    if hasattr(socket, "TCP_QUICKACK"):
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
        except Exception:
            pass
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    except Exception:
        pass
    sock.settimeout(data_timeout)


def recvall(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from socket with timeout error protection."""
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    try:
        while pos < n:
            nbytes = sock.recv_into(view[pos:], n - pos)
            if nbytes == 0:
                raise ConnectionError(f"Socket connection closed unexpectedly after reading {pos}/{n} bytes.")
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
    1. Create TCP socket
    2. Set TCP_NODELAY = 1, SO_KEEPALIVE = 1, buffer sizes
    3. Connect to relay
    4. Send auth byte immediately
    5. Configure data timeout (30s) and return socket
    """
    print(f"Connecting to relay at {host}:{port}...", flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(connect_timeout)

    try:
        sock.connect((host, port))
        sock.sendall(auth_byte)
        configure_socket(sock, data_timeout=data_timeout)
        print(f"✅ Connected to relay at {host}:{port} (auth byte sent, timeout={data_timeout}s)", flush=True)
        return sock
    except Exception as e:
        sock.close()
        raise ConnectionError(f"Failed to connect to relay at {host}:{port}: {e}") from e


def handshake(sock: socket.socket, timeout: float = 60.0) -> None:
    """
    Symmetric handshake to confirm both nodes are paired and ready.
    Both nodes send b"READY", both receive b"READY".
    """
    orig_timeout = sock.gettimeout()
    sock.settimeout(timeout)
    try:
        print("Sending READY handshake to peer through relay...", flush=True)
        sock.sendall(b"READY")
        ack = recvall(sock, 5)
        assert ack == b"READY", f"Handshake failed: expected b'READY', received {ack!r}"
        print("✅ Handshake successful: Both nodes are READY!", flush=True)
    finally:
        sock.settimeout(orig_timeout)


def send_tensor(
    sock: socket.socket,
    tensor: torch.Tensor,
    draft_tokens: Optional[List[int]] = None,
) -> None:
    """
    Serialize tensor: torch.float16/bfloat16, contiguous, .numpy().tobytes()
    Frame it: 8-byte big-endian length prefix + payload
    sock.sendall(frame)
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

    payload_len = 1 + 2 + (8 * num_drafts) + 1 + (4 * ndim) + 1 + len(tensor_bytes)

    buf = bytearray(LENGTH_PREFIX_SIZE + payload_len)
    struct.pack_into(LENGTH_PREFIX_FMT, buf, 0, payload_len)
    offset = LENGTH_PREFIX_SIZE

    struct.pack_into(">B", buf, offset, MSG_TENSOR)
    offset += 1

    struct.pack_into(">H", buf, offset, num_drafts)
    offset += 2
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

    buf[offset:offset + len(tensor_bytes)] = tensor_bytes
    sock.sendall(buf)


def recv_tensor(
    sock: socket.socket,
    shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[int]]]:
    """
    Read 8-byte length prefix
    Read exactly N bytes
    Reconstruct: torch.frombuffer(buf, dtype=dtype).reshape(shape)
    """
    len_bytes = recvall(sock, LENGTH_PREFIX_SIZE)
    payload_len = struct.unpack(LENGTH_PREFIX_FMT, len_bytes)[0]
    payload = recvall(sock, payload_len)
    offset = 0

    msg_type = struct.unpack_from(">B", payload, offset)[0]
    offset += 1

    if msg_type != MSG_TENSOR:
        raise ValueError(f"Expected MSG_TENSOR (0x01), got msg_type={msg_type}")

    num_drafts = struct.unpack_from(">H", payload, offset)[0]
    offset += 2
    draft_tokens: List[int] = []
    for _ in range(num_drafts):
        d = struct.unpack_from(">q", payload, offset)[0]
        draft_tokens.append(d)
        offset += 8

    ndim = struct.unpack_from(">B", payload, offset)[0]
    offset += 1
    decoded_shape = []
    for _ in range(ndim):
        dim = struct.unpack_from(">I", payload, offset)[0]
        decoded_shape.append(dim)
        offset += 4
    dtype_code = struct.unpack_from(">B", payload, offset)[0]
    offset += 1

    final_shape = shape if shape is not None else tuple(decoded_shape)
    final_dtype = dtype if dtype is not None else DTYPE_MAP.get(dtype_code, torch.float16)

    numel = 1
    for dim in final_shape:
        numel *= dim

    tensor_data = bytearray(payload[offset:])
    tensor = torch.frombuffer(tensor_data, dtype=final_dtype, count=numel).reshape(final_shape)

    if num_drafts > 0:
        return tensor, draft_tokens
    return tensor


def send_token(
    sock: socket.socket,
    token_id: int,
    accepted_count: int = 1,
    is_eos: bool = False,
) -> None:
    """Send generated token ID back to Node 0."""
    payload_len = 1 + 8 + 4 + 1
    buf = bytearray(LENGTH_PREFIX_SIZE + payload_len)
    struct.pack_into(LENGTH_PREFIX_FMT, buf, 0, payload_len)
    struct.pack_into(
        ">BqIB",
        buf,
        LENGTH_PREFIX_SIZE,
        MSG_TOKEN,
        int(token_id),
        int(accepted_count),
        1 if is_eos else 0,
    )
    sock.sendall(buf)


def recv_token(sock: socket.socket) -> Tuple[int, int, bool]:
    """Receive token ID response."""
    len_bytes = recvall(sock, LENGTH_PREFIX_SIZE)
    payload_len = struct.unpack(LENGTH_PREFIX_FMT, len_bytes)[0]
    payload = recvall(sock, payload_len)

    msg_type = struct.unpack_from(">B", payload, 0)[0]
    if msg_type != MSG_TOKEN:
        raise ValueError(f"Expected MSG_TOKEN (0x02), got msg_type={msg_type}")

    token_id, accepted_count, is_eos_val = struct.unpack_from(">qIB", payload, 1)
    return token_id, accepted_count, bool(is_eos_val)
