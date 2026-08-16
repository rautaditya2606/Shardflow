"""Transport layer — length-prefix TCP framing for tensor transfer and relay connection."""

from shardflow.transport.relay import (
    RELAY_HOST,
    RELAY_PORT,
    AUTH_BYTE,
    connect_to_relay,
    handshake,
    send_tensor,
    send_tensor_timed,
    recv_tensor,
    recv_tensor_timed,
    send_token,
    send_token_timed,
    recv_token,
    recv_token_timed,
    recvall,
)

__all__ = [
    "RELAY_HOST",
    "RELAY_PORT",
    "AUTH_BYTE",
    "connect_to_relay",
    "handshake",
    "send_tensor",
    "send_tensor_timed",
    "recv_tensor",
    "recv_tensor_timed",
    "send_token",
    "send_token_timed",
    "recv_token",
    "recv_token_timed",
    "recvall",
]
