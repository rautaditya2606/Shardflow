"""Transport layer — length-prefix TCP framing for tensor transfer and relay connection."""

from shardflow.transport.relay import (
    RELAY_HOST,
    RELAY_PORT,
    AUTH_BYTE,
    connect_to_relay,
    handshake,
    send_tensor,
    recv_tensor,
    send_token,
    recv_token,
    recvall,
)

__all__ = [
    "RELAY_HOST",
    "RELAY_PORT",
    "AUTH_BYTE",
    "connect_to_relay",
    "handshake",
    "send_tensor",
    "recv_tensor",
    "send_token",
    "recv_token",
    "recvall",
]
