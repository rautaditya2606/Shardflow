"""
Integration tests for NodeServer and NodeClient TCP transport, socket buffers, and connection reuse.
"""

import asyncio
import socket
import pytest
import torch

from shardflow.transport.connection import NodeServer, NodeClient
from shardflow.transport.protocol import MessageType, TensorMessage


@pytest.mark.asyncio
async def test_tcp_server_client_roundtrip_with_nodelay(free_port: int):
    """Verify bidirectional message exchange over localhost with TCP_NODELAY and socket buffer sizing."""
    received_messages = []

    async def echo_handler(msg: TensorMessage) -> TensorMessage:
        received_messages.append(msg)
        return TensorMessage(
            msg_type=MessageType.TOKEN_ID,
            session_id=msg.session_id,
            token_id=12345,
        )

    server = NodeServer(host="127.0.0.1", port=free_port, handler=echo_handler)
    await server.start()

    client = NodeClient(host="127.0.0.1", port=free_port)
    await client.connect()

    assert client.is_connected is True
    assert client.reconnect_count == 0

    # Explicitly verify TCP_NODELAY is active on the OS socket
    client_sock = client._writer.get_extra_info("socket")
    assert client_sock is not None
    assert client_sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) == 1
    assert client_sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF) >= 131072
    assert client_sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF) >= 131072

    # Send tensor activation
    msg = TensorMessage(
        msg_type=MessageType.ACTIVATION,
        session_id="transport-test-sess",
        tensor=torch.randn(1, 1, 32, dtype=torch.float16),
    )

    response = await client.send_recv(msg, timeout=5.0)

    assert len(received_messages) == 1
    assert response.msg_type == MessageType.TOKEN_ID
    assert response.token_id == 12345
    assert client.last_hop_latency_ms >= 0.0

    await client.close()
    await server.stop()
