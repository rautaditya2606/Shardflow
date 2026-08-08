"""
Integration tests for failure resilience, socket timeouts, and hung connection cleanup.
"""

import asyncio
import pytest
from shardflow.transport.connection import NodeClient, NodeServer
from shardflow.transport.protocol import MessageType, TensorMessage


@pytest.mark.asyncio
async def test_node_client_closes_socket_on_hung_timeout(free_port: int):
    """Verify that if a remote server hangs, NodeClient times out and cleanly closes the socket."""
    async def hung_handler(msg: TensorMessage) -> None:
        await asyncio.sleep(5.0)

    server = NodeServer(host="127.0.0.1", port=free_port, handler=hung_handler)
    await server.start()

    client = NodeClient(host="127.0.0.1", port=free_port, send_timeout=2.0, recv_timeout=0.3)
    await client.connect()

    msg = TensorMessage(msg_type=MessageType.CLEAR, session_id="timeout-sess")

    with pytest.raises(asyncio.TimeoutError):
        await client.send_recv(msg, timeout=0.3)

    assert client.is_connected is False

    await client.close()
    await server.stop()
