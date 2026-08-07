"""
Async TCP connection management for node-to-node and orchestrator-to-node communication.

Provides:
    - NodeServer: async TCP server for pipeline nodes
    - NodeClient: async TCP client for orchestrator → node or node → node connections
"""

import asyncio
import logging
import socket
from typing import Callable, Awaitable, Optional

from shardflow.transport.protocol import (
    TensorMessage,
    send_message,
    recv_message,
)

logger = logging.getLogger(__name__)


class NodeServer:
    """
    Async TCP server for a pipeline node.

    Listens for incoming connections and dispatches received messages
    to a handler callback.
    """

    def __init__(
        self,
        host: str,
        port: int,
        handler: Callable[[TensorMessage], Awaitable[Optional[TensorMessage]]],
        recv_timeout: float = 30.0,
    ):
        """
        Args:
            host: bind address
            port: bind port
            handler: async callback that processes a TensorMessage and optionally
                     returns a response TensorMessage (e.g., logits for the final node)
            recv_timeout: seconds to wait for each message before timing out
        """
        self.host = host
        self.port = port
        self.handler = handler
        self.recv_timeout = recv_timeout
        self._server: Optional[asyncio.Server] = None

    async def start(self) -> None:
        """Start the TCP server."""
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
            reuse_address=True,
        )
        addr = self._server.sockets[0].getsockname()
        logger.info("Node server listening on %s:%d", addr[0], addr[1])

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single TCP connection — process messages in a loop."""
        peer = writer.get_extra_info("peername")
        logger.info("New connection from %s", peer)

        sock = writer.get_extra_info("socket")
        if sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        try:
            while True:
                try:
                    msg = await recv_message(reader, timeout=self.recv_timeout)
                except asyncio.TimeoutError:
                    continue
                except (asyncio.IncompleteReadError, ConnectionError):
                    logger.info("Connection closed by %s", peer)
                    break

                response = await self.handler(msg)
                if response is not None:
                    await send_message(writer, response)

        except Exception:
            logger.exception("Error handling connection from %s", peer)
        finally:
            writer.close()
            await writer.wait_closed()
            logger.info("Connection to %s closed", peer)

    async def stop(self) -> None:
        """Stop the TCP server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("Node server stopped")

    async def serve_forever(self) -> None:
        """Start and run until cancelled."""
        await self.start()
        await self._server.serve_forever()


class NodeClient:
    """
    Async TCP client for connecting to a pipeline node.
    """

    def __init__(
        self,
        host: str,
        port: int,
        send_timeout: float = 15.0,
        recv_timeout: float = 60.0,
    ):
        self.host = host
        self.port = port
        self.send_timeout = send_timeout
        self.recv_timeout = recv_timeout
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False

    async def connect(self, max_retries: int = 15, retry_delay: float = 2.0) -> None:
        """Establish TCP connection to the node, with retry for bootstrapping nodes."""
        logger.info("Connecting to %s:%d ...", self.host, self.port)
        use_ssl = True if self.port == 443 else None
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port, ssl=use_ssl),
                    timeout=self.send_timeout,
                )
                sock = self._writer.get_extra_info("socket")
                if sock:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._connected = True
                logger.info("Connected to %s:%d (ssl=%s)", self.host, self.port, bool(use_ssl))
                return
            except (OSError, asyncio.TimeoutError) as e:
                last_err = e
                if attempt < max_retries:
                    logger.debug(
                        "Connect to %s:%d failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        self.host, self.port, attempt, max_retries, e, retry_delay
                    )
                    await asyncio.sleep(retry_delay)
        raise ConnectionError(
            f"Failed to connect to {self.host}:{self.port} after {max_retries} attempts: {last_err}"
        )

    async def send(self, msg: TensorMessage) -> None:
        """Send a message to the connected node."""
        if not self.is_connected:
            raise ConnectionError("Not connected. Call connect() first.")
        try:
            await send_message(self._writer, msg)
        except Exception:
            self._connected = False
            raise

    async def recv(self) -> TensorMessage:
        """Receive a message from the connected node."""
        if not self.is_connected:
            raise ConnectionError("Not connected. Call connect() first.")
        try:
            return await recv_message(self._reader, timeout=self.recv_timeout)
        except Exception:
            self._connected = False
            raise

    async def send_recv(self, msg: TensorMessage, timeout: Optional[float] = None) -> TensorMessage:
        """Send a message and wait for a response with optional timeout override."""
        await self.send(msg)
        t = timeout if timeout is not None else self.recv_timeout
        try:
            return await recv_message(self._reader, timeout=t)
        except Exception:
            self._connected = False
            raise

    async def close(self) -> None:
        """Close the connection."""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._connected = False
        logger.info("Disconnected from %s:%d", self.host, self.port)

    @property
    def is_connected(self) -> bool:
        if not self._connected:
            return False
        if self._writer is None or self._writer.is_closing():
            self._connected = False
            return False
        return True
