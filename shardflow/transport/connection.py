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


def _configure_socket_options(sock: Optional[socket.socket]) -> None:
    """Apply high-performance and aggressive keepalive options to TCP socket."""
    if not sock:
        return
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except Exception:
        pass
    if hasattr(socket, "TCP_KEEPIDLE"):
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)
        except Exception:
            pass
    if hasattr(socket, "TCP_KEEPINTVL"):
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
        except Exception:
            pass
    if hasattr(socket, "TCP_KEEPCNT"):
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
        except Exception:
            pass
    if hasattr(socket, "TCP_QUICKACK"):
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
        except Exception:
            pass
    try:
        # ponytail: 4MB socket buffer fits full 7B prefill activations (~3.5MB) in a single TCP write
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4194304)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4194304)
    except Exception:
        pass


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
        import time
        peer = writer.get_extra_info("peername")
        logger.info("New connection from %s", peer)

        sock = writer.get_extra_info("socket")
        _configure_socket_options(sock)

        try:
            while True:
                try:
                    msg = await recv_message(reader, timeout=self.recv_timeout)
                except asyncio.TimeoutError:
                    continue
                except (asyncio.IncompleteReadError, ConnectionError):
                    logger.info("Connection closed by %s", peer)
                    break

                now_us = int(time.perf_counter() * 1_000_000)
                if msg.send_ts_us > 0:
                    hop_ms = (now_us - msg.send_ts_us) / 1000.0
                    logger.debug("Hop latency for %s from %s: %.2f ms", msg.msg_type.name, peer, hop_ms)

                response = await self.handler(msg)
                if response is not None:
                    await send_message(writer, response)

        except Exception:
            logger.exception("Error handling connection from %s", peer)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.info("Connection to %s closed", peer)

    async def stop(self) -> None:
        """Stop the TCP server."""
        if self._server:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
            logger.info("Node server stopped")

    async def serve_forever(self) -> None:
        """Start and run until cancelled."""
        await self.start()
        await self._server.serve_forever()


class StreamReceiverServer:
    """
    Lightweight TCP server running on the Gateway to receive direct token streams
    from the terminal pipeline node (Node N) with sub-millisecond dispatch.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9600, recv_timeout: float = 60.0):
        self.host = host
        self.port = port
        self.recv_timeout = recv_timeout
        self._server: Optional[asyncio.Server] = None
        self._session_queues: dict[str, asyncio.Queue[TensorMessage]] = {}
        self.bound_port: int = port

    async def start(self) -> int:
        """Start the stream receiver server and return the bound port."""
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
            reuse_address=True,
        )
        addr = self._server.sockets[0].getsockname()
        self.bound_port = addr[1]
        logger.info("Gateway StreamReceiverServer listening on %s:%d", addr[0], self.bound_port)
        return self.bound_port

    def register_session(self, session_id: str) -> asyncio.Queue[TensorMessage]:
        """Register a session queue for incoming streamed tokens."""
        q: asyncio.Queue[TensorMessage] = asyncio.Queue()
        self._session_queues[session_id] = q
        return q

    def unregister_session(self, session_id: str) -> None:
        """Unregister and cleanup a session queue."""
        self._session_queues.pop(session_id, None)

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle incoming token stream from terminal node."""
        peer = writer.get_extra_info("peername")
        sock = writer.get_extra_info("socket")
        _configure_socket_options(sock)

        try:
            while True:
                try:
                    msg = await recv_message(reader, timeout=self.recv_timeout)
                except asyncio.TimeoutError:
                    continue
                except (asyncio.IncompleteReadError, ConnectionError):
                    break

                q = self._session_queues.get(msg.session_id)
                if q is not None:
                    await q.put(msg)
                else:
                    logger.debug("Received token for unknown or finished session %s", msg.session_id)

                if msg.is_eos or (msg.finish_reason is not None and msg.finish_reason != ""):
                    # Stream complete for this session
                    break

        except Exception as e:
            logger.debug("Stream connection from %s closed: %s", peer, e)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def stop(self) -> None:
        """Stop the stream receiver server."""
        if self._server:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
            logger.info("StreamReceiverServer stopped")


_local_ips_cache: Optional[set[str]] = None


def get_local_machine_ips() -> set[str]:
    """
    Resolve and cache local interface and Tailscale IPs once on startup.
    Avoids blocking DNS or subprocess calls during mid-session reconnect loops.
    """
    global _local_ips_cache
    if _local_ips_cache is not None:
        return _local_ips_cache

    local_ips = {"127.0.0.1", "localhost", "0.0.0.0"}

    # Tier 1: Hostname & DNS resolution
    try:
        hostname = socket.gethostname()
        local_ips.add(socket.gethostbyname(hostname))
        for addr_info in socket.getaddrinfo(hostname, None):
            local_ips.add(addr_info[4][0])
    except Exception as e:
        logger.debug("DNS interface resolution error: %s", e)

    # Tier 2: Kernel outbound route socket inspection (non-blocking)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ips.add(s.getsockname()[0])
    except Exception as e:
        logger.debug("Outbound route socket inspection error: %s", e)

    # Tier 3: Environment variable fallback
    env_ips = os.getenv("SHARDFLOW_LOCAL_IPS", "").strip()
    if env_ips:
        for ip in env_ips.split(","):
            ip = ip.strip()
            if ip:
                local_ips.add(ip)

    # Tier 4: Tailscale IP detection
    try:
        import shutil
        import subprocess
        tailscale_bin = shutil.which("tailscale")
        if tailscale_bin:
            res = subprocess.run([tailscale_bin, "ip", "-4"], capture_output=True, text=True, timeout=1.0)
            if res.returncode == 0:
                ts_ip = res.stdout.strip()
                if ts_ip:
                    local_ips.add(ts_ip)
    except Exception as e:
        logger.debug("Tailscale interface inspection error: %s", e)

    _local_ips_cache = local_ips
    return _local_ips_cache


async def _open_socks5_connection(
    target_host: str,
    target_port: int,
    proxy_host: str = "127.0.0.1",
    proxy_port: int = 1055,
    timeout: float = 5.0,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """
    Open a TCP stream through the local Tailscale SOCKS5 proxy.
    Used as a transparent fallback when kernel TUN mode is unavailable (userspace mode).
    With kernel TUN mode, asyncio.open_connection() to 100.x.x.x routes through WireGuard
    directly and this function is never called.
    """
    import socket
    import struct

    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(proxy_host, proxy_port),
        timeout=timeout,
    )
    writer.write(b"\x05\x01\x00")
    await writer.drain()
    res = await reader.readexactly(2)
    if res != b"\x05\x00":
        writer.close()
        raise ConnectionError(f"SOCKS5 auth failed: {res}")

    try:
        ip_bytes = socket.inet_aton(target_host)
        req = b"\x05\x01\x00\x01" + ip_bytes + struct.pack("!H", target_port)
    except Exception:
        host_bytes = target_host.encode("ascii")
        req = b"\x05\x01\x00\x03" + struct.pack("!B", len(host_bytes)) + host_bytes + struct.pack("!H", target_port)

    writer.write(req)
    await writer.drain()
    resp = await reader.readexactly(4)
    if resp[1] != 0:
        writer.close()
        raise ConnectionError(f"SOCKS5 connect failed (code={resp[1]})")

    atyp = resp[3]
    if atyp == 1:
        await reader.readexactly(4 + 2)
    elif atyp == 3:
        dlen = (await reader.readexactly(1))[0]
        await reader.readexactly(dlen + 2)
    elif atyp == 4:
        await reader.readexactly(16 + 2)

    return reader, writer


class NodeClient:
    """
    Async TCP client for connecting to a pipeline node with zero-latency transport optimizations
    and automatic transparent reconnection across idle timeouts and transient disconnects.
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
        self.reconnect_count: int = 0
        self.last_hop_latency_ms: float = 0.0
        self._lock = asyncio.Lock()

    async def connect(self, max_retries: int = 15, retry_delay: float = 1.0) -> None:
        """Establish TCP connection to the node, with retry for bootstrapping nodes."""
        # Same-host optimization: if target host matches local interface / Tailscale IP on this machine, route via 127.0.0.1
        connect_host = self.host
        try:
            local_ips = get_local_machine_ips()
            if self.host in local_ips:
                connect_host = "127.0.0.1"
                logger.info("Same-host routing detected for %s — connecting via local loopback 127.0.0.1:%d", self.host, self.port)
        except Exception as e:
            logger.debug("Same-host detection exception (falling back to %s): %s", self.host, e)
            connect_host = self.host

        logger.info("Connecting to %s:%d ...", connect_host, self.port)
        use_ssl = True if self.port == 443 else None
        last_err = None
        is_tailscale_ip = connect_host.startswith("100.") or ".ts.net" in connect_host
        for attempt in range(1, max_retries + 1):
            try:
                # Direct TCP — with kernel TUN, 100.x.x.x routes through WireGuard natively
                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_connection(connect_host, self.port, ssl=use_ssl),
                    timeout=self.send_timeout,
                )
                sock = self._writer.get_extra_info("socket")
                _configure_socket_options(sock)
                self._connected = True
                if attempt > 1:
                    self.reconnect_count += 1
                logger.info("Connected to %s:%d (ssl=%s, reconnects=%d)", self.host, self.port, bool(use_ssl), self.reconnect_count)
                return
            except (OSError, asyncio.TimeoutError) as e:
                # SOCKS5 fallback: only for Tailscale IPs when kernel TUN is unavailable (userspace mode)
                if is_tailscale_ip:
                    try:
                        self._reader, self._writer = await _open_socks5_connection(
                            target_host=connect_host,
                            target_port=self.port,
                            timeout=self.send_timeout,
                        )
                        sock = self._writer.get_extra_info("socket")
                        _configure_socket_options(sock)
                        self._connected = True
                        if attempt > 1:
                            self.reconnect_count += 1
                        logger.info("Connected to %s:%d via SOCKS5 (Tailscale userspace mode)", self.host, self.port)
                        return
                    except Exception as s_err:
                        last_err = s_err
                else:
                    last_err = e

                if attempt < max_retries:
                    logger.debug(
                        "Connect to %s:%d failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        self.host, self.port, attempt, max_retries, last_err, retry_delay
                    )
                    await asyncio.sleep(retry_delay)
        raise ConnectionError(
            f"Failed to connect to {self.host}:{self.port} after {max_retries} attempts: {last_err}"
        )

    async def ensure_connected(self) -> None:
        """Ensure connection is established; reconnects if socket dropped or closed."""
        if not self.is_connected:
            async with self._lock:
                if not self.is_connected:
                    await self.connect(max_retries=5, retry_delay=0.5)

    async def send(self, msg: TensorMessage) -> None:
        """Send a message to the connected node with transparent auto-reconnect."""
        async with self._lock:
            await self.ensure_connected()
            try:
                await send_message(self._writer, msg)
            except Exception as e:
                logger.warning("Send to %s:%d failed (%s) — reconnecting...", self.host, self.port, e)
                await self.close()
                await self.connect(max_retries=5, retry_delay=0.5)
                await send_message(self._writer, msg)

    async def recv(self) -> TensorMessage:
        """Receive a message from the connected node and record hop latency."""
        import time
        async with self._lock:
            if not self.is_connected:
                raise ConnectionError("Not connected. Call connect() first.")
            try:
                msg = await recv_message(self._reader, timeout=self.recv_timeout)
                now_us = int(time.perf_counter() * 1_000_000)
                if msg.send_ts_us > 0:
                    self.last_hop_latency_ms = (now_us - msg.send_ts_us) / 1000.0
                return msg
            except Exception:
                self._connected = False
                raise

    async def send_recv(self, msg: TensorMessage, timeout: Optional[float] = None) -> TensorMessage:
        """Send a message and wait for response with automatic transparent retry on dead sockets."""
        import time
        async with self._lock:
            t = timeout if timeout is not None else self.recv_timeout
            try:
                await self.ensure_connected()
                await send_message(self._writer, msg)
                resp = await recv_message(self._reader, timeout=t)
                now_us = int(time.perf_counter() * 1_000_000)
                if resp.send_ts_us > 0:
                    self.last_hop_latency_ms = (now_us - resp.send_ts_us) / 1000.0
                return resp
            except Exception as e:
                logger.warning("send_recv to %s:%d failed (%s) — reconnecting and retrying...", self.host, self.port, e)
                await self.close()
                try:
                    await self.connect(max_retries=5, retry_delay=0.5)
                    await send_message(self._writer, msg)
                    resp = await recv_message(self._reader, timeout=t)
                    now_us = int(time.perf_counter() * 1_000_000)
                    if resp.send_ts_us > 0:
                        self.last_hop_latency_ms = (now_us - resp.send_ts_us) / 1000.0
                    return resp
                except Exception:
                    self._connected = False
                    await self.close()
                    raise

    async def close(self) -> None:
        """Close the connection cleanly."""
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
