"""
Async HTTP transport for node-to-node communication across WAN / Cloudflare tunnels.

Provides:
    - HTTPNodeClient: async HTTP client implementing send_recv() / send() via aiohttp
    - HTTPNodeServer: async aiohttp HTTP server wrapping node _handle_message()
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Awaitable
import aiohttp
from aiohttp import web

from shardflow.transport.protocol import (
    TensorMessage,
    encode_message,
    decode_message,
    LENGTH_PREFIX_SIZE,
)

logger = logging.getLogger(__name__)


class HTTPNodeClient:
    """
    Async HTTP client for connecting to a remote pipeline node via an HTTP/HTTPS endpoint
    (e.g., Cloudflare Quick Tunnel https://*.trycloudflare.com).

    Reuses a persistent aiohttp.ClientSession for HTTP keep-alive connection reuse (~43ms steady-state RTT).
    Includes automatic single-retry logic for transient 5xx or network hiccups.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self.last_hop_latency_ms: float = 0.0

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or initialize the persistent aiohttp ClientSession."""
        if self._session is None or self._session.closed:
            async with self._lock:
                if self._session is None or self._session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=10,
                        keepalive_timeout=60.0,
                        enable_cleanup_closed=True,
                    )
                    self._session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    )
        return self._session

    @property
    def is_connected(self) -> bool:
        """Return True if target URL is configured and session is not closed."""
        if self._session is not None and self._session.closed:
            return False
        return bool(self.base_url)

    async def check_health(self, timeout: float = 5.0) -> bool:
        """Ping /health on remote node to verify reachability."""
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug("Health check failed for %s: %s", self.base_url, e)
            return False

    async def send_recv(
        self,
        msg: TensorMessage,
        timeout: Optional[float] = None,
    ) -> TensorMessage:
        """
        Send a TensorMessage via HTTP POST to /activate and wait for response TensorMessage.
        Includes 1-retry on transient 5xx, timeout, or network drops.
        """
        req_timeout = timeout or self.timeout
        encoded = encode_message(msg)
        payload = encoded[LENGTH_PREFIX_SIZE:]

        url = f"{self.base_url}/activate"
        headers = {"Content-Type": "application/octet-stream"}

        last_err: Optional[Exception] = None
        for attempt in range(2):
            session = await self._get_session()
            t0 = time.perf_counter()
            try:
                async with session.post(
                    url,
                    data=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=req_timeout),
                ) as resp:
                    rtt = (time.perf_counter() - t0) * 1000.0
                    self.last_hop_latency_ms = rtt

                    if resp.status == 200:
                        body = await resp.read()
                        response_msg = decode_message(body)
                        return response_msg
                    elif resp.status >= 500 and attempt == 0:
                        err_text = await resp.text()
                        logger.warning(
                            "HTTP POST to %s returned %d (attempt %d/2) — retrying: %s",
                            url, resp.status, attempt + 1, err_text[:150],
                        )
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        err_text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status} from {url}: {err_text[:200]}")

            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                last_err = e
                if attempt == 0:
                    logger.warning(
                        "HTTP POST to %s failed with %s (attempt 1/2) — retrying in 0.5s...",
                        url, e,
                    )
                    # Force recreate session on next attempt
                    await self.close()
                    await asyncio.sleep(0.5)
                    continue
                else:
                    raise RuntimeError(f"HTTP send_recv to {url} failed after retry: {e}") from e

        raise RuntimeError(f"HTTP send_recv to {url} failed: {last_err}")

    async def send(self, msg: TensorMessage) -> None:
        """Send a fire-and-forget or non-blocking TensorMessage (e.g. CLEAR)."""
        encoded = encode_message(msg)
        payload = encoded[LENGTH_PREFIX_SIZE:]
        url = f"{self.base_url}/activate"
        headers = {"Content-Type": "application/octet-stream"}

        session = await self._get_session()
        try:
            async with session.post(
                url,
                data=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                if resp.status not in (200, 204):
                    logger.debug("HTTP send returned status %d", resp.status)
        except Exception as e:
            logger.debug("HTTP send to %s encountered error: %s", url, e)

    async def close(self) -> None:
        """Close the persistent aiohttp session."""
        async with self._lock:
            if self._session is not None and not self._session.closed:
                await self._session.close()
                self._session = None


class HTTPNodeServer:
    """
    Async aiohttp HTTP server running on a pipeline node.
    Exposes /activate (POST) and /health (GET) endpoints.
    """

    def __init__(
        self,
        host: str,
        port: int,
        handler: Callable[[TensorMessage], Awaitable[Optional[TensorMessage]]],
    ):
        self.host = host
        self.port = port
        self.handler = handler
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    async def _handle_activate(self, request: web.Request) -> web.Response:
        """Receive activation payload, process via node handler, and return response."""
        try:
            data = await request.read()
            if not data:
                return web.Response(status=400, text="Empty request body")

            msg = decode_message(data)
            response = await self.handler(msg)

            if response is not None:
                resp_encoded = encode_message(response)
                resp_payload = resp_encoded[LENGTH_PREFIX_SIZE:]
                return web.Response(
                    body=resp_payload,
                    content_type="application/octet-stream",
                )
            return web.Response(status=204)

        except Exception as e:
            logger.exception("Error processing /activate request: %s", e)
            return web.Response(status=500, text=f"Internal Error: {e}")

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.Response(text="ok")

    async def start(self) -> None:
        """Start the HTTP server."""
        app = web.Application()
        app.router.add_post("/activate", self._handle_activate)
        app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        logger.info("HTTPNodeServer listening on http://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Stop the HTTP server."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
            logger.info("HTTPNodeServer stopped")
