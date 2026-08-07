"""
Shared helpers for registry HTTP interaction.

Used by node runners and orchestrator to poll assignments and avoid
blocking the asyncio event loop with synchronous requests.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


def poll_for_assignment(
    registry_url: str,
    node_id: str,
    *,
    timeout: float = 120.0,
    interval: float = 3.0,
) -> dict[str, Any]:
    """
    Poll /assignment/{node_id} until the cluster partition is final.

    Returns the assignment dict on success. Raises RuntimeError on timeout.
    """
    url = f"{registry_url.rstrip('/')}/assignment/{node_id}"
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "assigned" and data.get("cluster_ready", True):
                    return data
            elif resp.status_code == 202:
                pending = resp.json()
                logger.info(
                    "Assignment pending for %s: %s",
                    node_id,
                    pending.get("message", "waiting"),
                )
        except Exception as exc:
            logger.debug("Assignment poll error for %s: %s", node_id, exc)

        time.sleep(interval)

    raise RuntimeError(
        f"Timed out after {timeout:.0f}s waiting for final layer assignment for {node_id}"
    )


async def async_get_topology(registry_url: str, timeout: float = 5.0) -> list[tuple[str, int]]:
    """Fetch active node addresses without blocking the event loop."""

    def _fetch() -> list[tuple[str, int]]:
        resp = requests.get(f"{registry_url.rstrip('/')}/topology", timeout=timeout)
        resp.raise_for_status()
        nodes = resp.json().get("nodes", [])
        return [(n["addr"], n["port"]) for n in nodes if n.get("is_active", True)]

    return await asyncio.to_thread(_fetch)
