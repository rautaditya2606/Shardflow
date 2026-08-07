"""
Per-node KV cache store — manages cached key/value tensors by session ID.

Each node maintains its own KV cache for its layer slice. During generation:
- Prefill phase: full prompt runs through layers, K/V stored for all positions
- Decode phase: only new token runs, using cached K/V for context → O(1) per token

Cache is keyed by session_id so multiple concurrent sessions don't interfere.
"""

import asyncio
import logging
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)


from typing import Optional, Any
from transformers.cache_utils import DynamicCache, Cache


class KVCacheStore:
    """
    Per-session KV cache store for a node's layer slice.

    Stores transformers Cache / DynamicCache objects, one per active session.
    """

    def __init__(
        self,
        eviction_timeout: float = 60.0,
        max_sessions: int = 32,
    ):
        """
        Args:
            eviction_timeout: seconds of inactivity before a session's cache is evicted
            max_sessions: maximum concurrent cached sessions (VRAM protection)
        """
        self.eviction_timeout = eviction_timeout
        self.max_sessions = max_sessions

        # session_id → Cache object for this node's layer slice
        self._cache: dict[str, Cache] = {}
        # session_id → last access timestamp
        self._last_access: dict[str, float] = {}
        # Background eviction task
        self._eviction_task: Optional[asyncio.Task] = None

    def get(self, session_id: str) -> Optional[Cache]:
        """
        Get cached Cache object for a session.

        Returns:
            Cache object, or None if not cached.
        """
        if session_id in self._cache:
            self._last_access[session_id] = time.monotonic()
            return self._cache[session_id]
        return None

    def put(
        self,
        session_id: str,
        cache_obj: Cache,
    ) -> None:
        """
        Store or update Cache object for a session.

        Args:
            session_id: unique session identifier
            cache_obj: Cache object for this node's layer slice
        """
        # Evict oldest session if at capacity
        if session_id not in self._cache and len(self._cache) >= self.max_sessions:
            self._evict_oldest()

        self._cache[session_id] = cache_obj
        self._last_access[session_id] = time.monotonic()

    def evict(self, session_id: str) -> bool:
        """
        Explicitly evict a session's cache (e.g., on CLEAR message).

        Returns:
            True if the session was found and evicted.
        """
        if session_id in self._cache:
            del self._cache[session_id]
            del self._last_access[session_id]
            logger.info("Evicted KV cache for session %s", session_id)
            return True
        return False

    def _evict_oldest(self) -> None:
        """Evict the least recently used session."""
        if not self._last_access:
            return
        oldest_session = min(self._last_access, key=self._last_access.get)
        self.evict(oldest_session)
        logger.info("LRU eviction: removed session %s", oldest_session)

    def _evict_expired(self) -> int:
        """Evict all sessions that haven't been accessed within the timeout."""
        now = time.monotonic()
        expired = [
            sid for sid, last in self._last_access.items()
            if now - last > self.eviction_timeout
        ]
        for sid in expired:
            self.evict(sid)
        return len(expired)

    async def start_eviction_loop(self) -> None:
        """Start the background eviction loop."""
        self._eviction_task = asyncio.create_task(self._eviction_loop())
        logger.info(
            "KV cache eviction loop started (timeout=%ds, max_sessions=%d)",
            self.eviction_timeout, self.max_sessions,
        )

    async def _eviction_loop(self) -> None:
        """Background task that periodically evicts expired sessions."""
        while True:
            await asyncio.sleep(15)  # Check every 15 seconds
            evicted = self._evict_expired()
            if evicted > 0:
                logger.info("Evicted %d expired sessions (%d remaining)", evicted, len(self._cache))

    async def stop_eviction_loop(self) -> None:
        """Stop the background eviction loop."""
        if self._eviction_task:
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass
            self._eviction_task = None

    def clear_all(self) -> None:
        """Evict all sessions. Used during shutdown."""
        count = len(self._cache)
        self._cache.clear()
        self._last_access.clear()
        if count > 0:
            logger.info("Cleared all %d cached sessions", count)

    @property
    def active_sessions(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        """Get cache statistics."""
        total_memory = 0
        for session_id, cache_obj in self._cache.items():
            if hasattr(cache_obj, "layers") and isinstance(cache_obj.layers, (list, tuple)):
                for layer in cache_obj.layers:
                    for tensor in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                        if tensor is not None and hasattr(tensor, "nelement"):
                            total_memory += tensor.nelement() * tensor.element_size()
            elif hasattr(cache_obj, "key_cache") and hasattr(cache_obj, "value_cache"):
                for k in getattr(cache_obj, "key_cache", []):
                    if k is not None and hasattr(k, "nelement"):
                        total_memory += k.nelement() * k.element_size()
                for v in getattr(cache_obj, "value_cache", []):
                    if v is not None and hasattr(v, "nelement"):
                        total_memory += v.nelement() * v.element_size()
            elif isinstance(cache_obj, (list, tuple)):
                for item in cache_obj:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        k, v = item
                        if k is not None and hasattr(k, "nelement"):
                            total_memory += k.nelement() * k.element_size()
                        if v is not None and hasattr(v, "nelement"):
                            total_memory += v.nelement() * v.element_size()

        return {
            "active_sessions": self.active_sessions,
            "total_memory_mb": total_memory / 1e6,
            "max_sessions": self.max_sessions,
            "eviction_timeout": self.eviction_timeout,
        }
