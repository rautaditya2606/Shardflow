"""
Per-node KV cache store — manages cached key/value tensors by session ID.

Supports:
1. StaticKVCache: Pre-allocated, zero-allocation contiguous buffers for CUDA Graph replay.
2. DynamicCache: Fallback dynamic KV cache for CPU or unbounded sequence lengths.
3. Multi-session slot leasing with VRAM capping (default max 4 sessions on 16GB GPUs).
4. Automatic TTL and LRU eviction loops.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Union
import torch
from transformers.cache_utils import DynamicCache, StaticCache, Cache
from transformers import AutoConfig

logger = logging.getLogger(__name__)


class StaticKVSlot:
    """
    Pre-allocated static KV cache slot for a single active generation session.
    Reuses contiguous CUDA buffers across sessions via reset() without tensor re-allocation.
    """

    def __init__(
        self,
        slot_id: int,
        config: AutoConfig,
        max_seq_len: int = 2048,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.slot_id = slot_id
        self.config = config
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        self.dtype = dtype

        self.session_id: Optional[str] = None
        self.last_access: float = 0.0
        self.seq_len: int = 0

        # Pre-allocate StaticCache on target device
        # ponytail: pass device/dtype/max_batch_size via kwargs for forward-compat with
        # newer transformers that moved these params out of the explicit signature
        self.cache = StaticCache(
            config=config,
            max_cache_len=max_seq_len,
            device=self.device,
            dtype=self.dtype,
            max_batch_size=1,
        )

    def is_free(self) -> bool:
        return self.session_id is None

    def lease(self, session_id: str) -> StaticCache:
        """Bind this slot to a new session ID."""
        self.session_id = session_id
        self.last_access = time.monotonic()
        self.seq_len = 0
        self.cache.reset()
        return self.cache

    def release(self) -> None:
        """Release this slot for recycling."""
        self.session_id = None
        self.last_access = 0.0
        self.seq_len = 0
        try:
            self.cache.reset()
        except Exception:
            pass

    def touch(self) -> None:
        self.last_access = time.monotonic()


class KVCacheStore:
    """
    Per-session KV cache store for a node's layer slice.
    Manages both pre-allocated StaticKVSlots and dynamic fallback caches.
    """

    def __init__(
        self,
        eviction_timeout: float = 60.0,
        max_sessions: int = 4,
        max_seq_len: int = 2048,
        enable_static_cache: bool = True,
    ):
        """
        Args:
            eviction_timeout: seconds of inactivity before a session's cache is evicted
            max_sessions: maximum concurrent cached sessions (default 4 on 16GB GPUs to cap VRAM <= 2GB)
            max_seq_len: maximum pre-allocated sequence length for static cache
            enable_static_cache: if True, use StaticCache slots on CUDA
        """
        self.eviction_timeout = eviction_timeout
        self.max_sessions = max_sessions
        self.max_seq_len = max_seq_len
        self.enable_static_cache = enable_static_cache

        # Static cache slot pool
        self._static_slots: list[StaticKVSlot] = []
        self._session_to_slot: dict[str, StaticKVSlot] = {}

        # Fallback dynamic cache store: session_id -> Cache
        self._dynamic_cache: dict[str, Cache] = {}
        self._dynamic_last_access: dict[str, float] = {}

        # Background eviction task
        self._eviction_task: Optional[asyncio.Task] = None

    def initialize_static_pool(
        self,
        config: AutoConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Pre-allocate static KV cache slots on the target device."""
        if not self.enable_static_cache or device.type != "cuda" or not torch.cuda.is_available():
            logger.info("Static KV cache pool disabled or running on CPU — using DynamicCache fallback.")
            return

        try:
            logger.info(
                "Pre-allocating %d StaticKVSlots on %s (max_seq=%d, dtype=%s)...",
                self.max_sessions, device, self.max_seq_len, dtype,
            )
            self._static_slots = [
                StaticKVSlot(
                    slot_id=i,
                    config=config,
                    max_seq_len=self.max_seq_len,
                    device=str(device),
                    dtype=dtype,
                )
                for i in range(self.max_sessions)
            ]
            logger.info("Successfully pre-allocated %d static KV cache slots.", len(self._static_slots))
        except Exception as e:
            logger.warning("Could not pre-allocate static KV slots (%s) — falling back to DynamicCache.", e)
            self._static_slots = []

    def get_or_create(
        self,
        session_id: str,
        config: Optional[AutoConfig] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Cache:
        """
        Get or allocate a KV cache for session_id.
        Tries static slot leasing first, falls back to DynamicCache.
        """
        # 1. Check if already leased in a static slot
        if session_id in self._session_to_slot:
            slot = self._session_to_slot[session_id]
            slot.touch()
            return slot.cache

        # 2. Check if already in dynamic cache
        if session_id in self._dynamic_cache:
            self._dynamic_last_access[session_id] = time.monotonic()
            return self._dynamic_cache[session_id]

        # 3. Try to acquire a free static slot
        if self._static_slots:
            # Find free slot
            for slot in self._static_slots:
                if slot.is_free():
                    slot.lease(session_id)
                    self._session_to_slot[session_id] = slot
                    logger.info("Leased StaticKVSlot %d to session %s", slot.slot_id, session_id)
                    return slot.cache

            # If all slots leased, evict oldest static slot
            oldest_slot = min(self._static_slots, key=lambda s: s.last_access)
            old_sid = oldest_slot.session_id
            if old_sid:
                self.evict(old_sid)
            oldest_slot.lease(session_id)
            self._session_to_slot[session_id] = oldest_slot
            logger.info("Evicted session %s to lease StaticKVSlot %d to %s", old_sid, oldest_slot.slot_id, session_id)
            return oldest_slot.cache

        # 4. DynamicCache fallback
        if len(self._dynamic_cache) >= self.max_sessions:
            self._evict_oldest_dynamic()

        cache_obj = DynamicCache()
        self._dynamic_cache[session_id] = cache_obj
        self._dynamic_last_access[session_id] = time.monotonic()
        return cache_obj

    def get(self, session_id: str) -> Optional[Cache]:
        """Get existing cache for session_id without creating a new one."""
        if session_id in self._session_to_slot:
            slot = self._session_to_slot[session_id]
            slot.touch()
            return slot.cache
        if session_id in self._dynamic_cache:
            self._dynamic_last_access[session_id] = time.monotonic()
            return self._dynamic_cache[session_id]
        return None

    def put(self, session_id: str, cache_obj: Cache) -> None:
        """Store dynamic cache object."""
        if session_id not in self._session_to_slot:
            self._dynamic_cache[session_id] = cache_obj
            self._dynamic_last_access[session_id] = time.monotonic()

    def evict(self, session_id: str) -> bool:
        """Evict session from both static slot pool and dynamic cache."""
        evicted = False
        if session_id in self._session_to_slot:
            slot = self._session_to_slot.pop(session_id)
            slot.release()
            logger.info("Recycled StaticKVSlot %d for session %s", slot.slot_id, session_id)
            evicted = True

        if session_id in self._dynamic_cache:
            del self._dynamic_cache[session_id]
            self._dynamic_last_access.pop(session_id, None)
            logger.info("Evicted DynamicCache for session %s", session_id)
            evicted = True

        return evicted

    def _evict_oldest_dynamic(self) -> None:
        if not self._dynamic_last_access:
            return
        oldest_session = min(self._dynamic_last_access, key=self._dynamic_last_access.get)
        self.evict(oldest_session)

    def _evict_expired(self) -> int:
        now = time.monotonic()
        expired = []
        for sid, slot in list(self._session_to_slot.items()):
            if now - slot.last_access > self.eviction_timeout:
                expired.append(sid)

        for sid, last in list(self._dynamic_last_access.items()):
            if now - last > self.eviction_timeout:
                expired.append(sid)

        for sid in set(expired):
            self.evict(sid)
        return len(set(expired))

    async def start_eviction_loop(self) -> None:
        """Start the background eviction loop."""
        self._eviction_task = asyncio.create_task(self._eviction_loop())
        logger.info(
            "KV cache eviction loop started (timeout=%ds, max_sessions=%d)",
            self.eviction_timeout, self.max_sessions,
        )

    async def _eviction_loop(self) -> None:
        while True:
            await asyncio.sleep(15)
            evicted = self._evict_expired()
            if evicted > 0:
                logger.info("Evicted %d expired sessions (%d active)", evicted, self.active_sessions)

    async def stop_eviction_loop(self) -> None:
        if self._eviction_task:
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass
            self._eviction_task = None

    def clear_all(self) -> None:
        for slot in self._static_slots:
            slot.release()
        self._session_to_slot.clear()
        self._dynamic_cache.clear()
        self._dynamic_last_access.clear()

    @property
    def active_sessions(self) -> int:
        return len(self._session_to_slot) + len(self._dynamic_cache)

    def stats(self) -> dict:
        total_memory = 0
        for slot in self._static_slots:
            if hasattr(slot.cache, "layers") and isinstance(slot.cache.layers, (list, tuple)):
                for layer in slot.cache.layers:
                    for tensor in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                        if tensor is not None and hasattr(tensor, "nelement") and hasattr(tensor, "element_size"):
                            total_memory += tensor.nelement() * tensor.element_size()
        for session_id, cache_obj in self._dynamic_cache.items():
            if hasattr(cache_obj, "layers") and isinstance(cache_obj.layers, (list, tuple)):
                for layer in cache_obj.layers:
                    for tensor in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                        if tensor is not None and hasattr(tensor, "nelement") and hasattr(tensor, "element_size"):
                            total_memory += tensor.nelement() * tensor.element_size()
            elif hasattr(cache_obj, "key_cache") and hasattr(cache_obj, "value_cache"):
                for k in getattr(cache_obj, "key_cache", []):
                    if k is not None and hasattr(k, "nelement") and hasattr(k, "element_size"):
                        total_memory += k.nelement() * k.element_size()
                for v in getattr(cache_obj, "value_cache", []):
                    if v is not None and hasattr(v, "nelement") and hasattr(v, "element_size"):
                        total_memory += v.nelement() * v.element_size()

        return {
            "active_sessions": self.active_sessions,
            "total_memory_mb": total_memory / 1e6,
            "static_slots_allocated": len(self._static_slots),
            "static_slots_leased": len(self._session_to_slot),
            "dynamic_sessions": len(self._dynamic_cache),
            "max_sessions": self.max_sessions,
            "eviction_timeout": self.eviction_timeout,
        }
