"""
Layer 3 — Request Scheduler

Maintains an asyncio.Queue of pending requests, manages session lifecycles,
caps active session concurrency, and feeds requests into the orchestrator pipeline.
"""

import asyncio
import logging
from typing import Dict, Optional
from shardflow.scheduler.session import Session, SessionState

logger = logging.getLogger(__name__)


class RequestScheduler:
    """
    Sits between Gateway and Orchestrator.
    Manages pending requests queue and controls concurrent active sessions.
    """

    def __init__(self, max_concurrent_sessions: int = 16):
        self.max_concurrent_sessions = max_concurrent_sessions
        self.pending_queue: asyncio.Queue[Session] = asyncio.Queue()
        self.active_sessions: Dict[str, Session] = {}
        self._all_sessions: Dict[str, Session] = {}
        self._dispatch_task: Optional[asyncio.Task] = None

    async def submit_request(self, session: Session) -> str:
        """Submit a request session to the pending queue."""
        self._all_sessions[session.session_id] = session
        await self.pending_queue.put(session)
        logger.info(
            "Submitted session %s to queue (pending depth: %d)",
            session.session_id, self.pending_queue.qsize()
        )
        return session.session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """Look up active or queued session by ID."""
        return self._all_sessions.get(session_id) or self.active_sessions.get(session_id)

    def cancel_session(self, session_id: str) -> bool:
        """Mark session cancelled (whether active or pending in queue)."""
        session = self._all_sessions.get(session_id) or self.active_sessions.get(session_id)
        if session:
            session.state = SessionState.CANCELLED
            logger.info("Cancelled session %s", session_id)
            return True
        return False

    def remove_session(self, session_id: str):
        """Remove completed or failed session from active tracking."""
        self.active_sessions.pop(session_id, None)
        self._all_sessions.pop(session_id, None)

    @property
    def current_load(self) -> dict:
        return {
            "pending_queue_size": self.pending_queue.qsize(),
            "active_sessions_count": len(self.active_sessions),
            "max_concurrent_sessions": self.max_concurrent_sessions,
        }
