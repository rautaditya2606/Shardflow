"""
Session state & lifecycle management for scheduled requests.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class SessionState(Enum):
    PENDING = "PENDING"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    max_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 1.0
    state: SessionState = SessionState.PENDING
    created_at: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    generated_tokens: List[int] = field(default_factory=list)
    error_message: Optional[str] = None

    def mark_prefill(self):
        self.state = SessionState.PREFILLING
        if self.start_time is None:
            self.start_time = time.time()

    def mark_decode(self):
        self.state = SessionState.DECODING

    def mark_completed(self):
        self.state = SessionState.COMPLETED
        self.finish_time = time.time()

    def mark_failed(self, error: str):
        self.state = SessionState.FAILED
        self.error_message = error
        self.finish_time = time.time()
