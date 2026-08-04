"""
Prometheus-style metrics collector for Inference Orchestrator & Gateway.
"""

import time
from typing import Dict, Any, List


class MetricsCollector:
    """Collects runtime performance and health metrics."""

    def __init__(self):
        self.active_sessions: int = 0
        self.total_requests: int = 0
        self.total_tokens_generated: int = 0
        self.failed_requests: int = 0
        self.token_latencies: List[float] = []
        self.start_time: float = time.time()

    def record_request_start(self):
        self.active_sessions += 1
        self.total_requests += 1

    def record_request_end(self, tokens_generated: int, success: bool = True):
        self.active_sessions = max(0, self.active_sessions - 1)
        if success:
            self.total_tokens_generated += tokens_generated
        else:
            self.failed_requests += 1

    def record_token_latency(self, latency_seconds: float):
        self.token_latencies.append(latency_seconds)
        # Keep last 1000 sample points
        if len(self.token_latencies) > 1000:
            self.token_latencies = self.token_latencies[-1000:]

    def get_summary(self) -> Dict[str, Any]:
        avg_latency = (
            sum(self.token_latencies) / len(self.token_latencies)
            if self.token_latencies
            else 0.0
        )
        p95_latency = (
            sorted(self.token_latencies)[int(len(self.token_latencies) * 0.95)]
            if self.token_latencies
            else 0.0
        )

        uptime = time.time() - self.start_time
        overall_tok_s = self.total_tokens_generated / uptime if uptime > 0 else 0.0

        return {
            "uptime_seconds": round(uptime, 1),
            "active_sessions": self.active_sessions,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "overall_tokens_per_sec": round(overall_tok_s, 2),
            "avg_token_latency_sec": round(avg_latency, 4),
            "p95_token_latency_sec": round(p95_latency, 4),
        }


# Global metrics instance
metrics = MetricsCollector()
