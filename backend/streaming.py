"""Adaptive Streaming Backpressure & Load Shedding Manager."""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Optional

from .config import config
from .models import LoadSheddingStatus

logger = logging.getLogger(__name__)

LEVEL_NAMES = {
    0: "NORMAL",
    1: "SCORE_ONLY",
    2: "PASSTHROUGH",
    3: "SHED_LOAD",
}


class StreamingBackpressureManager:
    """Manages adaptive load shedding across 4 discrete processing tiers."""

    def __init__(
        self,
        auto_mode: Optional[bool] = None,
        score_only_latency_ms: Optional[float] = None,
        passthrough_latency_ms: Optional[float] = None,
        critical_threshold: Optional[float] = None,
    ):
        self.auto_mode = auto_mode if auto_mode is not None else config.load_shed_auto
        self.score_only_latency_ms = score_only_latency_ms if score_only_latency_ms is not None else config.load_shed_score_only_latency_ms
        self.passthrough_latency_ms = passthrough_latency_ms if passthrough_latency_ms is not None else config.load_shed_passthrough_latency_ms
        self.critical_threshold = critical_threshold if critical_threshold is not None else config.load_shed_critical_threshold

        self._manual_level: int = 0
        self._latencies: collections.deque[float] = collections.deque(maxlen=50)
        self._active_requests: int = 0
        self._total_requests: int = 0
        self._shed_requests_count: int = 0
        self._lock = threading.Lock()

    @property
    def current_level(self) -> int:
        """Determines active load shedding level based on auto metrics or manual override."""
        with self._lock:
            if not self.auto_mode:
                return self._manual_level

            avg_lat = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
            depth = self._active_requests

            # Level 3: Extreme overload (shed low priority)
            if depth >= 50 or avg_lat >= 3000.0:
                return 3
            # Level 2: High overload (passthrough edge scores)
            if depth >= 25 or avg_lat >= self.passthrough_latency_ms:
                return 2
            # Level 1: Moderate load (skip heavy VLM inference)
            if depth >= 10 or avg_lat >= self.score_only_latency_ms:
                return 1
            # Level 0: Normal operation
            return 0

    @property
    def level_name(self) -> str:
        return LEVEL_NAMES.get(self.current_level, "NORMAL")

    def set_level(self, level: Optional[int] = None, auto_mode: Optional[bool] = None):
        """Overrides load shedding level or toggles auto mode."""
        with self._lock:
            if auto_mode is not None:
                self.auto_mode = auto_mode
            if level is not None:
                self._manual_level = max(0, min(3, level))
                logger.info("Set streaming load level to %d (%s)", self._manual_level, LEVEL_NAMES.get(self._manual_level))

    def should_skip_vlm(self) -> bool:
        """Returns True if VLM generation should be skipped to conserve compute."""
        return self.current_level >= 1

    def should_skip_baseline_knn(self) -> bool:
        """Returns True if central baseline kNN scoring should be bypassed."""
        return self.current_level >= 2

    def should_drop_escalation(self, edge_score: float) -> bool:
        """Returns True if an escalation should be dropped under severe load."""
        lvl = self.current_level
        if lvl == 3 and edge_score < self.critical_threshold:
            with self._lock:
                self._shed_requests_count += 1
            logger.warning("Load shedding dropped non-critical escalation (edge_score=%.3f < %.3f)", edge_score, self.critical_threshold)
            return True
        return False

    def enter_request(self):
        """Call at start of escalation request to increment concurrency depth."""
        with self._lock:
            self._active_requests += 1
            self._total_requests += 1

    def exit_request(self, latency_ms: float):
        """Call at completion of escalation request to record latency."""
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._latencies.append(latency_ms)

    def get_status(self) -> LoadSheddingStatus:
        """Returns real-time backpressure status and metrics."""
        with self._lock:
            avg_lat = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
            depth = self._active_requests
            tot = self._total_requests
            shed = self._shed_requests_count
            auto = self.auto_mode

        lvl = self.current_level
        return LoadSheddingStatus(
            current_level=lvl,
            level_name=LEVEL_NAMES.get(lvl, "NORMAL"),
            auto_mode=auto,
            avg_latency_ms=round(avg_lat, 2),
            queue_depth=depth,
            total_requests=tot,
            shed_requests_count=shed,
        )

    def clear(self):
        """Resets tracking state (for tests)."""
        with self._lock:
            self._latencies.clear()
            self._active_requests = 0
            self._total_requests = 0
            self._shed_requests_count = 0
            self._manual_level = 0
            self.auto_mode = True


# Global singleton instance
streaming_manager = StreamingBackpressureManager()
