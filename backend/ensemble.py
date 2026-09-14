"""Multi-model ensemble scoring combining edge and cloud embeddings with temporal boosting."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import config


@dataclass
class EnsembleResult:
    """Detailed output from the multi-model ensemble fusion engine."""
    edge_score: float
    cloud_score: float
    ensemble_score: float
    confidence: float
    is_anomaly: bool
    temporal_boost: float = 0.0
    weights_used: Dict[str, float] = field(default_factory=dict)


class EnsembleScorer:
    """Combines edge and cloud anomaly scores with temporal context and adaptive thresholds.

    Formula:
        Raw Score = (w_cloud * S_cloud) + (w_edge * S_edge)
        Final Score = min(1.0, Raw Score + Temporal Boost)
    """

    def __init__(
        self,
        cloud_weight: float = config.ensemble_cloud_weight,
        edge_weight: float = config.ensemble_edge_weight,
        default_threshold: float = config.ensemble_threshold,
        boost_window_min: float = config.temporal_boost_window_min,
        boost_factor: float = config.temporal_boost_factor,
        max_boost: float = config.max_temporal_boost,
    ):
        self.cloud_weight = cloud_weight
        self.edge_weight = edge_weight
        self.default_threshold = default_threshold
        self.boost_window_min = boost_window_min
        self.boost_factor = boost_factor
        self.max_boost = max_boost

        # Per-device recent escalation timestamps
        self._recent_escalations: Dict[str, List[float]] = defaultdict(list)
        # Per-device customized decision thresholds
        self._device_thresholds: Dict[str, float] = {}

    def score(
        self,
        edge_score: float,
        cloud_score: float,
        device_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> EnsembleResult:
        """Computes multi-model ensemble score and evaluates anomaly boundary."""
        raw_ensemble = (self.cloud_weight * cloud_score) + (self.edge_weight * edge_score)

        boost = 0.0
        if device_id and timestamp is not None:
            boost = self._compute_temporal_boost(device_id, timestamp)

        ensemble_score = min(1.0, max(0.0, raw_ensemble + boost))

        # Confidence is higher when edge and cloud models agree
        score_diff = abs(edge_score - cloud_score)
        confidence = max(0.0, min(1.0, 1.0 - score_diff))

        # Check against per-device adaptive threshold or global default
        threshold = self.get_device_threshold(device_id) if device_id else self.default_threshold
        is_anomaly = ensemble_score >= threshold

        return EnsembleResult(
            edge_score=edge_score,
            cloud_score=cloud_score,
            ensemble_score=ensemble_score,
            confidence=confidence,
            is_anomaly=is_anomaly,
            temporal_boost=boost,
            weights_used={"cloud": self.cloud_weight, "edge": self.edge_weight},
        )

    def _compute_temporal_boost(self, device_id: str, timestamp: float) -> float:
        """Applies a temporal boost if multiple escalations arrive from the same device in a short window."""
        history = self._recent_escalations[device_id]
        history.append(timestamp)

        # Retain only timestamps within the sliding window
        cutoff = timestamp - (self.boost_window_min * 60.0)
        self._recent_escalations[device_id] = [t for t in history if t >= cutoff]

        count = len(self._recent_escalations[device_id])
        if count <= 1:
            return 0.0

        # Increment boost for each consecutive escalation beyond the first
        return min(self.max_boost, (count - 1) * self.boost_factor)

    def set_device_threshold(self, device_id: str, threshold: float):
        self._device_thresholds[device_id] = threshold

    def get_device_threshold(self, device_id: Optional[str]) -> float:
        if device_id and device_id in self._device_thresholds:
            return self._device_thresholds[device_id]
        return self.default_threshold

    def clear_history(self, device_id: Optional[str] = None):
        """Clears temporal escalation history (useful for tests or state resets)."""
        if device_id:
            self._recent_escalations.pop(device_id, None)
        else:
            self._recent_escalations.clear()


# Global ensemble scorer singleton
ensemble_scorer = EnsembleScorer()
