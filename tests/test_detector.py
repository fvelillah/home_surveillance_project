"""Tests for QdrantEdgeDetector."""

import time
from pathlib import Path
import numpy as np
import pytest

from edge.detector import QdrantEdgeDetector, InMemoryTwoShardIndex


def test_in_memory_two_shard_index():
    index = InMemoryTwoShardIndex(embedding_dim=128)

    # Seed baseline for channel 1
    v1 = np.zeros(128, dtype=np.float32)
    v1[0] = 1.0  # Unit vector on axis 0
    index.insert_baseline(channel=1, vectors=[v1])

    # Query identical vector
    results = index.search_knn(channel=1, query_vec=v1, shard="baseline", top_k=1)
    assert len(results) == 1
    dist, _ = results[0]
    assert np.isclose(dist, 0.0, atol=1e-5)

    # Query orthogonal vector
    v2 = np.zeros(128, dtype=np.float32)
    v2[1] = 1.0  # Unit vector on axis 1
    results_orth = index.search_knn(channel=1, query_vec=v2, shard="baseline", top_k=1)
    assert len(results_orth) == 1
    dist_orth, _ = results_orth[0]
    assert np.isclose(dist_orth, 1.0, atol=1e-5)


def test_detector_scoring_and_triage(tmp_path: Path):
    detector = QdrantEdgeDetector(
        qdrant_path=tmp_path / "qdrant",
        embedding_dim=64,
        triage_threshold=0.060,
        alpha_baseline=0.7,
        max_recent_points=10,
    )

    # Seed normal baseline for channel 1: 5 identical or similar vectors
    base_v = np.zeros(64, dtype=np.float32)
    base_v[0] = 1.0
    baseline_vecs = [base_v.copy() for _ in range(5)]
    detector.seed_baseline(channel=1, vectors=baseline_vecs)

    # 1. Score normal clip (same as baseline)
    score_normal, details_normal = detector.score_clip(
        channel=1,
        embedding=base_v,
        clip_id="clip-normal-1",
        timestamp=time.time(),
    )
    assert score_normal < 0.060
    assert not details_normal["is_escalated"]
    assert not detector.is_escalation(score_normal)

    # 2. Score anomalous clip (orthogonal vector)
    anomaly_v = np.zeros(64, dtype=np.float32)
    anomaly_v[10] = 1.0
    score_anom, details_anom = detector.score_clip(
        channel=1,
        embedding=anomaly_v,
        clip_id="clip-anom-1",
        timestamp=time.time() + 8.0,
    )
    assert score_anom >= 0.060
    assert details_anom["is_escalated"]
    assert detector.is_escalation(score_anom)


def test_detector_recent_shard_fifo_pruning(tmp_path: Path):
    max_pts = 5
    detector = QdrantEdgeDetector(
        qdrant_path=tmp_path / "qdrant_fifo",
        embedding_dim=32,
        max_recent_points=max_pts,
    )

    # Insert 12 vectors into channel 2
    for i in range(12):
        v = np.random.randn(32).astype(np.float32)
        v /= np.linalg.norm(v)
        detector.score_clip(
            channel=2,
            embedding=v,
            clip_id=f"clip-{i}",
            timestamp=100.0 + i,
        )

    stats = detector.get_stats()
    # In fallback or Qdrant mode, recent points for channel 2 must be bounded
    if stats.get("engine") == "in_memory_numpy":
        assert len(detector._fallback_index.recent[2]) == max_pts
    elif "recent_points" in stats:
        assert stats["recent_points"] <= max_pts + 1


def test_detector_channel_isolation(tmp_path: Path):
    detector = QdrantEdgeDetector(
        qdrant_path=tmp_path / "qdrant_iso",
        embedding_dim=32,
    )

    v1 = np.zeros(32, dtype=np.float32)
    v1[0] = 1.0
    detector.seed_baseline(channel=1, vectors=[v1])

    # Channel 2 has no baseline -> d_baseline is 0.0, has_baseline is False
    score, details = detector.score_clip(
        channel=2,
        embedding=v1,
        clip_id="clip-ch2",
        timestamp=100.0,
    )
    assert not details["has_baseline"]
