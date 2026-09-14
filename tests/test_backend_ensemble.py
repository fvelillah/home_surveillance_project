"""Tests for multi-model ensemble scoring and temporal boosting."""

import time
import pytest

from backend.ensemble import EnsembleScorer


def test_ensemble_raw_fusion():
    scorer = EnsembleScorer(
        cloud_weight=0.7,
        edge_weight=0.3,
        default_threshold=0.15,
    )

    # Edge score = 0.20, Cloud score = 0.10
    # Expected raw = 0.7 * 0.10 + 0.3 * 0.20 = 0.07 + 0.06 = 0.13
    res = scorer.score(edge_score=0.20, cloud_score=0.10)
    assert pytest.approx(res.ensemble_score, 1e-4) == 0.13
    assert res.is_anomaly is False
    assert pytest.approx(res.confidence, 1e-4) == 0.90  # 1.0 - |0.20 - 0.10|


def test_ensemble_temporal_boosting():
    scorer = EnsembleScorer(
        cloud_weight=0.7,
        edge_weight=0.3,
        default_threshold=0.15,
        boost_window_min=5.0,
        boost_factor=0.05,
        max_boost=0.20,
    )
    scorer.clear_history()

    now = time.time()

    # 1st escalation: no boost (count = 1)
    res1 = scorer.score(edge_score=0.10, cloud_score=0.10, device_id="cam-1", timestamp=now)
    assert res1.temporal_boost == 0.0
    assert pytest.approx(res1.ensemble_score, 1e-4) == 0.10

    # 2nd escalation within 10 seconds: boost = (2 - 1) * 0.05 = 0.05
    res2 = scorer.score(edge_score=0.10, cloud_score=0.10, device_id="cam-1", timestamp=now + 10)
    assert pytest.approx(res2.temporal_boost, 1e-4) == 0.05
    assert pytest.approx(res2.ensemble_score, 1e-4) == 0.15
    assert res2.is_anomaly is True  # 0.15 >= 0.15 threshold

    # 3rd escalation within 20 seconds: boost = (3 - 1) * 0.05 = 0.10
    res3 = scorer.score(edge_score=0.10, cloud_score=0.10, device_id="cam-1", timestamp=now + 20)
    assert pytest.approx(res3.temporal_boost, 1e-4) == 0.10
    assert pytest.approx(res3.ensemble_score, 1e-4) == 0.20

    # Camera 2 gets independent history (no boost)
    res_cam2 = scorer.score(edge_score=0.10, cloud_score=0.10, device_id="cam-2", timestamp=now + 25)
    assert res_cam2.temporal_boost == 0.0


def test_ensemble_adaptive_device_threshold():
    scorer = EnsembleScorer(
        cloud_weight=0.7,
        edge_weight=0.3,
        default_threshold=0.15,
    )

    # Set custom sensitivity for driveway (cam-2)
    scorer.set_device_threshold("cam-2", 0.08)
    assert scorer.get_device_threshold("cam-2") == 0.08
    assert scorer.get_device_threshold("cam-1") == 0.15

    # Score of 0.10 is NOT anomaly for cam-1, but IS for cam-2
    res_cam1 = scorer.score(edge_score=0.10, cloud_score=0.10, device_id="cam-1")
    res_cam2 = scorer.score(edge_score=0.10, cloud_score=0.10, device_id="cam-2")

    assert res_cam1.is_anomaly is False
    assert res_cam2.is_anomaly is True
