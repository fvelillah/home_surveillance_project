"""Unit and integration tests for scripts/evaluate_thresholds.py."""

import json
from pathlib import Path
import numpy as np
import pytest

from scripts.evaluate_thresholds import (
    compute_distribution,
    compute_roc_metrics,
    run_threshold_evaluation,
    update_env_thresholds,
)


def test_compute_distribution_metrics():
    """Tests percentile and summary statistics computation."""
    # 1. Empty set
    empty_dist = compute_distribution([])
    assert empty_dist.sample_count == 0
    assert empty_dist.mean == 0.0

    # 2. Known linear sequence 0.01 to 0.10
    scores = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    dist = compute_distribution(scores)

    assert dist.sample_count == 10
    assert np.isclose(dist.mean, 0.055, atol=1e-3)
    assert np.isclose(dist.p50, 0.055, atol=1e-2)
    assert dist.min == 0.01
    assert dist.max == 0.10
    assert dist.p95 > dist.p90
    assert dist.p99 >= dist.p95


def test_compute_roc_metrics():
    """Tests ROC AUC and F1 threshold sweep calculation."""
    # Well-separated normal and anomaly distributions
    normal_scores = [0.02, 0.03, 0.04, 0.05, 0.06]
    anomaly_scores = [0.20, 0.25, 0.30, 0.35, 0.40]

    roc = compute_roc_metrics(normal_scores, anomaly_scores, steps=100)

    assert roc.auc_roc >= 0.85
    assert roc.best_f1 == 1.0
    assert 0.06 <= roc.best_threshold <= 0.20
    assert roc.tpr_at_best_f1 == 1.0
    assert roc.fpr_at_best_f1 == 0.0


def test_update_env_thresholds(tmp_path: Path):
    """Tests modifying EDGE_TRIAGE_THRESHOLD and CLOUD_ANOMALY_THRESHOLD in .env."""
    mock_env = tmp_path / ".env"
    initial_content = """
# Configuration
EDGE_HOST=0.0.0.0
EDGE_TRIAGE_THRESHOLD=0.050
CLOUD_ANOMALY_THRESHOLD=0.100
"""
    mock_env.write_text(initial_content)

    success = update_env_thresholds(mock_env, edge_triage_th=0.065, cloud_anomaly_th=0.145)
    assert success is True

    updated = mock_env.read_text()
    assert "EDGE_TRIAGE_THRESHOLD=0.065" in updated
    assert "CLOUD_ANOMALY_THRESHOLD=0.145" in updated
    assert "EDGE_HOST=0.0.0.0" in updated


def test_run_threshold_evaluation_end_to_end(tmp_path: Path):
    """Tests end-to-end threshold evaluation with JSON output."""
    test_dir = Path(__file__).resolve().parent.parent / "data" / "tests"
    if not test_dir.exists():
        pytest.skip("data/tests not present")

    out_json = tmp_path / "eval_report.json"
    res = run_threshold_evaluation(
        normal_video_dir=test_dir,
        channels=[1],
        apply_env=False,
        json_output_path=out_json,
        max_windows=2,
    )

    assert res["status"] == "success"
    assert "overall_distribution" in res
    assert "recommended_thresholds" in res
    assert out_json.exists()

    data = json.loads(out_json.read_text())
    assert data["status"] == "success"
    assert "edge_triage_threshold" in data["recommended_thresholds"]
    assert "cloud_anomaly_threshold" in data["recommended_thresholds"]
