"""Tests for escalation handling, re-scoring, and metrics tracking."""

import pytest

from backend.anomaly import index_baseline_vectors
from backend.incidents import incident_engine
from backend.escalation import (
    EscalationRequest,
    EscalationTracker,
    handle_escalation,
    tracker,
)


@pytest.mark.asyncio
async def test_handle_escalation_confirmed_and_rejected():
    tracker.clear()
    incident_engine.clear()
    incident_engine.reset_channel_ema("cam-1")

    # Index normal baseline for camera 1
    base_vec = [1.0, 0.0, 0.0, 0.0]
    index_baseline_vectors(
        vectors=[base_vec],
        metadata_list=[{"id": "b1", "camera_id": "cam-1"}],
    )

    # 1. Escalation that matches baseline (false escalation from edge)
    normal_req = EscalationRequest(
        edge_device_id="cam-1",
        edge_score=0.07,
        edge_embedding=[1.0, 0.0, 0.0, 0.0],
        camera_id="cam-1",
    )
    result_normal = await handle_escalation(normal_req)
    assert pytest.approx(result_normal.cloud_score, 1e-4) == 0.0
    assert result_normal.is_confirmed_anomaly is False
    assert result_normal.incident_id is None

    # 2. Escalation with orthogonal anomalous vector
    anom_req = EscalationRequest(
        edge_device_id="cam-1",
        edge_score=0.25,
        edge_embedding=[0.0, 1.0, 0.0, 0.0],
        camera_id="cam-1",
    )
    result_anom = await handle_escalation(anom_req)
    assert result_anom.cloud_score >= 0.9
    assert result_anom.is_confirmed_anomaly is True
    assert result_anom.incident_id is not None
    assert result_anom.incident_id.startswith("inc-")


def test_escalation_tracker_metrics():
    test_tracker = EscalationTracker()

    from backend.escalation import EscalationResult

    r1 = EscalationResult(
        escalation_id="esc-1",
        edge_device_id="cam-1",
        edge_score=0.10,
        cloud_score=0.02,
        ensemble_score=0.04,
        is_confirmed_anomaly=False,
        confidence=0.92,
    )
    r2 = EscalationResult(
        escalation_id="esc-2",
        edge_device_id="cam-1",
        edge_score=0.20,
        cloud_score=0.25,
        ensemble_score=0.23,
        is_confirmed_anomaly=True,
        confidence=0.95,
        incident_id="inc-esc-2",
    )
    r3 = EscalationResult(
        escalation_id="esc-3",
        edge_device_id="cam-2",
        edge_score=0.18,
        cloud_score=0.30,
        ensemble_score=0.26,
        is_confirmed_anomaly=True,
        confidence=0.88,
        incident_id="inc-esc-3",
    )

    test_tracker.record(r1)
    test_tracker.record(r2)
    test_tracker.record(r3)

    assert test_tracker.escalation_count == 3
    # 2 out of 3 confirmed = 66.67%
    assert pytest.approx(test_tracker.confirmation_rate, 1e-2) == 0.67
    assert pytest.approx(test_tracker.edge_accuracy, 1e-2) == 0.67

    cam1_stats = test_tracker.device_stats("cam-1")
    assert cam1_stats["escalation_count"] == 2
    assert cam1_stats["confirmation_rate"] == 0.5
    assert cam1_stats["false_positive_count"] == 1
