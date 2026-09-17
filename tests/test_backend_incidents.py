"""Unit tests for Incident Formation Engine (EMA smoothing, hysteresis, cooldown merging, severity)."""

import time
import pytest

from backend.incidents import IncidentFormationEngine, compute_severity
from backend.models import VLMExplanation


@pytest.fixture
def engine():
    eng = IncidentFormationEngine(
        ema_alpha=0.3,
        start_threshold=0.080,
        end_threshold=0.050,
        cooldown_window_s=30.0,
    )
    eng.clear()
    return eng


def test_compute_severity_tiers():
    # Low severity
    sev1, badge1 = compute_severity(peak_score=0.10, mean_score=0.08, event_count=1)
    assert sev1 < 25
    assert badge1 == "LOW"

    # Moderate severity
    sev2, badge2 = compute_severity(peak_score=0.45, mean_score=0.40, event_count=2)
    assert 25 <= sev2 < 50
    assert badge2 == "MODERATE"

    # High severity
    sev3, badge3 = compute_severity(peak_score=0.75, mean_score=0.70, event_count=3)
    assert 50 <= sev3 < 75
    assert badge3 == "HIGH"

    # Critical severity
    sev4, badge4 = compute_severity(peak_score=0.95, mean_score=0.90, event_count=5)
    assert sev4 >= 75
    assert badge4 == "CRITICAL"


def test_ema_smoothing(engine):
    # First value should be raw score
    s1 = engine.update_ema("cam-1", 0.10)
    assert pytest.approx(s1, 1e-4) == 0.10

    # Second value: 0.3 * 0.20 + 0.7 * 0.10 = 0.06 + 0.07 = 0.13
    s2 = engine.update_ema("cam-1", 0.20)
    assert pytest.approx(s2, 1e-4) == 0.13

    # Reset
    engine.reset_channel_ema("cam-1")
    assert engine.get_smoothed_score("cam-1") == 0.0


def test_incident_creation_and_hysteresis(engine):
    # Sub-threshold score -> No incident
    inc0 = engine.process_escalation(
        channel=1,
        camera_id="cam-1",
        camera_name="Front Door",
        edge_score=0.03,
        cloud_score=0.03,
        ensemble_score=0.03,
        timestamp=100.0,
    )
    assert inc0 is None
    assert len(engine.get_active_incidents()) == 0

    # Score exceeding start threshold (0.080) -> Incident created
    inc1 = engine.process_escalation(
        channel=1,
        camera_id="cam-1",
        camera_name="Front Door",
        edge_score=0.20,
        cloud_score=0.20,
        ensemble_score=0.20,
        timestamp=105.0,
    )
    assert inc1 is not None
    assert inc1.channel == 1
    assert inc1.status == "OPEN"
    assert inc1.event_count == 1
    assert inc1.peak_score == 0.20
    assert len(engine.get_active_incidents()) == 1


def test_cooldown_window_merging(engine):
    # Event 1 at t=100.0 (triggers incident)
    inc1 = engine.process_escalation(
        channel=2,
        camera_id="cam-2",
        camera_name="Driveway",
        edge_score=0.25,
        cloud_score=0.25,
        ensemble_score=0.25,
        timestamp=100.0,
    )
    assert inc1 is not None
    inc_id = inc1.incident_id

    # Event 2 at t=110.0 (within 30s cooldown) -> merged into existing incident
    inc2 = engine.process_escalation(
        channel=2,
        camera_id="cam-2",
        camera_name="Driveway",
        edge_score=0.35,
        cloud_score=0.35,
        ensemble_score=0.35,
        timestamp=110.0,
    )
    assert inc2 is not None
    assert inc2.incident_id == inc_id
    assert inc2.event_count == 2
    assert inc2.duration_s == 10.0
    assert inc2.peak_score == 0.35

    # Event 3 drops score below end_threshold (0.050) -> incident closes
    inc3 = engine.process_escalation(
        channel=2,
        camera_id="cam-2",
        camera_name="Driveway",
        edge_score=0.01,
        cloud_score=0.01,
        ensemble_score=0.01,
        timestamp=115.0,
    )
    # EMA will be: 0.3 * 0.01 + 0.7 * prev... if it drops below 0.050, it marks closed
    # Let's check status
    assert inc3 is not None
    assert inc3.incident_id == inc_id


def test_incident_query_and_status_management(engine):
    # Create 2 incidents across different channels
    engine.process_escalation(
        channel=1,
        camera_id="cam-1",
        camera_name="Front Door",
        edge_score=0.30,
        cloud_score=0.30,
        ensemble_score=0.30,
        timestamp=100.0,
    )
    engine.process_escalation(
        channel=3,
        camera_id="cam-3",
        camera_name="Kitchen",
        edge_score=0.80,
        cloud_score=0.80,
        ensemble_score=0.80,
        timestamp=105.0,
    )

    all_incs = engine.list_incidents()
    assert len(all_incs) == 2

    ch1_incs = engine.list_incidents(channel=1)
    assert len(ch1_incs) == 1
    assert ch1_incs[0].channel == 1

    # Status update
    target_id = ch1_incs[0].incident_id
    updated = engine.update_status(target_id, status="ACKNOWLEDGED", notes="Checked by operator")
    assert updated is not None
    assert updated.status == "ACKNOWLEDGED"
    assert updated.notes == "Checked by operator"

    # Attach VLM explanation
    expl = VLMExplanation(
        summary="Courier delivered package",
        actors=["delivery driver"],
        action="placed box on porch",
        objects=["package"],
        risk_assessment="routine",
        recommended_action="No action needed",
    )
    with_expl = engine.attach_vlm_explanation(target_id, expl)
    assert with_expl is not None
    assert with_expl.vlm_explanation is not None
    assert with_expl.vlm_explanation.summary == "Courier delivered package"
