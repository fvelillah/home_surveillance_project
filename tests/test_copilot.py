"""Tests for AI Security Copilot Chat & Daily Surveillance Digest (backend/copilot.py)."""

import time
from unittest.mock import patch
import pytest

from backend.config import config
from backend.copilot import SecurityCopilotEngine, copilot_engine
from backend.incidents import incident_engine
from backend.models import IncidentRecord, VLMExplanation


@pytest.fixture(autouse=True)
def clean_copilot_and_incidents():
    """Clears both incident engine and copilot sessions before each test."""
    incident_engine.clear()
    copilot_engine.clear_all()
    yield
    incident_engine.clear()
    copilot_engine.clear_all()


def test_infer_channel_from_query():
    engine = SecurityCopilotEngine()
    cameras = config.parse_camera_names()
    ch1_name = cameras.get(1, "Studio")
    ch2_name = cameras.get(2, "Dining Room")
    ch3_name = cameras.get(3, "Kitchen")

    assert engine._infer_channel_from_query(f"Was there any motion in the {ch1_name}?") == 1
    assert engine._infer_channel_from_query(f"Check activity in the {ch2_name}") == 2
    assert engine._infer_channel_from_query(f"Any activity in {ch3_name}?") == 3
    assert engine._infer_channel_from_query("Show incidents on ch 4") == 4
    assert engine._infer_channel_from_query("What happened on camera 9?") == 9
    assert engine._infer_channel_from_query("Is everything quiet?") is None


def test_copilot_chat_empty_database():
    engine = SecurityCopilotEngine()
    resp = engine.chat(message="Did anyone come to the front door today?")
    assert "no anomalous events" in resp.response.lower() or "normal" in resp.response.lower()
    assert len(resp.cited_incidents) == 0
    assert len(resp.evidence) == 0
    assert resp.session_id.startswith("session-")
    assert resp.latency_ms >= 0


def test_copilot_chat_grounded_evidence_and_citations():
    engine = SecurityCopilotEngine()

    # Form an incident on Channel 1 (Front Door)
    inc = incident_engine.process_escalation(
        channel=1,
        camera_id="cam-1",
        camera_name="Front Door",
        edge_score=0.85,
        cloud_score=0.90,
        ensemble_score=0.88,
        timestamp=time.time() - 300,
    )
    assert inc is not None
    incident_engine.attach_vlm_explanation(
        inc.incident_id,
        VLMExplanation(
            summary="Masked individual wearing dark hoodie tampering with smart lock.",
            actors=["masked intruder", "individual"],
            action="tampering with front door lock",
            objects=["lockpick", "hoodie"],
            risk_assessment="breach",
            recommended_action="Dispatch police immediately",
        ),
    )

    resp = engine.chat(
        message="Who was at the front door recently?",
        session_id="session-test-qa",
    )

    assert resp.session_id == "session-test-qa"
    assert len(resp.cited_incidents) >= 1
    assert inc.incident_id in resp.cited_incidents
    assert len(resp.evidence) >= 1

    ev = resp.evidence[0]
    assert ev.channel == 1
    assert ev.camera_name == "Front Door"
    assert ev.severity_badge == "CRITICAL"
    assert "smart lock" in ev.summary.lower() or "tampering" in ev.summary.lower()
    assert "CRITICAL" in resp.response or "Alert" in resp.response


def test_copilot_chat_multi_turn_history():
    engine = SecurityCopilotEngine()
    sess = "session-multi-turn"

    r1 = engine.chat("Hello copilot, status check please", session_id=sess)
    r2 = engine.chat("Any alerts on camera 2?", session_id=sess)

    history = engine.get_history(sess)
    assert len(history) == 4  # 2 user turns + 2 assistant turns
    assert history[0].role == "user"
    assert history[0].content == "Hello copilot, status check please"
    assert history[1].role == "assistant"
    assert history[2].role == "user"
    assert history[3].role == "assistant"

    # Clear history
    deleted = engine.clear_history(sess)
    assert deleted is True
    assert len(engine.get_history(sess)) == 0


def test_daily_digest_empty_records():
    engine = SecurityCopilotEngine()
    digest = engine.generate_daily_digest()

    assert digest.total_incidents == 0
    assert digest.total_events == 0
    assert digest.critical_incidents == 0
    assert digest.threat_level == "LOW"
    assert "OPTIMAL" in digest.executive_summary or "normal" in digest.executive_summary.lower()
    assert len(digest.channel_summaries) == 9
    assert len(digest.key_incidents) == 0
    assert "# 🛡️ Daily Home Surveillance Security Digest" in digest.markdown_text
    assert "**LOW**" in digest.markdown_text


def test_daily_digest_multiple_incidents_and_threat_levels():
    engine = SecurityCopilotEngine()
    t_now = time.time()

    # Channel 1: High severity incident
    inc1 = incident_engine.process_escalation(
        channel=1,
        camera_id="cam-1",
        camera_name="Front Door",
        edge_score=0.60,
        cloud_score=0.65,
        ensemble_score=0.62,
        timestamp=t_now - 7200,
    )
    incident_engine.attach_vlm_explanation(
        inc1.incident_id,
        VLMExplanation(
            summary="Unknown visitor rang doorbell and waited 5 minutes.",
            actors=["visitor"],
            action="ringing doorbell",
            objects=["doorbell"],
            risk_assessment="suspicious",
            recommended_action="Check doorbell intercom",
        ),
    )

    # Channel 5: Critical severity breach on Back Patio
    inc2 = incident_engine.process_escalation(
        channel=5,
        camera_id="cam-5",
        camera_name="Back Patio",
        edge_score=0.88,
        cloud_score=0.92,
        ensemble_score=0.90,
        timestamp=t_now - 3600,
    )
    incident_engine.attach_vlm_explanation(
        inc2.incident_id,
        VLMExplanation(
            summary="Intruder shattered patio sliding glass door with heavy tool.",
            actors=["intruder"],
            action="breaking patio glass door",
            objects=["heavy tool", "glass door"],
            risk_assessment="breach",
            recommended_action="Emergency dispatch",
        ),
    )

    # Generate 24-hour digest
    digest = engine.generate_daily_digest(
        start_time=t_now - 86400,
        end_time=t_now,
    )

    assert digest.total_incidents == 2
    assert digest.critical_incidents == 1
    assert digest.threat_level == "SEVERE"
    assert len(digest.key_incidents) == 2

    # Verify top key incident is the Critical Back Patio breach
    top_inc = digest.key_incidents[0]
    assert top_inc.channel == 5
    assert top_inc.camera_name == "Back Patio"
    assert top_inc.severity_badge == "CRITICAL"
    assert "sliding glass" in top_inc.summary or "patio" in top_inc.summary.lower()

    # Check channel breakdown
    ch5_sum = next(cs for cs in digest.channel_summaries if cs.channel == 5)
    assert ch5_sum.total_incidents == 1
    assert ch5_sum.critical_count == 1
    assert ch5_sum.peak_severity >= 75

    # Check markdown output contents
    assert "Daily Home Surveillance Security Digest" in digest.markdown_text
    assert "Back Patio" in digest.markdown_text
    assert "**SEVERE**" in digest.markdown_text
    assert "🚨 CRITICAL" in "\n".join(digest.recommended_actions)
