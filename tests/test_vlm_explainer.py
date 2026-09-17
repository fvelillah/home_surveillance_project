"""Tests for VLM Scene Explainer (exclusive Twelve Labs Pegasus integration)."""

import json
from unittest.mock import MagicMock, patch
import pytest

from backend.config import config
from backend.models import IncidentRecord
from backend.vlm_explainer import (
    _parse_vlm_text_response,
    explain_incident,
)


def test_parse_vlm_text_clean_json():
    raw = '{"summary": "Person walking near porch", "actors": ["person"], "action": "walking", "objects": ["box"], "risk_assessment": "routine", "recommended_action": "None"}'
    parsed = _parse_vlm_text_response(raw)
    assert parsed["summary"] == "Person walking near porch"
    assert parsed["risk_assessment"] == "routine"


def test_parse_vlm_text_markdown_json_fences():
    raw = """```json
{
  "summary": "Vehicle parked in driveway",
  "actors": ["car", "driver"],
  "action": "parking",
  "objects": ["vehicle"],
  "risk_assessment": "routine",
  "recommended_action": "Check driveway"
}
```"""
    parsed = _parse_vlm_text_response(raw)
    assert parsed["summary"] == "Vehicle parked in driveway"
    assert "car" in parsed["actors"]


def test_parse_vlm_text_raw_fallback():
    raw = "Just a general string describing motion in the backyard."
    parsed = _parse_vlm_text_response(raw)
    assert "backyard" in parsed["summary"]
    assert parsed["risk_assessment"] == "suspicious"


@pytest.mark.asyncio
async def test_explain_incident_raises_when_disabled():
    inc = IncidentRecord(
        incident_id="inc-test-123",
        channel=1,
        camera_id="cam-1",
        camera_name="Front Door",
        start_time=100.0,
        end_time=110.0,
        duration_s=10.0,
        peak_score=0.45,
        mean_score=0.40,
        severity=38,
        severity_badge="MODERATE",
        status="OPEN",
        event_count=1,
        created_at=100.0,
        updated_at=100.0,
    )

    with patch("backend.twelvelabs_client.is_enabled", return_value=False):
        with pytest.raises(RuntimeError, match="Twelve Labs Pegasus VLM is not enabled"):
            await explain_incident(inc, video_id="vid-123")


@pytest.mark.asyncio
async def test_explain_incident_raises_without_video_or_clip():
    inc = IncidentRecord(
        incident_id="inc-test-123",
        channel=1,
        camera_id="cam-1",
        camera_name="Front Door",
        start_time=100.0,
        end_time=110.0,
        duration_s=10.0,
        peak_score=0.45,
        mean_score=0.40,
        severity=38,
        severity_badge="MODERATE",
        status="OPEN",
        event_count=1,
        created_at=100.0,
        updated_at=100.0,
    )

    with patch("backend.twelvelabs_client.is_enabled", return_value=True):
        with pytest.raises(ValueError, match="no video_id or clip_path provided"):
            await explain_incident(inc)


@pytest.mark.asyncio
async def test_explain_incident_pegasus_mocked():
    inc = IncidentRecord(
        incident_id="inc-test-456",
        channel=2,
        camera_id="cam-2",
        camera_name="Driveway",
        start_time=100.0,
        end_time=115.0,
        duration_s=15.0,
        peak_score=0.88,
        mean_score=0.82,
        severity=80,
        severity_badge="CRITICAL",
        status="OPEN",
        event_count=2,
        created_at=100.0,
        updated_at=100.0,
    )

    mock_analysis = MagicMock()
    mock_analysis.text = json.dumps({
        "summary": "Suspicious figure loitering near vehicle in driveway.",
        "actors": ["masked individual"],
        "action": "examining vehicle door handles",
        "objects": ["car", "flashlight"],
        "risk_assessment": "breach",
        "recommended_action": "Check perimeter lights and alert security",
    })

    with patch("backend.twelvelabs_client.is_enabled", return_value=True), \
         patch("backend.twelvelabs_client.analyze_video", return_value=mock_analysis):

        expl = await explain_incident(inc, video_id="vid-pegasus-999")
        assert expl is not None
        assert expl.model == config.pegasus_model
        assert expl.risk_assessment == "breach"
        assert "masked individual" in expl.actors
        assert "door handles" in expl.action
        assert expl.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_explain_incident_pegasus_with_clip_upload():
    inc = IncidentRecord(
        incident_id="inc-test-789",
        channel=3,
        camera_id="cam-3",
        camera_name="Backyard",
        start_time=200.0,
        end_time=215.0,
        duration_s=15.0,
        peak_score=0.92,
        mean_score=0.85,
        severity=85,
        severity_badge="CRITICAL",
        status="OPEN",
        event_count=3,
        created_at=200.0,
        updated_at=200.0,
    )

    mock_upload = {"pegasus_video_id": "vid-uploaded-777", "status": "indexed"}
    mock_analysis = MagicMock()
    mock_analysis.text = json.dumps({
        "summary": "Unidentified person jumping fence in backyard.",
        "actors": ["intruder"],
        "action": "scaling fence",
        "objects": ["perimeter fence"],
        "risk_assessment": "breach",
        "recommended_action": "Trigger siren and alert operator",
    })

    with patch("backend.twelvelabs_client.is_enabled", return_value=True), \
         patch("backend.twelvelabs_client.upload_video", return_value=mock_upload) as mock_up, \
         patch("backend.twelvelabs_client.analyze_video", return_value=mock_analysis) as mock_an:

        expl = await explain_incident(inc, clip_path="/tmp/fake_clip.mp4")
        assert expl is not None
        assert expl.model == config.pegasus_model
        assert expl.summary == "Unidentified person jumping fence in backyard."
        assert expl.risk_assessment == "breach"
        mock_up.assert_called_once_with("/tmp/fake_clip.mp4", index_type="pegasus")
        mock_an.assert_called_once()
        assert mock_an.call_args[1]["video_id"] == "vid-uploaded-777"


@pytest.mark.asyncio
async def test_explain_incident_pegasus_api_failure():
    inc = IncidentRecord(
        incident_id="inc-test-999",
        channel=1,
        camera_id="cam-1",
        camera_name="Front Door",
        start_time=100.0,
        end_time=110.0,
        duration_s=10.0,
        peak_score=0.50,
        mean_score=0.45,
        severity=45,
        severity_badge="MODERATE",
        status="OPEN",
        event_count=1,
        created_at=100.0,
        updated_at=100.0,
    )

    with patch("backend.twelvelabs_client.is_enabled", return_value=True), \
         patch("backend.twelvelabs_client.analyze_video", side_effect=Exception("API timeout")):

        with pytest.raises(RuntimeError, match="Twelve Labs Pegasus VLM analysis failed"):
            await explain_incident(inc, video_id="vid-error-123")

