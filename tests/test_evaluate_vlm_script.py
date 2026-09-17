"""Automated unit tests for VLM CLI evaluation tool (scripts/evaluate_vlm.py)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from scripts.evaluate_vlm import (
    create_incident_record,
    discover_test_videos,
    extract_video_metadata,
    generate_synthetic_video,
    mock_vlm_explanation,
    run_evaluation,
    main,
)
from backend.models import IncidentRecord, VLMExplanation


def test_generate_and_extract_metadata(tmp_path: Path):
    """Verifies synthetic video generation and OpenCV metadata extraction."""
    vid_path = tmp_path / "test_clip.mp4"
    generated = generate_synthetic_video(vid_path, duration_s=2.0, fps=10, width=320, height=240)
    assert generated.exists()
    assert generated.stat().st_size > 0

    meta = extract_video_metadata(generated)
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["fps"] == 10.0
    assert meta["frame_count"] == 20
    assert meta["duration_s"] == 2.0
    assert meta["size_mb"] > 0


def test_discover_test_videos(tmp_path: Path):
    """Verifies discovering video files by supported extension."""
    (tmp_path / "clip1.mp4").write_text("dummy")
    (tmp_path / "clip2.avi").write_text("dummy")
    (tmp_path / "clip3.mkv").write_text("dummy")
    (tmp_path / "notes.txt").write_text("not a video")

    videos = discover_test_videos(tmp_path)
    names = [v.name for v in videos]
    assert "clip1.mp4" in names
    assert "clip2.avi" in names
    assert "clip3.mkv" in names
    assert "notes.txt" not in names


def test_create_incident_record(tmp_path: Path):
    """Verifies IncidentRecord factory mapping severity to badges."""
    vid_path = tmp_path / "test.mp4"
    meta = {"duration_s": 8.5}

    rec_critical = create_incident_record(vid_path, meta, camera_name="Driveway", channel=2, severity=85)
    assert rec_critical.severity_badge == "CRITICAL"
    assert rec_critical.camera_name == "Driveway"
    assert rec_critical.channel == 2
    assert rec_critical.duration_s == 8.5

    rec_mod = create_incident_record(vid_path, meta, severity=40)
    assert rec_mod.severity_badge == "MODERATE"

    rec_low = create_incident_record(vid_path, meta, severity=10)
    assert rec_low.severity_badge == "LOW"


@pytest.mark.asyncio
async def test_run_evaluation_mock(tmp_path: Path):
    """Verifies run_evaluation logic under --mock flag."""
    vid_path = tmp_path / "intrusion_test.mp4"
    generate_synthetic_video(vid_path, duration_s=1.0, fps=10, width=160, height=120)

    args = MagicMock()
    args.camera_name = "Front Gate"
    args.channel = 3
    args.severity = 80
    args.incident_id = "inc-test-mock"
    args.mock = True
    args.json = True

    result = await run_evaluation(vid_path, video_id=None, args=args)
    assert "video_metadata" in result
    assert "incident" in result
    assert "vlm_explanation" in result
    assert result["incident"]["incident_id"] == "inc-test-mock"
    assert result["vlm_explanation"]["risk_assessment"] == "breach"


@pytest.mark.asyncio
async def test_run_evaluation_live_pegasus_mocked(tmp_path: Path):
    """Verifies live Pegasus execution flow with mocked Twelve Labs API client."""
    vid_path = tmp_path / "real_incident.mp4"
    generate_synthetic_video(vid_path, duration_s=1.0, fps=10, width=160, height=120)

    mock_expl = VLMExplanation(
        summary="A delivery vehicle stopped at the front driveway.",
        actors=["delivery truck", "driver"],
        action="parked in driveway",
        objects=["vehicle", "package"],
        risk_assessment="routine",
        recommended_action="No response needed",
        model="pegasus1.2",
        latency_ms=450.0,
    )

    args = MagicMock()
    args.camera_name = "Driveway"
    args.channel = 2
    args.severity = 30
    args.incident_id = "inc-live-123"
    args.mock = False
    args.json = False

    with patch("backend.twelvelabs_client.is_enabled", return_value=True), \
         patch("scripts.evaluate_vlm.explain_incident", return_value=mock_expl) as mock_func:

        result = await run_evaluation(vid_path, video_id=None, args=args)
        mock_func.assert_called_once()
        assert result["vlm_explanation"]["summary"] == "A delivery vehicle stopped at the front driveway."
        assert result["vlm_explanation"]["risk_assessment"] == "routine"


def test_cli_main_with_sample_and_output(tmp_path: Path):
    """Verifies full CLI invocation producing output file in JSON mode."""
    out_json = tmp_path / "analysis_result.json"
    target_dir = tmp_path / "test_videos"

    test_args = [
        "evaluate_vlm.py",
        "--dir", str(target_dir),
        "--generate-sample",
        "--mock",
        "--json",
        "-o", str(out_json),
    ]

    with patch("sys.argv", test_args):
        code = main()
        assert code == 0

    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "video_metadata" in data
    assert "incident" in data
    assert "vlm_explanation" in data
