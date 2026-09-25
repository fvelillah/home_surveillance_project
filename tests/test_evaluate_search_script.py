"""Automated unit tests for Semantic Search CLI tool (scripts/evaluate_search.py)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.incidents import incident_engine
from backend.models import IncidentRecord, VLMExplanation
from scripts.evaluate_search import (
    discover_test_videos,
    extract_video_metadata,
    generate_synthetic_video,
    index_test_videos,
    mock_vlm_for_clip,
    main,
)


@pytest.fixture(autouse=True)
def clean_incidents():
    """Ensures incident engine is clean before and after each test."""
    incident_engine.clear()
    yield
    incident_engine.clear()


def test_discover_test_videos(tmp_path: Path):
    """Verifies finding video files by supported extension."""
    (tmp_path / "clip1.mp4").write_text("dummy")
    (tmp_path / "clip2.avi").write_text("dummy")
    (tmp_path / "clip3.mov").write_text("dummy")
    (tmp_path / "notes.txt").write_text("not a video")

    videos = discover_test_videos(tmp_path)
    names = [v.name for v in videos]
    assert "clip1.mp4" in names
    assert "clip2.avi" in names
    assert "clip3.mov" in names
    assert "notes.txt" not in names


def test_generate_synthetic_video_and_metadata(tmp_path: Path):
    """Verifies synthetic video creation and OpenCV metadata extraction."""
    vid_path = tmp_path / "intrusion_clip.mp4"
    generated = generate_synthetic_video(vid_path, duration_s=1.0, fps=10, width=320, height=240, scene_type="intrusion")
    assert generated.exists()
    assert generated.stat().st_size > 0

    meta = extract_video_metadata(generated)
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["fps"] == 10.0
    assert meta["duration_s"] == 1.0


def test_mock_vlm_for_clip(tmp_path: Path):
    """Verifies mock VLM generator creates context-relevant explanations."""
    v_intrusion = tmp_path / "test_window_intruder.mp4"
    vlm_int = mock_vlm_for_clip(v_intrusion, 1, "Front Door")
    assert vlm_int.risk_assessment == "breach"
    assert "intruder" in vlm_int.actors

    v_delivery = tmp_path / "test_courier_delivery.mp4"
    vlm_del = mock_vlm_for_clip(v_delivery, 2, "Driveway")
    assert vlm_del.risk_assessment == "routine"
    assert "delivery driver" in vlm_del.actors


@pytest.mark.asyncio
async def test_index_test_videos(tmp_path: Path):
    """Verifies indexing discovered video files into IncidentFormationEngine."""
    vid1 = generate_synthetic_video(tmp_path / "intrusion_1.mp4", duration_s=1.0, fps=10, width=160, height=120, scene_type="intrusion")
    vid2 = generate_synthetic_video(tmp_path / "delivery_1.mp4", duration_s=1.0, fps=10, width=160, height=120, scene_type="delivery")

    indexed = await index_test_videos([vid1, vid2], mock_mode=True)
    assert len(indexed) == 2
    assert len(incident_engine.list_incidents()) == 2


def test_cli_main_direct_query(tmp_path: Path):
    """Verifies CLI execution with --query, --mock, --json, and --output."""
    test_dir = tmp_path / "test_clips"
    test_dir.mkdir(parents=True, exist_ok=True)
    generate_synthetic_video(test_dir / "clip_intrusion.mp4", duration_s=1.0, fps=10, width=160, height=120, scene_type="intrusion")

    out_file = tmp_path / "search_results.json"
    cli_args = [
        "evaluate_search.py",
        "--dir", str(test_dir),
        "--query", "intruder window",
        "--mock",
        "--json",
        "-o", str(out_file),
    ]

    with patch("sys.argv", cli_args):
        code = main()
        assert code == 0

    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["query"] == "intruder window"
    assert data["total_matches"] >= 1
    assert "results" in data
