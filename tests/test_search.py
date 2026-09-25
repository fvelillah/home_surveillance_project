"""Tests for Multi-Modal Semantic Video Search Engine (backend/search.py)."""

import time
from unittest.mock import MagicMock, patch
import pytest

from backend.incidents import incident_engine
from backend.models import IncidentRecord, VLMExplanation
from backend.search import (
    SemanticSearchEngine,
    _compute_text_similarity,
    _tokenize,
    search_engine,
)
from backend.twelvelabs_client import SearchResult


@pytest.fixture(autouse=True)
def clean_incident_engine():
    """Ensures incident engine is clean before each test."""
    incident_engine.clear()
    yield
    incident_engine.clear()


def test_tokenize():
    tokens = _tokenize("Find the delivery driver near front-door 123!")
    assert "find" in tokens
    assert "delivery" in tokens
    assert "driver" in tokens
    assert "front-door" in tokens
    assert "123" in tokens


def test_compute_text_similarity_matches():
    inc = IncidentRecord(
        incident_id="inc-test-1",
        channel=1,
        camera_id="cam-1",
        camera_name="Front Door",
        start_time=1000.0,
        end_time=1010.0,
        duration_s=10.0,
        peak_score=0.75,
        mean_score=0.70,
        severity=78,
        severity_badge="CRITICAL",
        status="OPEN",
        event_count=2,
        vlm_explanation=VLMExplanation(
            summary="Intruder attempting to pry open front door with crowbar.",
            actors=["intruder", "person"],
            action="prying front door with crowbar",
            objects=["crowbar", "backpack"],
            risk_assessment="breach",
            recommended_action="Dispatch security",
        ),
        created_at=1000.0,
        updated_at=1010.0,
    )

    # Actor match
    score, reason, matched = _compute_text_similarity({"intruder"}, inc)
    assert score > 0.40
    assert any("actors" in m for m in matched)

    # Object match
    score, reason, matched = _compute_text_similarity({"crowbar"}, inc)
    assert score > 0.20
    assert any("objects" in m for m in matched)

    # Action match
    score, reason, matched = _compute_text_similarity({"prying"}, inc)
    assert score > 0.30
    assert any("action" in m for m in matched)

    # Camera match
    score, reason, matched = _compute_text_similarity({"door"}, inc)
    assert score > 0.20
    assert any("camera" in m for m in matched)


def test_search_empty_database():
    engine = SemanticSearchEngine()
    res = engine.search(query="delivery driver")
    assert res.total_matches == 0
    assert len(res.results) == 0
    assert res.query == "delivery driver"
    assert res.latency_ms >= 0


def test_search_metadata_matching_and_ranking():
    engine = SemanticSearchEngine()

    # Create 3 distinct incidents
    inc1 = incident_engine.process_escalation(
        channel=1,
        camera_id="cam-1",
        camera_name="Front Door",
        edge_score=0.20,
        cloud_score=0.25,
        ensemble_score=0.22,
        timestamp=1000.0,
    )
    assert inc1 is not None
    incident_engine.attach_vlm_explanation(
        inc1.incident_id,
        VLMExplanation(
            summary="Delivery courier in uniform drops package on front porch.",
            actors=["delivery driver", "courier"],
            action="dropping package on porch",
            objects=["cardboard box", "parcel"],
            risk_assessment="routine",
            recommended_action="None",
        ),
    )

    inc2 = incident_engine.process_escalation(
        channel=2,
        camera_id="cam-2",
        camera_name="Driveway",
        edge_score=0.30,
        cloud_score=0.35,
        ensemble_score=0.32,
        timestamp=1100.0,
    )
    assert inc2 is not None
    incident_engine.attach_vlm_explanation(
        inc2.incident_id,
        VLMExplanation(
            summary="White delivery van pulls into driveway and turns around.",
            actors=["delivery van", "vehicle"],
            action="driving in driveway",
            objects=["van"],
            risk_assessment="routine",
            recommended_action="None",
        ),
    )

    inc3 = incident_engine.process_escalation(
        channel=9,
        camera_id="cam-9",
        camera_name="Perimeter Gate",
        edge_score=0.80,
        cloud_score=0.85,
        ensemble_score=0.82,
        timestamp=1200.0,
    )
    assert inc3 is not None
    incident_engine.attach_vlm_explanation(
        inc3.incident_id,
        VLMExplanation(
            summary="Masked subject climbs over perimeter fence at night.",
            actors=["masked intruder"],
            action="climbing perimeter gate",
            objects=["fence", "ladder"],
            risk_assessment="breach",
            recommended_action="Contact police",
        ),
    )

    # Search for "package delivery"
    res1 = engine.search(query="package delivery")
    assert res1.total_matches >= 1
    assert res1.results[0].incident_id == inc1.incident_id
    assert "package" in res1.results[0].summary.lower() or "courier" in res1.results[0].actors

    # Search for "intruder climbing gate"
    res2 = engine.search(query="intruder climbing gate")
    assert res2.total_matches >= 1
    assert res2.results[0].incident_id == inc3.incident_id
    assert res2.results[0].channel == 9


def test_search_spatial_and_temporal_filters():
    engine = SemanticSearchEngine()

    inc1 = incident_engine.process_escalation(
        channel=1,
        camera_id="cam-1",
        camera_name="Front Door",
        edge_score=0.30,
        cloud_score=0.30,
        ensemble_score=0.30,
        timestamp=1000.0,
    )
    inc2 = incident_engine.process_escalation(
        channel=2,
        camera_id="cam-2",
        camera_name="Driveway",
        edge_score=0.30,
        cloud_score=0.30,
        ensemble_score=0.30,
        timestamp=2000.0,
    )

    # Channel filter
    res_ch1 = engine.search(query="", channel=1)
    assert res_ch1.total_matches == 1
    assert res_ch1.results[0].channel == 1

    res_ch2 = engine.search(query="", channel=2)
    assert res_ch2.total_matches == 1
    assert res_ch2.results[0].channel == 2

    # Time filters
    res_t1 = engine.search(query="", start_time=500.0, end_time=1500.0)
    assert res_t1.total_matches == 1
    assert res_t1.results[0].incident_id == inc1.incident_id

    res_t2 = engine.search(query="", start_time=1500.0, end_time=2500.0)
    assert res_t2.total_matches == 1
    assert res_t2.results[0].incident_id == inc2.incident_id


def test_search_severity_and_status_filters():
    engine = SemanticSearchEngine()

    inc1 = incident_engine.process_escalation(
        channel=1,
        camera_id="cam-1",
        camera_name="Front Door",
        edge_score=0.10,
        cloud_score=0.10,
        ensemble_score=0.10,
        timestamp=1000.0,
    )
    inc2 = incident_engine.process_escalation(
        channel=2,
        camera_id="cam-2",
        camera_name="Driveway",
        edge_score=0.85,
        cloud_score=0.90,
        ensemble_score=0.88,
        timestamp=1100.0,
    )

    # Severity badge filter
    res_crit = engine.search(query="", severity_badge="CRITICAL")
    assert res_crit.total_matches == 1
    assert res_crit.results[0].channel == 2

    # Min severity score filter
    res_min = engine.search(query="", min_severity=50)
    assert res_min.total_matches == 1
    assert res_min.results[0].channel == 2

    # Status filter
    incident_engine.update_status(inc1.incident_id, "CLOSED")
    res_closed = engine.search(query="", status="CLOSED")
    assert res_closed.total_matches == 1
    assert res_closed.results[0].incident_id == inc1.incident_id


def test_search_pagination():
    engine = SemanticSearchEngine()

    for i in range(1, 10):
        incident_engine.process_escalation(
            channel=i,
            camera_id=f"cam-{i}",
            camera_name=f"Camera {i}",
            edge_score=0.20,
            cloud_score=0.20,
            ensemble_score=0.20,
            timestamp=1000.0 + (i * 10),
        )

    res_page1 = engine.search(query="", limit=4, offset=0)
    assert res_page1.total_matches == 9
    assert len(res_page1.results) == 4

    res_page2 = engine.search(query="", limit=4, offset=4)
    assert res_page2.total_matches == 9
    assert len(res_page2.results) == 4

    res_page3 = engine.search(query="", limit=4, offset=8)
    assert res_page3.total_matches == 9
    assert len(res_page3.results) == 1


def test_search_marengo_visual_search_fusion():
    engine = SemanticSearchEngine()

    inc = incident_engine.process_escalation(
        channel=1,
        camera_id="cam-1",
        camera_name="Front Door",
        edge_score=0.30,
        cloud_score=0.30,
        ensemble_score=0.30,
        timestamp=1000.0,
        clip_url="http://storage.local/clips/video_123.mp4",
    )
    assert inc is not None

    mock_search_results = [
        SearchResult(
            video_id="video_123",
            score=0.92,
            start=2.0,
            end=8.0,
            confidence="high",
            metadata={},
        )
    ]

    with patch("backend.twelvelabs_client.is_enabled", return_value=True):
        with patch("backend.twelvelabs_client.search_videos", return_value=mock_search_results):
            res = engine.search(query="person approaching door with box")
            assert res.total_matches >= 1
            top_match = res.results[0]
            assert top_match.incident_id == inc.incident_id
            assert "marengo_visual" in top_match.match_reason
            assert top_match.score >= 0.60
