"""
Web UI Option 2 API Contract & Integration Test Suite.
Verifies all FastAPI endpoints consumed by the Next.js Cyber-Defense Web Console:
- Health & Cluster Telemetry (/health)
- Camera Stream Catalog (/api/v1/cameras)
- Consolidated Incidents & Filtering (/api/v1/incidents, /api/v1/incidents/active)
- Incident Status Transitions (/api/v1/incidents/{id}/status)
- Semantic Natural Language Video Search (/api/v1/search/semantic)
- Conversational Security Copilot Chat (/api/v1/copilot/chat)
- 24-Hour Surveillance Digest (/api/v1/digest/daily, /api/v1/digest/summary)
- Adaptive Load Shedding Controls (/api/v1/streaming/status, /api/v1/streaming/level)
- Memory Governor & Quarantine Buffer (/api/v1/memory/quarantine)
"""

import time
import pytest
from starlette.testclient import TestClient

from backend.main import app
from backend.incidents import incident_engine
from backend.models import IncidentRecord, VLMExplanation


@pytest.fixture
def client():
    return TestClient(app)


def test_ui_health_contract(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "qdrant_connected" in data
    assert "edge_devices_count" in data
    assert "uptime_s" in data


def test_ui_cameras_contract(client):
    resp = client.get("/api/v1/cameras")
    assert resp.status_code == 200
    cameras = resp.json()
    assert len(cameras) == 9
    for cam in cameras:
        assert "channel" in cam
        assert "name" in cam
        assert "status" in cam
        assert "substream_url" in cam
        assert "snapshot_url" in cam
        assert "latest_score" in cam


def test_ui_incident_lifecycle_contract(client):
    now = time.time()
    # Seed an incident into the engine with required schema fields
    inc = IncidentRecord(
        incident_id="ui-test-inc-01",
        channel=2,
        camera_id="cam-2",
        camera_name="Main Entry & Porch",
        start_time=now - 30.0,
        end_time=now,
        duration_s=30.0,
        peak_score=0.912,
        mean_score=0.850,
        severity=91,
        severity_badge="CRITICAL",
        status="OPEN",
        event_count=3,
        created_at=now - 30.0,
        updated_at=now,
        vlm_explanation=VLMExplanation(
            summary="Suspicious movement detected near main entrance porch.",
            actors=["person"],
            risk_assessment="suspicious",
        ),
    )
    incident_engine._active_incidents[f"ch_{inc.channel}"] = inc
    incident_engine._incidents.append(inc)
    incident_engine._incident_by_id["ui-test-inc-01"] = inc

    # 1. Fetch active incidents
    resp_active = client.get("/api/v1/incidents/active")
    assert resp_active.status_code == 200
    active = resp_active.json()
    assert any(i["incident_id"] == "ui-test-inc-01" for i in active)

    # 2. Fetch with channel filter
    resp_filtered = client.get("/api/v1/incidents?channel=2")
    assert resp_filtered.status_code == 200
    filtered = resp_filtered.json()
    assert any(i["incident_id"] == "ui-test-inc-01" for i in filtered)

    # 3. Update status to ACKNOWLEDGED
    resp_ack = client.post(
        "/api/v1/incidents/ui-test-inc-01/status",
        json={"status": "ACKNOWLEDGED", "notes": "Officer dispatched to investigate"},
    )
    assert resp_ack.status_code == 200
    updated = resp_ack.json()
    assert updated["status"] == "ACKNOWLEDGED"
    assert updated["notes"] == "Officer dispatched to investigate"

    # 4. Resolve / Close incident
    resp_close = client.post(
        "/api/v1/incidents/ui-test-inc-01/status",
        json={"status": "CLOSED", "notes": "Resolved: False positive (family member)"},
    )
    assert resp_close.status_code == 200
    closed = resp_close.json()
    assert closed["status"] == "CLOSED"


def test_ui_semantic_search_contract(client):
    resp = client.post(
        "/api/v1/search/semantic",
        json={
            "query": "person walking near porch",
            "threshold": "low",
            "limit": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "query" in data
    assert "total_matches" in data
    assert "results" in data
    assert "latency_ms" in data


def test_ui_copilot_chat_contract(client):
    resp = client.post(
        "/api/v1/copilot/chat",
        json={
            "message": "Are there any open critical threats?",
            "session_id": "test-ui-session",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "session_id" in data


def test_ui_daily_digest_contract(client):
    # 1. Summary endpoint
    resp_sum = client.get("/api/v1/digest/summary")
    assert resp_sum.status_code == 200
    sum_data = resp_sum.json()
    assert "threat_level" in sum_data
    assert "executive_summary" in sum_data

    # 2. Full daily digest endpoint
    resp_full = client.get("/api/v1/digest/daily?format=full")
    assert resp_full.status_code == 200
    digest = resp_full.json()
    assert "digest_id" in digest
    assert "threat_level" in digest
    assert "executive_summary" in digest


def test_ui_governance_contract(client):
    # 1. Streaming status
    resp_stream = client.get("/api/v1/streaming/status")
    assert resp_stream.status_code == 200
    stream_data = resp_stream.json()
    assert "current_level" in stream_data
    assert "level_name" in stream_data
    assert "auto_mode" in stream_data

    # 2. Change level to 1
    resp_set = client.post(
        "/api/v1/streaming/level",
        json={"level": 1, "auto_mode": False},
    )
    assert resp_set.status_code == 200
    assert resp_set.json()["current_level"] == 1

    # Reset to level 0 auto
    client.post("/api/v1/streaming/level", json={"level": 0, "auto_mode": True})

    # 3. Quarantine list
    resp_quar = client.get("/api/v1/memory/quarantine")
    assert resp_quar.status_code == 200
    assert isinstance(resp_quar.json(), list)
