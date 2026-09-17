"""Integration tests for Central Cloud FastAPI endpoints."""

import json
from unittest.mock import MagicMock, patch
import pytest
from starlette.testclient import TestClient

from backend.main import app
from backend.anomaly import index_baseline_vectors
from backend.config import config
from backend.escalation import tracker


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "qdrant_connected" in data
    assert "edge_devices_count" in data
    assert data["edge_devices_count"] >= 9  # 9 Dahua channels initialized


def test_cameras_catalog(client):
    resp = client.get("/api/v1/cameras")
    assert resp.status_code == 200
    cameras = resp.json()
    assert len(cameras) == 9

    ch1 = next(c for c in cameras if c["channel"] == 1)
    expected_name = config.parse_camera_names()[1]
    assert ch1["name"] == expected_name
    assert "/api/v1/stream/1/live" in ch1["substream_url"]
    assert "/api/v1/stream/1/snapshot" in ch1["snapshot_url"]


def test_baseline_index_and_score(client):
    # 1. Index 2 baseline vectors
    vec = [1.0, 0.0, 0.0, 0.0]
    resp_index = client.post(
        "/api/v1/baseline/index",
        json={
            "vectors": [vec, vec],
            "metadata": [{"camera_id": "cam-1"}, {"camera_id": "cam-1"}],
        },
    )
    assert resp_index.status_code == 200
    assert resp_index.json()["status"] == "ok"

    # 2. Score vector matching baseline
    resp_score = client.post(
        "/api/v1/score",
        json={
            "embedding": [1.0, 0.0, 0.0, 0.0],
            "camera_id": "cam-1",
            "k": 2,
        },
    )
    assert resp_score.status_code == 200
    score_data = resp_score.json()
    assert pytest.approx(score_data["anomaly_score"], 1e-4) == 0.0
    assert score_data["is_anomaly"] is False
    assert len(score_data["neighbors"]) >= 1


def test_qdrant_stats(client):
    resp = client.get("/api/v1/qdrant/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "collection" in data
    assert "points_count" in data


def test_escalate_json_payload(client):
    tracker.clear()
    index_baseline_vectors(
        vectors=[[1.0, 0.0, 0.0, 0.0]],
        metadata_list=[{"camera_id": "cam-1"}],
    )

    # Submit an anomalous escalation via JSON
    resp = client.post(
        "/api/v1/escalate",
        json={
            "edge_device_id": "cam-1",
            "edge_score": 0.22,
            "edge_embedding": [0.0, 1.0, 0.0, 0.0],
            "timestamp_ms": 1710000000000,
            "scene_id": "front_gate",
            "camera_id": "cam-1",
            "channel": 1,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "escalation_id" in data
    assert data["edge_device_id"] == "cam-1"
    assert data["cloud_score"] >= 0.8
    assert data["is_confirmed_anomaly"] is True
    assert data["incident_id"] is not None


def test_escalate_form_payload(client):
    tracker.clear()
    resp = client.post(
        "/api/v1/escalate",
        data={
            "edge_device_id": "cam-2",
            "edge_score": "0.18",
            "edge_embedding": json.dumps([0.0, 0.0, 1.0, 0.0]),
            "timestamp_ms": "1710000005000",
            "camera_id": "cam-2",
            "channel": "2",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["edge_device_id"] == "cam-2"
    assert "ensemble_score" in data


def test_escalations_stats_endpoint(client):
    resp = client.get("/api/v1/escalations/stats")
    assert resp.status_code == 200
    stats = resp.json()
    assert "escalation_count" in stats
    assert "confirmation_rate" in stats


def test_edges_endpoints(client):
    # 1. List edges
    resp = client.get("/api/v1/edges")
    assert resp.status_code == 200
    devices = resp.json()
    assert len(devices) >= 9

    # 2. Register custom edge node
    resp_reg = client.post(
        "/api/v1/edges/register",
        json={
            "name": "Auxiliary Edge Worker",
            "location": "North Outpost",
            "device_id": "edge-aux-north",
            "threshold": 0.12,
        },
    )
    assert resp_reg.status_code == 200
    reg_data = resp_reg.json()
    assert reg_data["device_id"] == "edge-aux-north"
    assert reg_data["threshold"] == 0.12

    # 3. Get device stats
    resp_stats = client.get("/api/v1/edges/edge-aux-north/stats")
    assert resp_stats.status_code == 200
    stats_data = resp_stats.json()
    assert stats_data["device_id"] == "edge-aux-north"
    assert "escalation_details" in stats_data


def test_twelvelabs_status_endpoint(client):
    resp = client.get("/api/v1/twelvelabs/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "enabled" in data
    assert "marengo_model" in data
    assert "pegasus_model" in data


def test_incidents_api_lifecycle(client):
    # 1. Trigger an escalation that creates an incident
    resp_esc = client.post(
        "/api/v1/escalate",
        json={
            "edge_device_id": "cam-1",
            "edge_score": 0.35,
            "edge_embedding": [0.0, 1.0, 0.0, 0.0],
            "timestamp_ms": 1710000000000,
            "scene_id": "front_porch",
            "camera_id": "cam-1",
            "channel": 1,
        },
    )
    assert resp_esc.status_code == 200
    esc_data = resp_esc.json()
    inc_id = esc_data.get("incident_id")
    assert inc_id is not None

    # 2. List incidents
    resp_list = client.get("/api/v1/incidents")
    assert resp_list.status_code == 200
    inc_list = resp_list.json()
    assert len(inc_list) >= 1
    assert any(i["incident_id"] == inc_id for i in inc_list)

    # 3. Get active incidents
    resp_act = client.get("/api/v1/incidents/active")
    assert resp_act.status_code == 200

    # 4. Get specific incident
    resp_single = client.get(f"/api/v1/incidents/{inc_id}")
    assert resp_single.status_code == 200
    single_data = resp_single.json()
    assert single_data["incident_id"] == inc_id
    assert single_data["channel"] == 1
    assert "severity" in single_data
    assert "severity_badge" in single_data

    # 5. Update incident status
    resp_status = client.post(
        f"/api/v1/incidents/{inc_id}/status",
        json={"status": "ACKNOWLEDGED", "notes": "Verified by operator"},
    )
    assert resp_status.status_code == 200
    assert resp_status.json()["status"] == "ACKNOWLEDGED"

    # 6. Trigger explain incident without Pegasus enabled -> 503
    with patch("backend.twelvelabs_client.is_enabled", return_value=False):
        resp_disabled = client.post(f"/api/v1/incidents/{inc_id}/explain")
        assert resp_disabled.status_code == 503

    # 7. Trigger explain incident with Pegasus enabled and mocked
    mock_analysis = MagicMock()
    mock_analysis.text = json.dumps({
        "summary": "Front porch activity verified by Pegasus.",
        "actors": ["visitor"],
        "action": "approaching doorway",
        "objects": ["doorbell"],
        "risk_assessment": "routine",
        "recommended_action": "No action needed",
    })
    with patch("backend.twelvelabs_client.is_enabled", return_value=True), \
         patch("backend.twelvelabs_client.analyze_video", return_value=mock_analysis):
        resp_expl = client.post(f"/api/v1/incidents/{inc_id}/explain?video_id=vid-pegasus-api-123")
        assert resp_expl.status_code == 200
        expl_data = resp_expl.json()
        assert expl_data.get("vlm_explanation") is not None
        assert expl_data["vlm_explanation"]["model"] == config.pegasus_model
        assert expl_data["vlm_explanation"]["summary"] == "Front porch activity verified by Pegasus."


def test_memory_governor_api(client):
    # 1. Stage vector to quarantine
    resp_stage = client.post(
        "/api/v1/memory/quarantine/stage",
        params={
            "camera_id": "cam-1",
            "channel": 1,
            "anomaly_score": 0.02,
        },
        json=[1.0, 0.0, 0.0, 0.0],
    )
    assert resp_stage.status_code == 200
    stage_data = resp_stage.json()
    assert stage_data["status"] == "ok"
    vec_id = stage_data["item"]["vector_id"]

    # 2. List quarantine
    resp_qlist = client.get("/api/v1/memory/quarantine")
    assert resp_qlist.status_code == 200
    qitems = resp_qlist.json()
    assert len(qitems) >= 1

    # 3. Promote quarantine
    resp_prom = client.post(
        "/api/v1/memory/quarantine/promote",
        json={"vector_ids": [vec_id], "force": True},
    )
    assert resp_prom.status_code == 200
    assert resp_prom.json()["promoted_count"] >= 1

    # 4. Memory stats
    resp_stats = client.get("/api/v1/memory/stats")
    assert resp_stats.status_code == 200
    assert "total_baseline_points" in resp_stats.json()

    # 5. Retention scrub
    resp_scrub = client.post("/api/v1/memory/scrub", params={"retention_days": 7})
    assert resp_scrub.status_code == 200
    assert resp_scrub.json()["status"] == "ok"


def test_streaming_backpressure_api(client):
    # 1. Get status
    resp_stat = client.get("/api/v1/streaming/status")
    assert resp_stat.status_code == 200
    stat_data = resp_stat.json()
    assert "current_level" in stat_data
    assert "level_name" in stat_data

    # 2. Set manual level override
    resp_set = client.post(
        "/api/v1/streaming/level",
        json={"level": 1, "auto_mode": False},
    )
    assert resp_set.status_code == 200
    set_data = resp_set.json()
    assert set_data["current_level"] == 1
    assert set_data["level_name"] == "SCORE_ONLY"

    # 3. Restore auto mode
    resp_auto = client.post(
        "/api/v1/streaming/level",
        json={"auto_mode": True},
    )
    assert resp_auto.status_code == 200
    assert resp_auto.json()["auto_mode"] is True

