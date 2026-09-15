"""Integration tests for Central Cloud FastAPI endpoints."""

import json
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
