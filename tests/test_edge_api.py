"""Tests for Edge FastAPI Endpoints and HTTP Stream Proxy."""

from pathlib import Path
import numpy as np
import pytest

try:
    from starlette.testclient import TestClient
except RuntimeError:
    TestClient = None

pytestmark = pytest.mark.skipif(TestClient is None, reason="httpx is required for TestClient")

import edge.main as main_module
from edge.config import EdgeConfig
from edge.main import app, EdgeAppState
from edge.stream_worker import FrameItem


@pytest.fixture(autouse=True)
def setup_edge_app_state(tmp_path: Path):
    """Set up isolated EdgeAppState for each test."""
    cfg = EdgeConfig()
    cfg.num_cameras = 2
    cfg.storage_dir = tmp_path / "storage"
    cfg.incident_clips_dir = tmp_path / "storage" / "incidents"
    cfg.snapshots_dir = tmp_path / "storage" / "snapshots"
    cfg.qdrant_edge_path = tmp_path / "qdrant"
    cfg.model_type = "mobilenet_v3"
    cfg.embedding_dim = 576

    # Inject app_state directly to avoid starting background network workers
    state = EdgeAppState(cfg=cfg)
    main_module.app_state = state

    # Populate dummy frame in channel 1 worker
    worker = state.capture_manager.get_worker(1)
    if worker:
        dummy_frame = np.zeros((480, 704, 3), dtype=np.uint8)
        dummy_frame[:, :, 1] = 200  # Green channel
        with worker._lock:
            worker._buffer.append(
                FrameItem(
                    frame=dummy_frame,
                    timestamp=100.0,
                    monotonic_ts=100.0,
                    frame_idx=1,
                )
            )

    yield state

    # Teardown
    state.detector.close()
    main_module.app_state = None


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "HEALTHY"
    assert "uptime_seconds" in data
    assert "cameras" in data
    assert "model" in data
    assert "detector" in data
    assert "queue" in data


def test_cameras_endpoints():
    client = TestClient(app)
    # List all cameras
    resp = client.get("/api/v1/cameras")
    assert resp.status_code == 200
    cameras = resp.json()["cameras"]
    assert len(cameras) == 2
    assert cameras[0]["channel"] == 1
    assert "stream_proxy_url" in cameras[0]

    # Single camera detail
    resp_ch = client.get("/api/v1/cameras/1")
    assert resp_ch.status_code == 200
    ch_data = resp_ch.json()
    assert ch_data["channel"] == 1
    assert "history" in ch_data

    # Unknown camera
    resp_404 = client.get("/api/v1/cameras/99")
    assert resp_404.status_code == 404


def test_scores_endpoint():
    client = TestClient(app)
    resp = client.get("/api/v1/scores")
    assert resp.status_code == 200
    data = resp.json()
    assert "triage_threshold" in data
    assert "latest_scores" in data
    assert "history" in data


def test_queue_endpoint():
    client = TestClient(app)
    resp = client.get("/api/v1/queue")
    assert resp.status_code == 200
    data = resp.json()
    assert "stats" in data
    assert "pending_count" in data


def test_snapshot_endpoint():
    client = TestClient(app)
    resp = client.get("/api/v1/stream/1/snapshot")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 100


def test_seed_baseline_endpoint():
    client = TestClient(app)
    resp = client.post(
        "/api/v1/baseline/seed",
        json={"channel": 1, "num_vectors": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SEEDED"
    assert data["vectors_seeded"] == 5


def test_trigger_triage_endpoint():
    client = TestClient(app)
    resp = client.post(
        "/api/v1/triage/trigger",
        json={"channel": 1, "score": 0.95},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "TRIGGERED"
    assert data["channel"] == 1
    assert data["score"] == 0.95
    assert "incident_id" in data


def test_websocket_stream_endpoint():
    client = TestClient(app)
    with client.websocket_connect("/api/v1/stream/1/ws") as websocket:
        data = websocket.receive_bytes()
        assert len(data) > 100
        # Check JPEG header magic bytes
        assert data[:2] == b"\xff\xd8"
