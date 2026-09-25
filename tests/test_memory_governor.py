"""Unit tests for Memory Governor & Anti-Poisoning Engine with Qdrant Staging."""

import time
from unittest.mock import MagicMock, patch
import pytest

from backend.memory import MemoryGovernor, _to_point_id


@pytest.fixture
def governor():
    gov = MemoryGovernor(
        quarantine_collection_name="test_anomaly_quarantine",
        quarantine_duration_s=3600,
        max_vectors_per_camera=500,
        retention_days=7,
        anti_poisoning_threshold=0.080,
    )
    gov.clear()
    return gov


def test_to_point_id():
    pid1 = _to_point_id("qvec-1234567890ab")
    pid2 = _to_point_id("qvec-1234567890ab")
    assert pid1 == pid2
    # Valid standard UUID
    assert len(pid1) == 36
    assert "-" in pid1


def test_validate_candidate_rejections(governor):
    # 1. Empty vector
    v1, r1 = governor.validate_candidate([], "cam-1", 0.02)
    assert not v1
    assert "empty" in r1

    # 2. Zero vector
    v2, r2 = governor.validate_candidate([0.0, 0.0, 0.0, 0.0], "cam-1", 0.02)
    assert not v2
    assert "zero" in r2

    # 3. Anomaly score >= threshold (Anti-poisoning trigger)
    v3, r3 = governor.validate_candidate([1.0, 0.0, 0.0, 0.0], "cam-1", 0.15)
    assert not v3
    assert "Anti-poisoning reject" in r3

    # 4. Valid normal candidate
    v4, r4 = governor.validate_candidate([1.0, 0.0, 0.0, 0.0], "cam-1", 0.02)
    assert v4
    assert "Valid" in r4


def test_stage_to_quarantine_in_memory(governor):
    with patch("backend.memory.get_qdrant", return_value=None):
        item, msg = governor.stage_to_quarantine(
            vector=[0.5, 0.5, 0.5, 0.5],
            camera_id="cam-1",
            channel=1,
            anomaly_score=0.03,
            metadata={"scene": "driveway_normal"},
        )
        assert item is not None
        assert item.camera_id == "cam-1"
        assert item.status == "QUARANTINED"
        assert item.expires_at > item.created_at

        quarantine_list = governor.list_quarantine()
        assert len(quarantine_list) == 1
        assert quarantine_list[0].vector_id == item.vector_id


def test_stage_to_quarantine_qdrant_persisted(governor):
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    with patch("backend.memory.get_qdrant", return_value=mock_client):
        item, msg = governor.stage_to_quarantine(
            vector=[0.5, 0.5, 0.5, 0.5],
            camera_id="cam-2",
            channel=2,
            anomaly_score=0.02,
        )
        assert item is not None
        assert msg == "Staged to quarantine"

        # Verify collection created and point upserted
        mock_client.create_collection.assert_called_once()
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == "test_anomaly_quarantine"
        points = call_args[1]["points"]
        assert len(points) == 1
        assert points[0].payload["camera_id"] == "cam-2"
        assert points[0].payload["status"] == "QUARANTINED"


def test_promote_quarantined_vectors(governor):
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    with patch("backend.memory.get_qdrant", return_value=mock_client):
        item, _ = governor.stage_to_quarantine(
            vector=[1.0, 0.0, 0.0, 0.0],
            camera_id="cam-1",
            channel=1,
            anomaly_score=0.02,
        )
        assert item is not None

        # Force promote immediately
        promoted_count = governor.promote_quarantined_vectors(
            vector_ids=[item.vector_id],
            force=True,
        )
        assert promoted_count >= 1

        # Check status in memory and that set_payload was called in Qdrant
        items = governor.list_quarantine(status="PROMOTED")
        assert len(items) == 1
        assert items[0].status == "PROMOTED"
        mock_client.set_payload.assert_called_once()


def test_scrub_retention(governor):
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    with patch("backend.memory.get_qdrant", return_value=mock_client):
        # Stage item with old timestamp
        item, _ = governor.stage_to_quarantine(
            vector=[0.0, 1.0, 0.0, 0.0],
            camera_id="cam-2",
            channel=2,
            anomaly_score=0.01,
        )
        assert item is not None

        # Artificially age created_at
        item.created_at = time.time() - (10 * 86400.0)

        # Scrub with retention 7 days
        res = governor.scrub_retention(retention_days=7)
        assert res["purged_quarantine_items"] == 1
        assert len(governor.list_quarantine()) == 0
        mock_client.delete.assert_called_once()


def test_memory_stats(governor):
    with patch("backend.memory.get_qdrant", return_value=None):
        governor.stage_to_quarantine(
            vector=[1.0, 0.0, 0.0, 0.0],
            camera_id="cam-3",
            channel=3,
            anomaly_score=0.04,
        )
        stats = governor.get_stats()
        assert stats.total_quarantined == 1
        assert stats.quarantine_by_camera.get("cam-3") == 1
        assert stats.max_vectors_per_camera == 500
        assert stats.retention_days == 7

