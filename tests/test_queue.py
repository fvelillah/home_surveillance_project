"""Tests for SQLiteOfflineQueue and AutoDrainWorker."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from edge.queue import SQLiteOfflineQueue, IncidentItem, AutoDrainWorker


def test_sqlite_queue_enqueue_and_peek(tmp_path: Path):
    db_file = tmp_path / "test_queue.db"
    queue = SQLiteOfflineQueue(db_path=db_file)

    assert queue.db_path.exists()
    assert queue.get_stats()["total"] == 0

    item = IncidentItem(
        channel=1,
        camera_name="Front Door",
        timestamp=time.time(),
        score=0.88,
        clip_path="/tmp/test_clip.mp4",
        snapshot_path="/tmp/test_snap.jpg",
        embedding=[0.1, 0.2, 0.3],
    )

    success = queue.enqueue(item)
    assert success

    stats = queue.get_stats()
    assert stats["total"] == 1
    assert stats["pending"] == 1

    pending = queue.peek_pending(limit=10)
    assert len(pending) == 1
    p = pending[0]
    assert p.incident_id == item.incident_id
    assert p.channel == 1
    assert p.score == 0.88
    assert p.status == "PENDING"
    assert p.embedding == [0.1, 0.2, 0.3]


def test_sqlite_queue_state_transitions(tmp_path: Path):
    queue = SQLiteOfflineQueue(db_path=tmp_path / "state_test.db")

    item = IncidentItem(
        channel=2,
        camera_name="Driveway",
        timestamp=time.time(),
        score=0.75,
        clip_path="/tmp/clip2.mp4",
    )
    queue.enqueue(item)

    # 1. Mark syncing
    queue.mark_syncing([item.incident_id])
    assert queue.get_stats()["syncing"] == 1

    # 2. Mark synced
    queue.mark_synced(item.incident_id)
    stats = queue.get_stats()
    assert stats["synced"] == 1
    assert stats["pending"] == 0
    assert stats["syncing"] == 0


def test_sqlite_queue_retry_and_fail(tmp_path: Path):
    queue = SQLiteOfflineQueue(db_path=tmp_path / "retry_test.db")

    item = IncidentItem(
        channel=3,
        camera_name="Backyard",
        timestamp=time.time(),
        score=0.92,
        clip_path="/tmp/clip3.mp4",
    )
    queue.enqueue(item)

    # Fail once (retry 1 < max 3) -> should revert to PENDING with retry_count=1
    queue.mark_failed(item.incident_id, "Connection refused", max_retries=3)
    pending = queue.peek_pending()
    assert len(pending) == 1
    assert pending[0].retry_count == 1
    assert pending[0].status == "PENDING"

    # Fail 2 more times -> status should transition to FAILED
    queue.mark_failed(item.incident_id, "Connection refused", max_retries=3)
    queue.mark_failed(item.incident_id, "Connection refused", max_retries=3)
    stats = queue.get_stats()
    assert stats["failed"] == 1
    assert stats["pending"] == 0


def test_sqlite_queue_crash_resilience(tmp_path: Path):
    """Verify that incidents remain persisted when a new queue instance opens the DB."""
    db_file = tmp_path / "durability.db"

    # Instance 1 enqueues item
    q1 = SQLiteOfflineQueue(db_path=db_file)
    item = IncidentItem(
        channel=4,
        camera_name="Garage",
        timestamp=12345.67,
        score=0.99,
        clip_path="/path/to/evidence.mp4",
    )
    q1.enqueue(item)
    del q1

    # Instance 2 reads from same DB file
    q2 = SQLiteOfflineQueue(db_path=db_file)
    pending = q2.peek_pending()
    assert len(pending) == 1
    assert pending[0].incident_id == item.incident_id
    assert pending[0].channel == 4
    assert pending[0].score == 0.99


def test_auto_drain_worker_sync(tmp_path: Path):
    queue = SQLiteOfflineQueue(db_path=tmp_path / "drain_test.db")
    item = IncidentItem(
        channel=1,
        camera_name="Front Door",
        timestamp=time.time(),
        score=0.80,
        clip_path="/tmp/clip.mp4",
    )
    queue.enqueue(item)

    drain_worker = AutoDrainWorker(
        queue=queue,
        central_url="http://mock-cloud:9876",
        sync_interval_s=0.1,
    )

    # Mock requests.post to simulate successful HTTP 200 response from central cloud
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("requests.post", return_value=mock_resp) as mock_post:
        synced = drain_worker.drain_once()
        assert synced == 1
        assert mock_post.called
        assert "/api/v1/escalate" in mock_post.call_args[0][0]

    stats = queue.get_stats()
    assert stats["synced"] == 1
    assert stats["pending"] == 0
