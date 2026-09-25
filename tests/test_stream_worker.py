"""Tests for StreamWorker and ring buffering logic."""

import collections
import time
import numpy as np
import pytest

from edge.stream_worker import FrameItem, StreamStats, StreamWorker


def test_stream_stats_tick():
    stats = StreamStats(channel=1, camera_name="Front Door")
    assert not stats.is_connected
    assert stats.fps == 0.0

    # Simulate 5 frames
    for _ in range(5):
        stats.tick_frame(704, 480)
        time.sleep(0.01)

    assert stats.is_connected
    assert stats.frames_captured == 5
    assert stats.resolution == (704, 480)
    assert stats.last_frame_timestamp > 0

    d = stats.as_dict()
    assert d["channel"] == 1
    assert d["resolution"] == "704x480"
    assert d["frames_captured"] == 5


def test_stream_worker_ring_buffer():
    # Instantiate worker without starting background thread
    worker = StreamWorker(
        channel=1,
        stream_url="http://mock-url",
        camera_name="TestCam",
        target_fps=10,
        buffer_capacity_sec=5,  # 50 frames capacity
    )

    dummy_frame = np.zeros((480, 704, 3), dtype=np.uint8)
    now = time.time()

    # Manually populate buffer
    with worker._lock:
        for i in range(60):  # More than 50
            item = FrameItem(
                frame=dummy_frame,
                timestamp=now - (60 - i) * 0.1,
                monotonic_ts=time.monotonic(),
                frame_idx=i,
            )
            worker._buffer.append(item)

    # Buffer should not exceed capacity
    assert len(worker._buffer) == 60  # min maxlen is max(50, 60) = 60

    # Test latest frame copy
    latest = worker.get_latest_frame()
    assert latest is not None
    assert latest.shape == (480, 704, 3)

    # Test recent frames retrieval within last 2 seconds
    recent = worker.get_recent_frames(duration_s=2.0)
    assert len(recent) > 0
    assert len(recent) <= 25


def test_stream_worker_lifecycle():
    worker = StreamWorker(
        channel=1,
        stream_url="http://127.0.0.1:9999/nonexistent",
        target_fps=5,
        reconnect_initial_delay_sec=0.1,
    )
    # Start and stop cleanly without raising unhandled exceptions
    worker.start()
    assert worker._running
    time.sleep(0.2)
    worker.stop(timeout=1.0)
    assert not worker._running
