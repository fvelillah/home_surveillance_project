"""Tests for CaptureManager multi-channel coordination."""

import time
from pathlib import Path
import numpy as np
import pytest

from edge.config import EdgeConfig
from edge.capture_manager import CaptureManager
from edge.stream_worker import FrameItem


def test_capture_manager_init():
    cfg = EdgeConfig()
    cfg.num_cameras = 9
    mgr = CaptureManager(cfg=cfg)

    assert len(mgr.workers) == 9
    for ch in range(1, 10):
        worker = mgr.get_worker(ch)
        assert worker is not None
        assert worker.channel == ch


def test_capture_manager_all_stats():
    cfg = EdgeConfig()
    cfg.num_cameras = 4
    mgr = CaptureManager(cfg=cfg)

    stats = mgr.get_all_stats()
    assert len(stats) == 4
    for s in stats:
        assert "channel" in s
        assert "is_connected" in s
        assert "fps" in s


def test_capture_manager_fallback_clip_generation(tmp_path: Path):
    cfg = EdgeConfig()
    cfg.num_cameras = 2
    cfg.incident_clips_dir = tmp_path
    mgr = CaptureManager(cfg=cfg)

    worker = mgr.get_worker(1)
    assert worker is not None

    # Inject dummy frames into worker buffer
    dummy_frame = np.zeros((480, 704, 3), dtype=np.uint8)
    now = time.time()
    with worker._lock:
        for i in range(25):
            item = FrameItem(
                frame=dummy_frame,
                timestamp=now - (25 - i) * 0.1,
                monotonic_ts=time.monotonic(),
                frame_idx=i,
            )
            worker._buffer.append(item)

    out_file = tmp_path / "test_incident.mp4"
    # Will fail main-stream connection (dummy URL) and fall back to buffered frames
    saved_path = mgr.capture_incident_clip(
        channel=1,
        duration_s=2.0,
        output_path=out_file,
        target_fps=10,
    )

    assert saved_path is not None
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0
