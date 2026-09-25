"""Tests for synthetic feed simulator."""

import time
import cv2
import numpy as np
import pytest

from scripts.simulate_dahua_feeds import DahuaFeedSimulator


def test_feed_simulator_frame_generation():
    sim = DahuaFeedSimulator(num_channels=9)

    now = time.time()
    # Test sub-stream (subtype=1)
    sub_frame = sim.generate_frame(channel=1, subtype=1, t=now)
    assert isinstance(sub_frame, np.ndarray)
    assert sub_frame.shape == (480, 704, 3)

    # Test main-stream (subtype=0)
    main_frame = sim.generate_frame(channel=2, subtype=0, t=now)
    assert isinstance(main_frame, np.ndarray)
    assert main_frame.shape == (1296, 2304, 3)


def test_feed_simulator_jpeg_encoding():
    sim = DahuaFeedSimulator()
    frame = sim.generate_frame(channel=1, subtype=1, t=time.time())
    jpeg = sim.encode_jpeg(frame, quality=80)

    assert isinstance(jpeg, bytes)
    assert len(jpeg) > 1000
    # Check JPEG magic bytes: 0xFF, 0xD8
    assert jpeg[:2] == b"\xff\xd8"
