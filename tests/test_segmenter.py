"""Tests for SlidingWindowSegmenter."""

import time
import numpy as np
import pytest

from edge.segmenter import SlidingWindowSegmenter, VideoClip
from edge.stream_worker import FrameItem


def test_segmenter_extract_clip():
    segmenter = SlidingWindowSegmenter(
        clip_duration_s=10.0,
        clip_overlap_s=2.0,
        sample_frames=16,
        target_width=224,
        target_height=224,
    )

    assert segmenter.step_interval_s == 8.0

    # Insufficient frames -> returns None
    assert segmenter.extract_clip_from_frames([], channel=1) is None

    # Generate 50 dummy frames (5 seconds at 10 fps)
    dummy_bgr = np.zeros((480, 704, 3), dtype=np.uint8)
    dummy_bgr[:, :, 0] = 255  # Blue channel in BGR
    now = time.time()

    frames = [
        FrameItem(
            frame=dummy_bgr.copy(),
            timestamp=now - (50 - i) * 0.1,
            monotonic_ts=time.monotonic(),
            frame_idx=i,
        )
        for i in range(50)
    ]

    clip = segmenter.extract_clip_from_frames(frames, channel=1, camera_name="Front Door")
    assert clip is not None
    assert isinstance(clip, VideoClip)
    assert clip.channel == 1
    assert clip.camera_name == "Front Door"
    assert clip.sampled_frame_count == 16

    # Verify tensor shape: (T=16, H=224, W=224, C=3)
    assert clip.frames_rgb.shape == (16, 224, 224, 3)

    # Verify RGB conversion (Blue BGR [255, 0, 0] -> Red channel 0 in RGB is 0, Blue channel 2 is 255)
    assert clip.frames_rgb[0, 0, 0, 2] == 255
    assert clip.frames_rgb[0, 0, 0, 0] == 0

    # Test channel-first transposition
    ch_first = clip.to_channel_first()
    assert ch_first.shape == (16, 3, 224, 224)

    # Test normalized tensor
    norm_tensor = clip.to_normalized_tensor()
    assert norm_tensor.shape == (16, 3, 224, 224)
    assert norm_tensor.dtype == np.float32


def test_segmenter_pacing():
    segmenter = SlidingWindowSegmenter(clip_duration_s=10.0, clip_overlap_s=2.0)
    # step_interval_s = 8.0

    t0 = 1000.0
    assert segmenter.should_extract_new_clip(channel=1, current_time=t0)

    segmenter.mark_clip_extracted(channel=1, clip_end_time=t0)

    # 5 seconds later -> should not extract yet
    assert not segmenter.should_extract_new_clip(channel=1, current_time=t0 + 5.0)

    # 8.5 seconds later -> ready for next extraction
    assert segmenter.should_extract_new_clip(channel=1, current_time=t0 + 8.5)
