"""
Sliding Window Video Segmenter.

Extracts fixed-duration, overlapping video clips from the stream worker's rolling ring buffer,
applies uniform temporal subsampling and spatial resizing, and produces normalized tensors
ready for edge feature extraction (ONNX / VideoMAE).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from edge.config import EdgeConfig, config as default_config
from edge.stream_worker import FrameItem, StreamWorker

logger = logging.getLogger(__name__)


@dataclass
class VideoClip:
    """A segmented, preprocessed video clip ready for edge embedding inference."""
    channel: int
    camera_name: str
    frames_rgb: np.ndarray  # Shape: (T, H, W, C) uint8 or float32
    start_time: float
    end_time: float
    duration_s: float
    raw_frame_count: int
    sampled_frame_count: int

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.frames_rgb.shape

    def to_channel_first(self) -> np.ndarray:
        """Convert from (T, H, W, C) to (T, C, H, W) format."""
        return np.transpose(self.frames_rgb, (0, 3, 1, 2))

    def to_normalized_tensor(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> np.ndarray:
        """
        Normalize RGB frames to float32 in [0, 1] and apply ImageNet mean/std.
        Returns shape (1, T, C, H, W) or (T, C, H, W) float32 array.
        """
        arr = self.frames_rgb.astype(np.float32) / 255.0
        # Subtract mean and divide by std
        mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3)
        std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)
        norm_arr = (arr - mean_arr) / std_arr
        # Transpose to (T, C, H, W)
        return np.transpose(norm_arr, (0, 3, 1, 2))


class SlidingWindowSegmenter:
    """
    Sliding window clip extractor that polls stream ring buffers.
    
    Default Configuration: 
      - Window duration: 10.0 seconds
      - Overlap: 2.0 seconds (yields a new clip every 8.0s)
      - Sample frames: 16 uniformly distributed frames
      - Target resolution: 224x224 RGB
    """

    def __init__(
        self,
        clip_duration_s: Optional[float] = None,
        clip_overlap_s: Optional[float] = None,
        sample_frames: Optional[int] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        cfg: Optional[EdgeConfig] = None,
    ) -> None:
        self.config = cfg or default_config
        self.clip_duration_s = clip_duration_s if clip_duration_s is not None else self.config.clip_duration_s
        self.clip_overlap_s = clip_overlap_s if clip_overlap_s is not None else self.config.clip_overlap_s
        self.sample_frames = sample_frames if sample_frames is not None else self.config.segment_sample_frames
        self.target_width = target_width if target_width is not None else self.config.segment_target_width
        self.target_height = target_height if target_height is not None else self.config.segment_target_height

        # Track the last extracted end-time per channel to maintain step pacing
        self._last_clip_end_times: dict[int, float] = {}

    @property
    def step_interval_s(self) -> float:
        """Interval between successive clip extractions (duration - overlap)."""
        return max(self.clip_duration_s - self.clip_overlap_s, 1.0)

    def extract_clip_from_frames(
        self,
        frames: List[FrameItem],
        channel: int,
        camera_name: str = "",
    ) -> Optional[VideoClip]:
        """
        Process a list of FrameItems into a uniform VideoClip tensor.
        Requires at least `self.sample_frames` in the buffer.
        """
        if not frames or len(frames) < self.sample_frames:
            return None

        # Filter frames within window duration from the latest timestamp
        latest_ts = frames[-1].timestamp
        cutoff_ts = latest_ts - self.clip_duration_s
        window_frames = [f for f in frames if f.timestamp >= cutoff_ts]

        if len(window_frames) < self.sample_frames:
            window_frames = frames[-self.sample_frames:]

        # Uniform temporal sampling of frame indices
        num_avail = len(window_frames)
        sample_indices = np.linspace(0, num_avail - 1, self.sample_frames, dtype=int)

        sampled_rgb: List[np.ndarray] = []
        for idx in sample_indices:
            item = window_frames[idx]
            raw_bgr = item.frame

            # Resize if dimensions differ from target
            h, w = raw_bgr.shape[:2]
            if (w, h) != (self.target_width, self.target_height):
                resized = cv2.resize(
                    raw_bgr,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                resized = raw_bgr

            # BGR to RGB conversion
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            sampled_rgb.append(rgb)

        tensor_rgb = np.stack(sampled_rgb, axis=0)  # Shape: (T, H, W, C)

        return VideoClip(
            channel=channel,
            camera_name=camera_name or f"Camera {channel}",
            frames_rgb=tensor_rgb,
            start_time=window_frames[0].timestamp,
            end_time=window_frames[-1].timestamp,
            duration_s=window_frames[-1].timestamp - window_frames[0].timestamp,
            raw_frame_count=len(window_frames),
            sampled_frame_count=self.sample_frames,
        )

    def extract_latest_clip(self, worker: StreamWorker) -> Optional[VideoClip]:
        """Extract the most recent sliding window clip from a StreamWorker."""
        frames = worker.get_recent_frames(duration_s=self.clip_duration_s)
        return self.extract_clip_from_frames(
            frames=frames,
            channel=worker.channel,
            camera_name=worker.camera_name,
        )

    def should_extract_new_clip(self, channel: int, current_time: Optional[float] = None) -> bool:
        """Check if enough time has passed to extract a new sliding window step."""
        now = current_time or time.time()
        last_time = self._last_clip_end_times.get(channel, 0.0)
        return (now - last_time) >= self.step_interval_s

    def mark_clip_extracted(self, channel: int, clip_end_time: float) -> None:
        """Update last extraction timestamp for the channel."""
        self._last_clip_end_times[channel] = clip_end_time


