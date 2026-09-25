"""
Dahua Direct HTTP Video Stream Worker.

Connects directly to Dahua's native HTTP video engine (/cgi-bin/mjpg/video.cgi),
manages frame acquisition, ring-buffered storage, connection recovery with
exponential backoff, and live FPS/latency telemetry.
"""

from __future__ import annotations

import collections
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

# Suppress FFmpeg boundary noise during HTTP MJPEG handshake
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

logger = logging.getLogger(__name__)


@dataclass
class FrameItem:
    """A single captured video frame with precise timestamp metadata."""
    frame: np.ndarray
    timestamp: float  # Epoch timestamp (time.time())
    monotonic_ts: float  # Monotonic timestamp (time.monotonic())
    frame_idx: int


@dataclass
class StreamStats:
    """Real-time performance and health metrics for a camera stream."""
    channel: int
    camera_name: str = ""
    is_connected: bool = False
    fps: float = 0.0
    latency_ms: float = 0.0
    resolution: Tuple[int, int] = (0, 0)  # (width, height)
    frames_captured: int = 0
    dropped_frames: int = 0
    reconnect_count: int = 0
    last_frame_timestamp: float = 0.0
    _frame_times: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=60), repr=False)

    def tick_frame(self, frame_w: int, frame_h: int) -> None:
        now = time.monotonic()
        self._frame_times.append(now)
        self.frames_captured += 1
        self.last_frame_timestamp = time.time()
        self.resolution = (frame_w, frame_h)
        self.is_connected = True

        if len(self._frame_times) > 1:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self.fps = round((len(self._frame_times) - 1) / elapsed, 1)

    def mark_disconnected(self) -> None:
        self.is_connected = False
        self.fps = 0.0

    def as_dict(self) -> dict:
        return {
            "channel": self.channel,
            "camera_name": self.camera_name,
            "is_connected": self.is_connected,
            "fps": self.fps,
            "latency_ms": round(self.latency_ms, 1),
            "resolution": f"{self.resolution[0]}x{self.resolution[1]}" if self.resolution[0] > 0 else "N/A",
            "frames_captured": self.frames_captured,
            "dropped_frames": self.dropped_frames,
            "reconnect_count": self.reconnect_count,
            "last_frame_timestamp": self.last_frame_timestamp,
        }


class StreamWorker:
    """
    Dedicated worker thread handling HTTP video ingestion for a single camera channel.
    
    Features:
      - Direct HTTP multipart / MJPEG stream parsing
      - Mid-stream JPEG boundary synchronization
      - Sliding ring-buffer of recent frames
      - Automatic exponential-backoff reconnect
      - Non-blocking frame access for downstream segmenters
    """

    def __init__(
        self,
        channel: int,
        stream_url: str,
        camera_name: str = "",
        target_fps: int = 10,
        buffer_capacity_sec: int = 30,
        reconnect_initial_delay_sec: float = 1.0,
        reconnect_max_delay_sec: float = 30.0,
        stream_read_timeout_sec: float = 5.0,
    ) -> None:
        self.channel = channel
        self.stream_url = stream_url
        self.camera_name = camera_name or f"Camera {channel}"
        self.target_fps = target_fps
        self.buffer_capacity_sec = buffer_capacity_sec
        self.reconnect_initial_delay = reconnect_initial_delay_sec
        self.reconnect_max_delay = reconnect_max_delay_sec
        self.stream_read_timeout = stream_read_timeout_sec

        # Ring buffer capacity
        max_buffer_frames = max(target_fps * buffer_capacity_sec, 60)
        self._buffer: Deque[FrameItem] = collections.deque(maxlen=max_buffer_frames)
        self._lock = threading.Lock()

        # Telemetry
        self.stats = StreamStats(channel=channel, camera_name=self.camera_name)

        # Threading state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_counter = 0

    # ----------------------------------------------------------------------
    # Lifecycle Management
    # ----------------------------------------------------------------------

    def start(self) -> None:
        """Start the background stream worker thread."""
        with self._lock:
            if self._running:
                logger.warning("StreamWorker for channel %d already running", self.channel)
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._worker_loop,
                name=f"StreamWorker-Ch{self.channel}",
                daemon=True,
            )
            self._thread.start()
            logger.info("StreamWorker started for Channel %d (%s)", self.channel, self.camera_name)

    def stop(self, timeout: float = 3.0) -> None:
        """Gracefully stop the worker thread and release video capture."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            self._thread = None

        with self._lock:
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self.stats.mark_disconnected()

        logger.info("StreamWorker stopped for Channel %d", self.channel)

    def __enter__(self) -> StreamWorker:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ----------------------------------------------------------------------
    # Frame Access API
    # ----------------------------------------------------------------------

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the most recently captured video frame."""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1].frame.copy()

    def get_recent_frames(self, duration_s: float) -> List[FrameItem]:
        """
        Return all frames captured within the last `duration_s` seconds,
        ordered chronologically.
        """
        now = time.time()
        cutoff = now - duration_s
        with self._lock:
            return [
                item for item in self._buffer
                if item.timestamp >= cutoff
            ]

    def get_all_buffered_frames(self) -> List[FrameItem]:
        """Return a snapshot list of all frames currently in the ring buffer."""
        with self._lock:
            return list(self._buffer)

    def get_stats(self) -> dict:
        """Return current real-time telemetry dictionary."""
        with self._lock:
            return self.stats.as_dict()

    # ----------------------------------------------------------------------
    # Internal Worker Loop & Ingestion Logic
    # ----------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Main loop managing connection, boundary sync, and frame extraction."""
        current_delay = self.reconnect_initial_delay

        while self._running:
            try:
                t0 = time.perf_counter()
                logger.debug("Opening HTTP stream for channel %d: %s", self.channel, self.stream_url)
                
                cap = cv2.VideoCapture(self.stream_url)
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open HTTP video stream on channel {self.channel}")

                self._cap = cap
                connect_latency_ms = (time.perf_counter() - t0) * 1000
                with self._lock:
                    self.stats.latency_ms = connect_latency_ms
                    self.stats.is_connected = True

                # Mid-stream JPEG boundary synchronization (discard initial fragments)
                synced = False
                for _ in range(8):
                    if not self._running:
                        break
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        synced = True
                        h, w = frame.shape[:2]
                        with self._lock:
                            self.stats.tick_frame(w, h)
                        break
                    time.sleep(0.04)

                if not synced and self._running:
                    raise RuntimeError(f"Failed to synchronize JPEG boundary on channel {self.channel}")

                # Reset backoff delay on successful connection
                current_delay = self.reconnect_initial_delay

                # Continuous frame ingestion loop
                last_frame_fetch_time = time.monotonic()
                target_interval = 1.0 / max(self.target_fps, 1)

                consecutive_read_failures = 0

                while self._running:
                    ret, frame = cap.read()
                    now_mono = time.monotonic()
                    now_epoch = time.time()

                    if not ret or frame is None or frame.size == 0:
                        consecutive_read_failures += 1
                        with self._lock:
                            self.stats.dropped_frames += 1

                        if consecutive_read_failures > (self.target_fps * 3):  # 3 seconds of no frames
                            logger.warning(
                                "Channel %d: Stalled stream (%d consecutive failures)",
                                self.channel, consecutive_read_failures
                            )
                            break
                        time.sleep(0.02)
                        continue

                    consecutive_read_failures = 0
                    h, w = frame.shape[:2]

                    # Throttle frame insertion to target_fps to save memory & processing
                    if (now_mono - last_frame_fetch_time) >= (target_interval * 0.9):
                        last_frame_fetch_time = now_mono
                        self._frame_counter += 1

                        item = FrameItem(
                            frame=frame,
                            timestamp=now_epoch,
                            monotonic_ts=now_mono,
                            frame_idx=self._frame_counter,
                        )

                        with self._lock:
                            self._buffer.append(item)
                            self.stats.tick_frame(w, h)

            except Exception as e:
                if self._running:
                    logger.warning(
                        "Stream error on channel %d: %s. Reconnecting in %.1fs...",
                        self.channel, e, current_delay
                    )
                    with self._lock:
                        self.stats.mark_disconnected()
                        self.stats.reconnect_count += 1
            finally:
                if self._cap:
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None

            # Reconnection backoff sleep
            if self._running:
                sleep_start = time.monotonic()
                while self._running and (time.monotonic() - sleep_start) < current_delay:
                    time.sleep(0.1)
                # Exponential backoff
                current_delay = min(current_delay * 2.0, self.reconnect_max_delay)
