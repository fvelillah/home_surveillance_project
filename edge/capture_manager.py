"""
Multi-Channel Capture Manager for Dahua Surveillance System.

Coordinates concurrent HTTP video stream workers across all 9 camera channels,
aggregates telemetry, and provides on-demand high-resolution Main-Stream clip extraction
and instant snapshot capture.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

from edge.config import EdgeConfig, config as default_config
from edge.stream_worker import FrameItem, StreamWorker

logger = logging.getLogger(__name__)


class CaptureManager:
    """
    Manager orchestrating multi-channel Dahua HTTP video streaming.
    
    Responsibilities:
      - Supervises concurrent Sub-Stream workers (Channels 1 to N)
      - Provides unified frame querying for edge triage and segmentation
      - On-demand high-res Main-Stream incident clip recording (subtype=0)
      - Snapshot capture with fallback
      - System-wide health and telemetry monitoring
    """

    def __init__(self, cfg: Optional[EdgeConfig] = None) -> None:
        self.config = cfg or default_config
        self.workers: Dict[int, StreamWorker] = {}
        self._init_workers()

    def _init_workers(self) -> None:
        """Instantiate stream workers for all configured channels."""
        for ch in range(1, self.config.num_cameras + 1):
            sub_url = self.config.get_substream_url(ch)
            camera_name = self.config.get_camera_name(ch)

            worker = StreamWorker(
                channel=ch,
                stream_url=sub_url,
                camera_name=camera_name,
                target_fps=self.config.ingest_fps,
                buffer_capacity_sec=self.config.frame_buffer_capacity_sec,
                reconnect_initial_delay_sec=self.config.reconnect_initial_delay_sec,
                reconnect_max_delay_sec=self.config.reconnect_max_delay_sec,
                stream_read_timeout_sec=self.config.stream_read_timeout_sec,
            )
            self.workers[ch] = worker

    # ----------------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------------

    def start_all(self) -> None:
        """Start all camera stream workers."""
        logger.info("Starting all %d camera stream workers...", len(self.workers))
        for ch, worker in self.workers.items():
            worker.start()

    def stop_all(self, timeout: float = 3.0) -> None:
        """Stop all camera stream workers."""
        logger.info("Stopping all camera stream workers...")
        for ch, worker in self.workers.items():
            worker.stop(timeout=timeout)

    def restart_channel(self, channel: int) -> bool:
        """Restart a specific camera stream worker."""
        if channel not in self.workers:
            logger.warning("Cannot restart unknown channel %d", channel)
            return False
        logger.info("Restarting stream worker for channel %d", channel)
        self.workers[channel].stop()
        self.workers[channel].start()
        return True

    def __enter__(self) -> CaptureManager:
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop_all()

    # ----------------------------------------------------------------------
    # Frame Access & Telemetry
    # ----------------------------------------------------------------------

    def get_worker(self, channel: int) -> Optional[StreamWorker]:
        return self.workers.get(channel)

    def get_latest_frame(self, channel: int) -> Optional[np.ndarray]:
        worker = self.workers.get(channel)
        if not worker:
            return None
        return worker.get_latest_frame()

    def get_recent_frames(self, channel: int, duration_s: float) -> List[FrameItem]:
        worker = self.workers.get(channel)
        if not worker:
            return []
        return worker.get_recent_frames(duration_s=duration_s)

    def get_all_stats(self) -> List[dict]:
        """Aggregate telemetry statistics for all channels."""
        return [worker.get_stats() for worker in self.workers.values()]

    def get_channel_stats(self, channel: int) -> Optional[dict]:
        worker = self.workers.get(channel)
        if not worker:
            return None
        return worker.get_stats()

    # ----------------------------------------------------------------------
    # On-Demand Incident Recording & Snapshot APIs
    # ----------------------------------------------------------------------

    def capture_snapshot(self, channel: int, output_path: Optional[Path] = None) -> Optional[bytes]:
        """
        Capture a single still JPEG image for the given channel.
        First attempts direct NVR snapshot API, falls back to latest buffered frame.
        """
        snap_url = self.config.get_snapshot_url(channel)
        image_bytes: Optional[bytes] = None

        try:
            # 1. Try Dahua Snapshot CGI with Digest / Basic auth
            resp = requests.get(
                snap_url,
                auth=HTTPDigestAuth(self.config.nvr_user, self.config.nvr_password),
                timeout=3.0,
            )
            if resp.status_code == 401:
                resp = requests.get(
                    snap_url,
                    auth=HTTPBasicAuth(self.config.nvr_user, self.config.nvr_password),
                    timeout=3.0,
                )
            if resp.status_code == 200 and len(resp.content) > 1000:
                image_bytes = resp.content
        except Exception as e:
            logger.debug("Snapshot CGI failed on channel %d: %s. Falling back to buffered frame.", channel, e)

        # Fallback: encode latest frame from worker buffer
        if image_bytes is None:
            frame = self.get_latest_frame(channel)
            if frame is not None:
                ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ret:
                    image_bytes = buf.tobytes()

        if image_bytes and output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image_bytes)

        return image_bytes

    def capture_incident_clip(
        self,
        channel: int,
        duration_s: float = 10.0,
        output_path: Optional[Path] = None,
        target_fps: int = 25,
    ) -> Optional[Path]:
        """
        Record a high-resolution video clip of an escalated incident.
        Connects to Main-Stream (subtype=0) on-demand, records for `duration_s`,
        and writes MP4 video. Falls back to buffered Sub-Stream frames if Main-Stream is unreachable.
        """
        if output_path is None:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.config.incident_clips_dir / f"incident_ch{channel}_{ts_str}.mp4"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        main_url = self.config.get_mainstream_url(channel)

        logger.info("Recording high-res incident clip for channel %d (duration=%.1fs)...", channel, duration_s)

        frames: List[np.ndarray] = []
        cap = None

        try:
            # 1. Attempt on-demand Main-Stream (3K/4K) capture
            cap = cv2.VideoCapture(main_url)
            if cap.isOpened():
                t_end = time.monotonic() + duration_s
                while time.monotonic() < t_end:
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        frames.append(frame)
                    time.sleep(1.0 / target_fps)
        except Exception as e:
            logger.warning("Main-stream recording failed on channel %d: %s", channel, e)
        finally:
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass

        # 2. Fallback: Use buffered Sub-Stream frames if Main-Stream returned too few frames
        min_expected_frames = max(int(duration_s * 5), 5)
        if len(frames) < min_expected_frames:
            logger.info(
                "Main-Stream captured only %d frames; assembling fallback clip from buffered Sub-Stream",
                len(frames)
            )
            buffered_items = self.get_recent_frames(channel, duration_s=duration_s)
            if not buffered_items and channel in self.workers:
                all_buffered = self.workers[channel].get_all_buffered_frames()
                max_needed = max(int(duration_s * target_fps), 5)
                buffered_items = all_buffered[-max_needed:]
            frames = [item.frame for item in buffered_items]

        if not frames:
            logger.error("No frames available to write incident clip for channel %d", channel)
            return None

        # 3. Write frames to MP4 file
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (w, h))

        try:
            for f in frames:
                if f.shape[:2] != (h, w):
                    f = cv2.resize(f, (w, h))
                writer.write(f)
        finally:
            writer.release()

        logger.info(
            "Incident clip saved successfully: %s (%d frames, %dx%d)",
            output_path, len(frames), w, h
        )
        return output_path
