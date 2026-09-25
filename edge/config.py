"""
Edge Configuration and Environment Management.

Loads settings from environment variables and .env file with strong type validation
and engineering knobs for ingestion, buffering, and segmentation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import quote

# Load dotenv if present
try:
    from dotenv import load_dotenv
    # Search for .env in current dir and parent dirs
    env_path = Path(".env")
    if not env_path.exists():
        env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass


class EdgeConfig:
    """Surveillance Edge Configuration container."""

    def __init__(self) -> None:
        # --- Dahua NVR Connection ---
        self.nvr_ip: str = os.getenv("DAHUA_NVR_IP", "192.168.1.100")
        self.nvr_port: int = int(os.getenv("DAHUA_NVR_HTTP_PORT", "80"))
        self.nvr_user: str = os.getenv("DAHUA_NVR_USER", "admin")
        self.nvr_password: str = os.getenv("DAHUA_NVR_PASSWORD", "password")
        self.num_cameras: int = int(os.getenv("DAHUA_NUM_CAMERAS", "9"))

        # URL Templates
        self.substream_url_template: str = os.getenv(
            "DAHUA_SUBSTREAM_URL_TEMPLATE",
            "http://{user}:{password}@{ip}:{port}/cgi-bin/mjpg/video.cgi?channel={channel}&subtype=1"
        )
        self.mainstream_url_template: str = os.getenv(
            "DAHUA_MAINSTREAM_URL_TEMPLATE",
            "http://{user}:{password}@{ip}:{port}/cgi-bin/mjpg/video.cgi?channel={channel}&subtype=0"
        )
        self.snapshot_url_template: str = os.getenv(
            "DAHUA_SNAPSHOT_URL_TEMPLATE",
            "http://{user}:{password}@{ip}:{port}/cgi-bin/snapshot.cgi?channel={channel}"
        )

        # Camera Names mapping (format: "1:Front Door,2:Driveway,...")
        self._raw_camera_names: str = os.getenv(
            "CAMERA_NAMES",
            "1:Front Door,2:Driveway,3:Front Yard,4:Garage,5:Back Patio,6:Backyard,7:Side Alley North,8:Side Alley South,9:Perimeter Gate"
        )
        self.camera_names: Dict[int, str] = self._parse_camera_names(self._raw_camera_names)

        # --- Ingestion & Stream Worker Knobs ---
        self.ingest_fps: int = int(os.getenv("INGEST_FPS", "10"))
        self.frame_buffer_capacity_sec: int = int(os.getenv("FRAME_BUFFER_CAPACITY_SEC", "30"))
        self.reconnect_initial_delay_sec: float = float(os.getenv("RECONNECT_INITIAL_DELAY_SEC", "1.0"))
        self.reconnect_max_delay_sec: float = float(os.getenv("RECONNECT_MAX_DELAY_SEC", "30.0"))
        self.stream_read_timeout_sec: float = float(os.getenv("STREAM_READ_TIMEOUT_SEC", "5.0"))

        # --- Sliding Window Segmentation Knobs ---
        self.clip_duration_s: float = float(os.getenv("CLIP_DURATION_S", "10.0"))
        self.clip_overlap_s: float = float(os.getenv("CLIP_OVERLAP_S", "2.0"))
        self.segment_sample_frames: int = int(os.getenv("SEGMENT_SAMPLE_FRAMES", "16"))
        self.segment_target_width: int = int(os.getenv("SEGMENT_TARGET_WIDTH", "224"))
        self.segment_target_height: int = int(os.getenv("SEGMENT_TARGET_HEIGHT", "224"))

        # --- Storage Directories ---
        self.storage_dir: Path = Path(os.getenv("STORAGE_DIR", "./data/storage"))
        self.incident_clips_dir: Path = Path(os.getenv("INCIDENT_CLIPS_DIR", "./data/storage/incidents"))
        self.snapshots_dir: Path = Path(os.getenv("SNAPSHOTS_DIR", "./data/storage/snapshots"))

        # Create directories if they do not exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.incident_clips_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # --- Edge API Service ---
        self.edge_host: str = os.getenv("EDGE_HOST", "0.0.0.0")
        self.edge_port: int = int(os.getenv("EDGE_PORT", "7777"))

        # --- Mock Simulator ---
        self.mock_simulator_port: int = int(os.getenv("MOCK_SIMULATOR_PORT", "8888"))
        self.mock_simulator_num_cameras: int = int(os.getenv("MOCK_SIMULATOR_NUM_CAMERAS", "9"))

        # --- Phase 2: Edge Feature Extractor Knobs (PyTorch) ---
        self.model_type: str = os.getenv("MODEL_TYPE", "mobilenet_v3")
        self.embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "576"))

        # --- Phase 2: Qdrant Edge Two-Shard Knobs ---
        self.qdrant_edge_path: Path = Path(os.getenv("QDRANT_EDGE_PATH", "./data/qdrant_edge"))
        self.qdrant_edge_url: Optional[str] = os.getenv("QDRANT_EDGE_URL", None)
        self.edge_triage_threshold: float = float(os.getenv("EDGE_TRIAGE_THRESHOLD", "0.060"))
        self.edge_top_k: int = int(os.getenv("EDGE_TOP_K", "5"))
        self.edge_alpha_baseline: float = float(os.getenv("EDGE_ALPHA_BASELINE", "0.7"))
        self.edge_mutable_max_points: int = int(os.getenv("EDGE_MUTABLE_MAX_POINTS", "200"))

        # --- Phase 2: SQLite Offline Queue & Auto-Drain Knobs ---
        self.sqlite_queue_path: Path = Path(os.getenv("SQLITE_QUEUE_PATH", "./data/storage/offline_queue.db"))
        self.central_backend_url: str = os.getenv("CENTRAL_BACKEND_URL", "http://localhost:9876")
        self.drain_sync_interval_s: float = float(os.getenv("DRAIN_SYNC_INTERVAL_S", "5.0"))
        self.drain_max_retries: int = int(os.getenv("DRAIN_MAX_RETRIES", "5"))
        self.triage_poll_interval_s: float = float(os.getenv("TRIAGE_POLL_INTERVAL_S", "1.0"))

        # Create Qdrant storage dir if needed
        self.qdrant_edge_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _parse_camera_names(raw_str: str) -> Dict[int, str]:
        names: Dict[int, str] = {}
        if not raw_str:
            return names
        for item in raw_str.split(","):
            item = item.strip()
            if ":" in item:
                parts = item.split(":", 1)
                try:
                    ch = int(parts[0].strip())
                    name = parts[1].strip()
                    names[ch] = name
                except ValueError:
                    continue
        return names

    def get_camera_name(self, channel: int) -> str:
        return self.camera_names.get(channel, f"Camera {channel}")

    def get_substream_url(self, channel: int) -> str:
        # If user/password contain special chars, quote if needed
        return self.substream_url_template.format(
            user=self.nvr_user,
            password=self.nvr_password,
            ip=self.nvr_ip,
            port=self.nvr_port,
            channel=channel,
        )

    def get_mainstream_url(self, channel: int) -> str:
        return self.mainstream_url_template.format(
            user=self.nvr_user,
            password=self.nvr_password,
            ip=self.nvr_ip,
            port=self.nvr_port,
            channel=channel,
        )

    def get_snapshot_url(self, channel: int) -> str:
        return self.snapshot_url_template.format(
            user=self.nvr_user,
            password=self.nvr_password,
            ip=self.nvr_ip,
            port=self.nvr_port,
            channel=channel,
        )


# Global singleton instance
config = EdgeConfig()
