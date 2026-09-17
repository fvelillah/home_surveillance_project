"""Central Cloud Analytics Tier Configuration."""

from __future__ import annotations

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class BackendConfig(BaseSettings):
    """Central configuration knobs for cloud analytics, Qdrant baseline, and scoring."""

    # Central FastAPI Server
    central_host: str = os.getenv("CENTRAL_HOST", "0.0.0.0")
    central_port: int = int(os.getenv("CENTRAL_PORT", "9876"))

    # Local Model Server Fallback
    model_server_url: str = os.getenv("MODEL_SERVER_URL", "http://localhost:9877")

    # Qdrant Vector Baseline Cluster
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    collection_name: str = os.getenv("COLLECTION_NAME", "anomaly_baseline")

    # Scoring & Thresholds
    cloud_anomaly_threshold: float = float(os.getenv("CLOUD_ANOMALY_THRESHOLD", "0.15"))
    escalation_threshold: float = float(os.getenv("ESCALATION_THRESHOLD", "0.15"))
    confirmation_k: int = int(os.getenv("CONFIRMATION_K", "5"))

    # Multi-Model Ensemble Weights & Temporal Boosting
    ensemble_cloud_weight: float = float(os.getenv("ENSEMBLE_CLOUD_WEIGHT", "0.7"))
    ensemble_edge_weight: float = float(os.getenv("ENSEMBLE_EDGE_WEIGHT", "0.3"))
    ensemble_threshold: float = float(os.getenv("ENSEMBLE_THRESHOLD", "0.15"))
    temporal_boost_window_min: float = float(os.getenv("TEMPORAL_BOOST_WINDOW", "5.0"))
    temporal_boost_factor: float = float(os.getenv("TEMPORAL_BOOST_FACTOR", "0.1"))
    max_temporal_boost: float = float(os.getenv("MAX_TEMPORAL_BOOST", "0.3"))

    # Twelve Labs Cloud API Configuration
    twelve_labs_api_key: str = os.getenv("TWELVE_LABS_API_KEY", "")
    twelve_labs_api_url: str = os.getenv("TWELVE_LABS_API_URL", "https://api.twelvelabs.io/v1.3")
    marengo_index_name: str = os.getenv("TWELVE_LABS_MARENGO_INDEX_NAME", "dahua-surveillance-marengo")
    pegasus_index_name: str = os.getenv("TWELVE_LABS_PEGASUS_INDEX_NAME", "dahua-surveillance-pegasus")
    marengo_model: str = os.getenv("TWELVE_LABS_MARENGO_MODEL", "marengo3.0")
    pegasus_model: str = os.getenv("TWELVE_LABS_PEGASUS_MODEL", "pegasus1.5")
    twelve_labs_upload_timeout: int = int(os.getenv("TWELVE_LABS_UPLOAD_TIMEOUT", "600"))
    twelve_labs_max_clips: int = int(os.getenv("TWELVE_LABS_MAX_CLIPS", "10"))

    # Edge Node Connectivity
    edge_host: str = os.getenv("EDGE_HOST", "localhost")
    edge_port: int = int(os.getenv("EDGE_PORT", "7777"))

    # Multi-Camera Catalog
    num_cameras: int = int(os.getenv("DAHUA_NUM_CAMERAS", "9"))
    camera_names_raw: str = os.getenv(
        "CAMERA_NAMES",
        "1:Front Door,2:Driveway,3:Front Yard,4:Garage,5:Back Patio,6:Backyard,7:Side Alley North,8:Side Alley South,9:Perimeter Gate",
    )

    # Phase 4: Incident Formation & Hysteresis
    ema_alpha: float = float(os.getenv("EMA_ALPHA", "0.3"))
    incident_start_threshold: float = float(os.getenv("INCIDENT_START_THRESHOLD", "0.080"))
    incident_end_threshold: float = float(os.getenv("INCIDENT_END_THRESHOLD", "0.050"))
    cooldown_window_s: float = float(os.getenv("COOLDOWN_WINDOW_S", "30.0"))

    # Phase 4: Memory Governor & Anti-Poisoning
    quarantine_collection_name: str = os.getenv("QUARANTINE_COLLECTION_NAME", "anomaly_quarantine")
    quarantine_duration_s: int = int(os.getenv("QUARANTINE_DURATION_S", "3600"))
    max_vectors_per_camera: int = int(os.getenv("MAX_VECTORS_PER_CAMERA", "500"))
    baseline_retention_days: int = int(os.getenv("BASELINE_RETENTION_DAYS", "7"))
    anti_poisoning_threshold: float = float(os.getenv("ANTI_POISONING_THRESHOLD", "0.080"))

    # Phase 4: Streaming Backpressure & Load Shedding
    load_shed_auto: bool = os.getenv("LOAD_SHED_AUTO", "true").lower() in ("true", "1", "yes")
    load_shed_score_only_latency_ms: float = float(os.getenv("LOAD_SHED_SCORE_ONLY_LATENCY_MS", "500.0"))
    load_shed_passthrough_latency_ms: float = float(os.getenv("LOAD_SHED_PASSTHROUGH_LATENCY_MS", "1500.0"))
    load_shed_critical_threshold: float = float(os.getenv("LOAD_SHED_CRITICAL_THRESHOLD", "0.25"))

    def parse_camera_names(self) -> dict[int, str]:
        """Parses CAMERA_NAMES string into a channel -> name mapping."""
        cameras: dict[int, str] = {}
        if not self.camera_names_raw:
            return {i: f"Camera {i}" for i in range(1, self.num_cameras + 1)}

        for item in self.camera_names_raw.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                parts = item.split(":", 1)
                try:
                    ch = int(parts[0].strip())
                    name = parts[1].strip()
                    cameras[ch] = name
                except ValueError:
                    continue
        for i in range(1, self.num_cameras + 1):
            if i not in cameras:
                cameras[i] = f"Camera {i}"
        return cameras


# Global singleton instance
config = BackendConfig()
