"""Cloud escalation handler for edge device incident escalations."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .anomaly import score_clip
from .config import config
from .edge_registry import registry as edge_registry
from .ensemble import ensemble_scorer
from .incidents import incident_engine
from .streaming import streaming_manager
from .vlm_explainer import explain_incident
from . import twelvelabs_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class EscalationRequest:
    """Encapsulates escalated clip and telemetry from an edge device."""
    edge_device_id: str
    edge_score: float
    edge_embedding: List[float]
    clip_data: Optional[bytes] = None
    clip_filename: str = ""
    timestamp_ms: int = 0
    scene_id: str = ""
    camera_id: Optional[str] = ""
    channel: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationResult:
    """Outcome of cloud re-embedding, baseline evaluation, and ensemble scoring."""
    escalation_id: str
    edge_device_id: str
    edge_score: float
    cloud_score: float
    ensemble_score: float
    is_confirmed_anomaly: bool
    confidence: float
    incident_id: Optional[str] = None
    cloud_embedding: Optional[List[float]] = None
    temporal_boost: float = 0.0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Escalation Tracker
# ---------------------------------------------------------------------------


class EscalationTracker:
    """Tracks global and per-device escalation metrics across all camera feeds."""

    def __init__(self):
        self.escalations: List[EscalationResult] = []
        self._per_device: Dict[str, List[EscalationResult]] = {}

    @property
    def escalation_count(self) -> int:
        return len(self.escalations)

    @property
    def confirmation_rate(self) -> float:
        if not self.escalations:
            return 0.0
        confirmed = sum(1 for e in self.escalations if e.is_confirmed_anomaly)
        return round(confirmed / len(self.escalations), 4)

    @property
    def avg_cloud_score(self) -> float:
        if not self.escalations:
            return 0.0
        return round(sum(e.cloud_score for e in self.escalations) / len(self.escalations), 4)

    @property
    def edge_accuracy(self) -> float:
        """How often the edge device's escalation was confirmed by central cloud."""
        return self.confirmation_rate

    def record(self, result: EscalationResult):
        self.escalations.append(result)
        self._per_device.setdefault(result.edge_device_id, []).append(result)

    def device_stats(self, device_id: str) -> Dict[str, Any]:
        results = self._per_device.get(device_id, [])
        if not results:
            return {
                "escalation_count": 0,
                "confirmation_rate": 0.0,
                "avg_cloud_score": 0.0,
                "false_positive_count": 0,
            }
        confirmed = sum(1 for r in results if r.is_confirmed_anomaly)
        return {
            "escalation_count": len(results),
            "confirmation_rate": round(confirmed / len(results), 4),
            "avg_cloud_score": round(sum(r.cloud_score for r in results) / len(results), 4),
            "false_positive_count": len(results) - confirmed,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "escalation_count": self.escalation_count,
            "confirmation_rate": self.confirmation_rate,
            "avg_cloud_score": self.avg_cloud_score,
            "edge_accuracy": self.edge_accuracy,
        }

    def clear(self):
        self.escalations.clear()
        self._per_device.clear()


# Global escalation tracker singleton
tracker = EscalationTracker()


# ---------------------------------------------------------------------------
# Escalation Handler Pipeline
# ---------------------------------------------------------------------------


async def handle_escalation(request: EscalationRequest) -> EscalationResult:
    """Processes an escalated clip from an edge device:

    1. Checks adaptive load shedding / backpressure manager.
    2. Re-embeds video with Twelve Labs Marengo or local PyTorch Model Server.
    3. Scores embedding against the central Qdrant baseline.
    4. Fuses cloud and edge scores via EnsembleScorer with temporal boosting.
    5. Feeds into Incident Formation Engine (EMA smoothing, hysteresis, cooldown merging).
    6. Triggers VLM Scene Explainer and records metrics.
    """
    t0 = time.perf_counter()
    escalation_id = str(uuid.uuid4())[:12]
    streaming_manager.enter_request()

    try:
        # Check load shedding drop
        if streaming_manager.should_drop_escalation(request.edge_score):
            latency_ms = (time.perf_counter() - t0) * 1000
            streaming_manager.exit_request(latency_ms)
            return EscalationResult(
                escalation_id=escalation_id,
                edge_device_id=request.edge_device_id,
                edge_score=request.edge_score,
                cloud_score=0.0,
                ensemble_score=request.edge_score,
                is_confirmed_anomaly=False,
                confidence=0.0,
                incident_id=None,
                cloud_embedding=None,
                temporal_boost=0.0,
                latency_ms=latency_ms,
            )

        cloud_embedding: Optional[List[float]] = None
        pegasus_video_id: Optional[str] = None
        saved_tmp_path: Optional[str] = None

        # Step 1: Re-embed with full cloud/local model if video clip is attached
        if request.clip_data and not streaming_manager.should_skip_baseline_knn():
            if twelvelabs_client.is_enabled():
                try:
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp.write(request.clip_data)
                        saved_tmp_path = tmp.name

                    upload_result = twelvelabs_client.upload_video(saved_tmp_path, index_type="marengo")
                    video_id = upload_result.get("marengo_video_id")
                    if video_id:
                        emb = twelvelabs_client.get_video_embedding(video_id)
                        if emb:
                            cloud_embedding = emb
                            logger.info("Extracted Marengo embedding for escalation %s (dim=%d)", escalation_id, len(emb))
                except Exception as exc:
                    logger.warning("Twelve Labs re-embedding failed (%s), attempting local model server.", exc)

            # Fallback to local PyTorch Model Server
            if cloud_embedding is None:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        resp = await client.post(
                            f"{config.model_server_url}/embed",
                            files={"file": (request.clip_filename or "clip.mp4", request.clip_data, "video/mp4")},
                        )
                        if resp.status_code == 200:
                            embed_data = resp.json()
                            cloud_embedding = embed_data.get("embedding")
                except Exception as exc:
                    logger.debug("Local model server unavailable: %s", exc)

        # Step 2: Select active embedding for baseline scoring
        scoring_embedding = cloud_embedding if cloud_embedding is not None else request.edge_embedding

        # Step 3: Query Central Qdrant Baseline (unless skipped by load shedding)
        camera_key = request.camera_id or (f"cam-{request.channel}" if request.channel else request.edge_device_id)
        if streaming_manager.should_skip_baseline_knn():
            cloud_score = request.edge_score
        else:
            cloud_result = score_clip(
                embedding=scoring_embedding,
                collection_name=config.collection_name,
                k=config.confirmation_k,
                threshold=config.cloud_anomaly_threshold,
                scene_id=request.scene_id or None,
                camera_id=camera_key,
            )
            cloud_score = cloud_result.anomaly_score

        # Step 4: Multi-Model Ensemble Scoring
        now_ts = (request.timestamp_ms / 1000.0) if request.timestamp_ms > 0 else time.time()
        ens_result = ensemble_scorer.score(
            edge_score=request.edge_score,
            cloud_score=cloud_score,
            device_id=request.edge_device_id,
            timestamp=now_ts,
        )

        is_confirmed = ens_result.is_anomaly

        # Step 5: Incident Formation Engine (EMA smoothing, hysteresis, cooldown merging)
        ch_num = request.channel
        if ch_num is None:
            # Extract channel from device ID or camera key if possible
            if request.edge_device_id.startswith("cam-"):
                try:
                    ch_num = int(request.edge_device_id.replace("cam-", ""))
                except ValueError:
                    ch_num = 1
            else:
                ch_num = 1

        cam_names = config.parse_camera_names()
        cam_name = cam_names.get(ch_num, f"Camera {ch_num}")

        incident_record = incident_engine.process_escalation(
            channel=ch_num,
            camera_id=camera_key,
            camera_name=cam_name,
            edge_score=request.edge_score,
            cloud_score=cloud_score,
            ensemble_score=ens_result.ensemble_score,
            timestamp=now_ts,
            scene_id=request.scene_id or "",
        )

        incident_id = incident_record.incident_id if incident_record is not None else (f"inc-{escalation_id}" if is_confirmed else None)

        # Step 6: Trigger VLM Scene Explainer
        if incident_record is not None and not streaming_manager.should_skip_vlm():
            try:
                explanation = await explain_incident(
                    incident=incident_record,
                    video_id=pegasus_video_id,
                    clip_path=saved_tmp_path,
                )
                incident_engine.attach_vlm_explanation(incident_record.incident_id, explanation)
            except Exception as exc:
                logger.warning("VLM explanation attachment failed: %s", exc)

        if saved_tmp_path:
            Path(saved_tmp_path).unlink(missing_ok=True)

        # Step 7: Update registries and telemetry
        device = edge_registry.get(request.edge_device_id)
        if device:
            device.record_escalation(is_confirmed)

        latency_ms = (time.perf_counter() - t0) * 1000
        streaming_manager.exit_request(latency_ms)

        result = EscalationResult(
            escalation_id=escalation_id,
            edge_device_id=request.edge_device_id,
            edge_score=request.edge_score,
            cloud_score=cloud_score,
            ensemble_score=ens_result.ensemble_score,
            is_confirmed_anomaly=is_confirmed,
            confidence=ens_result.confidence,
            incident_id=incident_id,
            cloud_embedding=cloud_embedding,
            temporal_boost=ens_result.temporal_boost,
            latency_ms=latency_ms,
        )

        tracker.record(result)
        return result

    except Exception:
        latency_ms = (time.perf_counter() - t0) * 1000
        streaming_manager.exit_request(latency_ms)
        raise
