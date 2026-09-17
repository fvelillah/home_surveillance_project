"""
Edge FastAPI Service & Direct HTTP Video Streaming Proxy.

Serves the edge surveillance node on port 7777:
  - Direct HTTP MJPEG stream proxy (/api/v1/stream/{channel}/live) from StreamWorker ring buffers
  - Instant snapshot API (/api/v1/stream/{channel}/snapshot)
  - Real-time multi-channel telemetry & health check (/health, /api/v1/cameras)
  - Live anomaly scoring stream (/api/v1/scores)
  - SQLite offline queue metrics (/api/v1/queue)
  - Edge triage supervisor coordinating continuous segmentation, scoring, and cloud escalation
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from edge.capture_manager import CaptureManager
from edge.config import EdgeConfig, config as default_config
from edge.detector import QdrantEdgeDetector
from edge.model import EdgeFeatureExtractor
from edge.queue import AutoDrainWorker, IncidentItem, SQLiteOfflineQueue
from edge.segmenter import SlidingWindowSegmenter

logger = logging.getLogger("edge_node")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ----------------------------------------------------------------------
# Background Edge Triage Supervisor
# ----------------------------------------------------------------------

class EdgeTriageService:
    """
    Coordinates continuous multi-channel segmentation, feature extraction,
    two-shard anomaly scoring, and automatic incident escalation.
    """

    def __init__(
        self,
        capture_manager: CaptureManager,
        segmenter: SlidingWindowSegmenter,
        model: EdgeFeatureExtractor,
        detector: QdrantEdgeDetector,
        queue: SQLiteOfflineQueue,
        cfg: Optional[EdgeConfig] = None,
    ) -> None:
        self.config = cfg or default_config
        self.capture_manager = capture_manager
        self.segmenter = segmenter
        self.model = model
        self.detector = detector
        self.queue = queue

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Telemetry: last 30 scores per channel
        self.score_history: Dict[int, collections.deque] = {
            ch: collections.deque(maxlen=30) for ch in range(1, self.config.num_cameras + 1)
        }
        self.latest_scores: Dict[int, float] = {ch: 0.0 for ch in range(1, self.config.num_cameras + 1)}
        self.escalation_counts: Dict[int, int] = {ch: 0 for ch in range(1, self.config.num_cameras + 1)}

    def start(self) -> None:
        """Start the background triage evaluation loop."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._triage_loop,
            name="EdgeTriageSupervisor",
            daemon=True,
        )
        self._thread.start()
        logger.info("EdgeTriageSupervisor background loop started")

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the background triage supervisor."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("EdgeTriageSupervisor stopped")

    def _triage_loop(self) -> None:
        """Continuous polling across all configured camera feeds."""
        while self._running and not self._stop_event.is_set():
            try:
                self.evaluate_all_channels()
            except Exception as e:
                logger.error("Error during edge triage evaluation cycle: %s", e)

            self._stop_event.wait(self.config.triage_poll_interval_s)

    def evaluate_all_channels(self) -> None:
        """Run a single evaluation pass over all active channels."""
        for ch, worker in self.capture_manager.workers.items():
            if not self._running:
                break

            # Only evaluate if enough time has passed for a new sliding window step
            if not self.segmenter.should_extract_new_clip(ch):
                continue

            clip = self.segmenter.extract_latest_clip(worker)
            if not clip:
                continue

            self.segmenter.mark_clip_extracted(ch, clip.end_time)
            clip_id = str(uuid.uuid4())

            # 1. Feature Extraction
            embedding = self.model.extract_from_clip(clip)

            # 2. Qdrant Edge Two-Shard Scoring
            score, details = self.detector.score_clip(
                channel=ch,
                embedding=embedding,
                clip_id=clip_id,
                timestamp=clip.end_time,
            )

            # Record telemetry
            with self._lock:
                self.latest_scores[ch] = score
                self.score_history[ch].append(
                    {
                        "timestamp": clip.end_time,
                        "score": score,
                        "is_escalated": details["is_escalated"],
                    }
                )

            # 3. Evaluate Triage Escalation
            if details["is_escalated"]:
                logger.warning(
                    "🚨 ANOMALY ESCALATION on Channel %d (%s)! Score: %.3f >= T_edge (%.3f)",
                    ch,
                    worker.camera_name,
                    score,
                    self.config.edge_triage_threshold,
                )
                self.trigger_escalation(
                    channel=ch,
                    score=score,
                    embedding=embedding.tolist(),
                    clip_end_time=clip.end_time,
                )

    def trigger_escalation(
        self,
        channel: int,
        score: float,
        embedding: Optional[List[float]] = None,
        clip_end_time: Optional[float] = None,
    ) -> Optional[IncidentItem]:
        """Execute on-demand incident capture and enqueue to SQLite offline queue."""
        worker = self.capture_manager.get_worker(channel)
        cam_name = worker.camera_name if worker else f"Camera {channel}"
        ts = clip_end_time or time.time()

        # Capture high-res incident clip (subtype=0 with Sub fallback)
        clip_path = self.capture_manager.capture_incident_clip(channel, duration_s=10.0)
        snap_path = self.config.snapshots_dir / f"incident_ch{channel}_{int(ts)}.jpg"
        self.capture_manager.capture_snapshot(channel, output_path=snap_path)

        with self._lock:
            self.escalation_counts[channel] = self.escalation_counts.get(channel, 0) + 1

        if clip_path:
            incident = IncidentItem(
                channel=channel,
                camera_name=cam_name,
                timestamp=ts,
                score=score,
                clip_path=str(clip_path),
                snapshot_path=str(snap_path) if snap_path.exists() else None,
                embedding=embedding,
            )
            self.queue.enqueue(incident)
            return incident
        return None


# ----------------------------------------------------------------------
# Application State Container & Lifespan
# ----------------------------------------------------------------------

class EdgeAppState:
    def __init__(self, cfg: EdgeConfig) -> None:
        self.config = cfg
        self.start_time = time.time()
        self.capture_manager = CaptureManager(cfg=cfg)
        self.segmenter = SlidingWindowSegmenter(cfg=cfg)
        self.model = EdgeFeatureExtractor(cfg=cfg)
        self.detector = QdrantEdgeDetector(cfg=cfg)
        self.queue = SQLiteOfflineQueue(cfg=cfg)
        self.drain_worker = AutoDrainWorker(queue=self.queue, cfg=cfg)
        self.triage_service = EdgeTriageService(
            capture_manager=self.capture_manager,
            segmenter=self.segmenter,
            model=self.model,
            detector=self.detector,
            queue=self.queue,
            cfg=cfg,
        )


app_state: Optional[EdgeAppState] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for Edge service."""
    global app_state
    logger.info("Initializing Edge Surveillance Application...")
    app_state = EdgeAppState(cfg=default_config)

    # 1. Start camera workers
    app_state.capture_manager.start_all()

    # 2. Start offline queue auto-drain worker
    app_state.drain_worker.start()

    # 3. Start triage supervisor
    app_state.triage_service.start()

    logger.info("Edge Surveillance Node operational on port %d", app_state.config.edge_port)
    yield

    # Shutdown sequence
    logger.info("Shutting down Edge Surveillance Node...")
    if app_state:
        app_state.triage_service.stop()
        app_state.drain_worker.stop()
        app_state.capture_manager.stop_all()
        app_state.detector.close()
    logger.info("Edge Surveillance Node shutdown complete")


def create_app(cfg: Optional[EdgeConfig] = None) -> FastAPI:
    """FastAPI Application factory."""
    app = FastAPI(
        title="Dahua Surveillance Edge Node",
        description="Local AI edge triage, Direct HTTP video proxy, and Qdrant Edge vector scoring",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()


# ----------------------------------------------------------------------
# Helper: Multipart MJPEG Stream Generator
# ----------------------------------------------------------------------

async def mjpeg_frame_generator(
    channel: int, target_fps: int = 10, request: Optional[Request] = None
) -> AsyncGenerator[bytes, None]:
    """Yield multipart JPEG chunks for zero-latency direct browser streaming."""
    frame_interval = 1.0 / max(target_fps, 1)

    try:
        while True:
            if request and await request.is_disconnected():
                break

            frame_bytes: Optional[bytes] = None

            if app_state:
                frame = app_state.capture_manager.get_latest_frame(channel)
                if frame is not None:
                    ret, buf = await asyncio.to_thread(
                        cv2.imencode, ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    if ret:
                        frame_bytes = buf.tobytes()

            if frame_bytes is None:
                placeholder = np.zeros((480, 704, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    f"Camera {channel} Connecting...",
                    (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (180, 180, 180),
                    2,
                )
                ret, buf = await asyncio.to_thread(
                    cv2.imencode, ".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 60]
                )
                if ret:
                    frame_bytes = buf.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame_bytes)).encode("utf-8") + b"\r\n\r\n"
                + frame_bytes + b"\r\n"
            )

            await asyncio.sleep(frame_interval)
    except (asyncio.CancelledError, GeneratorExit):
        pass


# ----------------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------------

@app.get("/health", tags=["Telemetry"])
async def get_health():
    """System health check and component status summary."""
    if not app_state:
        raise HTTPException(status_code=503, detail="Edge service not initialized")

    uptime = time.time() - app_state.start_time
    worker_stats = app_state.capture_manager.get_all_stats()
    connected_workers = sum(1 for s in worker_stats if s.get("is_connected", False))
    q_stats = app_state.queue.get_stats()
    detector_stats = app_state.detector.get_stats()

    return {
        "status": "HEALTHY",
        "uptime_seconds": round(uptime, 1),
        "edge_port": app_state.config.edge_port,
        "cameras": {
            "total": len(worker_stats),
            "connected": connected_workers,
        },
        "model": {
            "type": app_state.model.model_type,
            "dim": app_state.model.embedding_dim,
            "device": app_state.model.device_str,
            "is_model_loaded": app_state.model.model is not None,
            "last_inference_ms": round(app_state.model.last_inference_ms, 2),
        },
        "detector": detector_stats,
        "queue": q_stats,
    }


@app.get("/api/v1/cameras", tags=["Cameras"])
async def list_cameras():
    """List all camera channels with live telemetry, decoded FPS, latency, and latest score."""
    if not app_state:
        raise HTTPException(status_code=503, detail="Edge service not initialized")

    stats_list = app_state.capture_manager.get_all_stats()
    enriched = []
    for s in stats_list:
        ch = s["channel"]
        s_copy = dict(s)
        s_copy["latest_score"] = round(app_state.triage_service.latest_scores.get(ch, 0.0), 3)
        s_copy["escalations"] = app_state.triage_service.escalation_counts.get(ch, 0)
        s_copy["stream_proxy_url"] = f"/api/v1/stream/{ch}/live"
        s_copy["snapshot_proxy_url"] = f"/api/v1/stream/{ch}/snapshot"
        enriched.append(s_copy)

    return {"cameras": enriched}


@app.get("/api/v1/cameras/{channel}", tags=["Cameras"])
async def get_camera_detail(channel: int):
    """Retrieve detailed telemetry for a single camera channel."""
    if not app_state:
        raise HTTPException(status_code=503, detail="Edge service not initialized")

    s = app_state.capture_manager.get_channel_stats(channel)
    if not s:
        raise HTTPException(status_code=404, detail=f"Channel {channel} not found")

    res = dict(s)
    res["latest_score"] = round(app_state.triage_service.latest_scores.get(channel, 0.0), 3)
    res["escalations"] = app_state.triage_service.escalation_counts.get(channel, 0)
    res["history"] = list(app_state.triage_service.score_history.get(channel, []))
    return res


@app.get("/api/v1/stream/{channel}/live", tags=["Video Stream"])
async def stream_live_mjpeg(request: Request, channel: int, fps: Optional[int] = Query(None, ge=1, le=25)):
    """
    Direct HTTP MJPEG multipart stream proxy.
    Directly streams decoded video frames from the camera's RAM ring buffer.
    """
    if not app_state or channel not in app_state.capture_manager.workers:
        raise HTTPException(status_code=404, detail=f"Camera channel {channel} not configured")

    target_fps = fps or app_state.config.ingest_fps
    return StreamingResponse(
        mjpeg_frame_generator(channel, target_fps=target_fps, request=request),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/v1/stream/{channel}/snapshot", tags=["Video Stream"])
async def get_camera_snapshot(channel: int):
    """Return instant JPEG snapshot for the requested camera channel."""
    if not app_state or channel not in app_state.capture_manager.workers:
        raise HTTPException(status_code=404, detail=f"Camera channel {channel} not configured")

    image_bytes = app_state.capture_manager.capture_snapshot(channel)
    if not image_bytes:
        raise HTTPException(status_code=504, detail=f"Snapshot unavailable for channel {channel}")

    return Response(content=image_bytes, media_type="image/jpeg")
 
 
@app.websocket("/api/v1/stream/{channel}/ws")
async def stream_live_ws(websocket: WebSocket, channel: int):
    """
    Direct WebSocket binary JPEG stream proxy.
    Bypasses browser HTTP/1.1 6-connection limits, allowing all 9 camera channels
    to stream concurrently with low latency and zero socket starvation.
    """
    await websocket.accept()
    if not app_state or channel not in app_state.capture_manager.workers:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    target_fps = app_state.config.ingest_fps or 10
    interval = 1.0 / max(target_fps, 1)

    try:
        while True:
            frame_bytes: Optional[bytes] = None
            if app_state:
                frame = app_state.capture_manager.get_latest_frame(channel)
                if frame is not None:
                    ret, buf = await asyncio.to_thread(
                        cv2.imencode, ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    if ret:
                        frame_bytes = buf.tobytes()

            if frame_bytes is None:
                placeholder = np.zeros((480, 704, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder,
                    f"Camera {channel} Connecting...",
                    (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (180, 180, 180),
                    2,
                )
                ret, buf = await asyncio.to_thread(
                    cv2.imencode, ".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 60]
                )
                if ret:
                    frame_bytes = buf.tobytes()

            if frame_bytes:
                await websocket.send_bytes(frame_bytes)

            await asyncio.sleep(interval)
    except (WebSocketDisconnect, ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception as exc:
        logger.debug("WS stream channel %d closed: %s", channel, exc)


@app.get("/api/v1/scores", tags=["Anomaly Scoring"])
async def get_live_scores():
    """Retrieve current anomaly scores and recent history across all 9 feeds."""
    if not app_state:
        raise HTTPException(status_code=503, detail="Edge service not initialized")

    return {
        "triage_threshold": app_state.config.edge_triage_threshold,
        "latest_scores": app_state.triage_service.latest_scores,
        "escalation_counts": app_state.triage_service.escalation_counts,
        "history": {
            ch: list(hist) for ch, hist in app_state.triage_service.score_history.items()
        },
    }


@app.get("/api/v1/queue", tags=["Offline Queue"])
async def get_queue_telemetry():
    """Retrieve SQLite offline persistence queue metrics and pending items."""
    if not app_state:
        raise HTTPException(status_code=503, detail="Edge service not initialized")

    q_stats = app_state.queue.get_stats()
    pending = app_state.queue.peek_pending(limit=10)
    return {
        "stats": q_stats,
        "pending_count": len(pending),
        "pending_items": [item.to_dict() for item in pending],
    }


class TriggerTriageRequest(BaseModel):
    channel: int = Field(..., ge=1, le=9)
    score: float = Field(default=0.85, ge=0.0, le=1.0)


@app.post("/api/v1/triage/trigger", tags=["Anomaly Scoring"])
async def trigger_manual_triage(req: TriggerTriageRequest):
    """Manually trigger an incident escalation for testing and verification."""
    if not app_state:
        raise HTTPException(status_code=503, detail="Edge service not initialized")

    worker = app_state.capture_manager.get_worker(req.channel)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Channel {req.channel} not found")

    dummy_emb = np.random.randn(app_state.config.embedding_dim).astype(np.float32)
    dummy_emb /= np.linalg.norm(dummy_emb)

    incident = app_state.triage_service.trigger_escalation(
        channel=req.channel,
        score=req.score,
        embedding=dummy_emb.tolist(),
        clip_end_time=time.time(),
    )

    if not incident:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to capture incident clip"},
        )

    return {
        "status": "TRIGGERED",
        "incident_id": incident.incident_id,
        "channel": incident.channel,
        "score": incident.score,
        "clip_path": incident.clip_path,
        "snapshot_path": incident.snapshot_path,
    }


class SeedBaselineRequest(BaseModel):
    channel: int = Field(..., ge=1, le=9)
    num_vectors: int = Field(default=10, ge=1, le=500)


@app.post("/api/v1/baseline/seed", tags=["Anomaly Scoring"])
async def seed_baseline_vectors(req: SeedBaselineRequest):
    """Seed synthetic baseline vectors for a channel in Qdrant Edge."""
    if not app_state:
        raise HTTPException(status_code=503, detail="Edge service not initialized")

    dim = app_state.config.embedding_dim
    rng = np.random.RandomState(42 + req.channel)
    vectors = [rng.randn(dim).astype(np.float32) for _ in range(req.num_vectors)]
    vectors = [v / np.linalg.norm(v) for v in vectors]

    count = app_state.detector.seed_baseline(req.channel, vectors)
    return {
        "status": "SEEDED",
        "channel": req.channel,
        "vectors_seeded": count,
    }
