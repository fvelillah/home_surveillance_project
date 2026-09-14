"""Central Cloud Analytics FastAPI Service on Port 9876."""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .anomaly import get_collection_stats, get_qdrant, index_baseline_vectors, score_clip
from .config import config
from .edge_registry import registry as edge_registry
from .ensemble import ensemble_scorer
from .escalation import EscalationRequest, handle_escalation, tracker as escalation_tracker
from .models import (
    AnalysisResultModel,
    AnomalyResultResponse,
    BaselineIndexRequest,
    CameraStreamInfo,
    EdgeDeviceModel,
    EdgeRegisterRequest,
    EmbedRequest,
    EnsembleResultModel,
    EscalationRequestModel,
    EscalationResponseModel,
    HealthResponse,
    NeighborResponse,
    SearchResultModel,
)
from . import twelvelabs_client

logger = logging.getLogger(__name__)
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Central Cloud Analytics Backend...")
    try:
        get_qdrant()
    except Exception as exc:
        logger.warning("Initial Qdrant connection check: %s", exc)
    yield
    logger.info("Shutting down Central Cloud Analytics Backend...")


app = FastAPI(
    title="Home Surveillance Central Cloud Analytics API",
    description="Central AI analytics tier for 9-channel Dahua surveillance, vector baseline scoring, and multi-model ensemble fusion.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration for Web Dashboard (Port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Health & Status Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    """Returns cluster health matrix across Qdrant, Model Server, Twelve Labs, and Edge Nodes."""
    qdrant_ok = False
    try:
        client = get_qdrant()
        if client is not None:
            client.get_collections()
            qdrant_ok = True
    except Exception:
        qdrant_ok = False

    model_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{config.model_server_url}/health")
            model_ok = (resp.status_code == 200)
    except Exception:
        model_ok = False

    devices = edge_registry.list_all()

    return HealthResponse(
        status="ok" if (qdrant_ok or model_ok or twelvelabs_client.is_enabled()) else "healthy_standalone",
        qdrant_connected=qdrant_ok,
        model_loaded=model_ok,
        twelve_labs_enabled=twelvelabs_client.is_enabled(),
        edge_devices_count=len(devices),
        uptime_s=round(time.time() - _start_time, 2),
    )


# ---------------------------------------------------------------------------
# Vector Scoring & Baseline Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/score", response_model=AnomalyResultResponse)
@app.post("/api/score", response_model=AnomalyResultResponse)
async def score_embedding_endpoint(req: EmbedRequest):
    """Scores a pre-computed embedding vector against the central Qdrant baseline."""
    t0 = time.perf_counter()
    result = score_clip(
        embedding=req.embedding,
        collection_name=req.collection_name,
        k=req.k,
        camera_id=req.camera_id,
        scene_id=req.scene_id,
    )
    total_ms = (time.perf_counter() - t0) * 1000

    return AnomalyResultResponse(
        anomaly_score=result.anomaly_score,
        is_anomaly=result.is_anomaly,
        neighbors=[
            NeighborResponse(
                clip_id=n.clip_id,
                similarity=n.similarity,
                source_video=n.source_video,
                scene_id=n.scene_id,
                camera_id=n.camera_id,
            )
            for n in result.neighbors
        ],
        latency_ms={"embed": 0.0, "search": result.latency.search_ms, "total": total_ms},
    )


@app.post("/api/v1/baseline/index")
async def index_baseline_endpoint(req: BaselineIndexRequest):
    """Upserts baseline calibration embeddings into Central Qdrant."""
    total_indexed = index_baseline_vectors(
        vectors=req.vectors,
        metadata_list=req.metadata,
        collection_name=req.collection_name,
    )
    return {
        "status": "ok",
        "indexed_points": total_indexed,
        "collection": req.collection_name or config.collection_name,
    }


@app.get("/api/v1/qdrant/stats")
@app.get("/api/qdrant/stats")
async def qdrant_stats_endpoint():
    """Returns point counts and vector metadata from the baseline collection."""
    return get_collection_stats()


# ---------------------------------------------------------------------------
# Edge Escalation Receiver & Scoring Pipeline
# ---------------------------------------------------------------------------


@app.post("/api/v1/escalate", response_model=EscalationResponseModel)
@app.post("/api/escalate", response_model=EscalationResponseModel)
async def escalate_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    edge_device_id: Optional[str] = Form(None),
    edge_score: Optional[float] = Form(None),
    edge_embedding: Optional[str] = Form(None),
    timestamp_ms: Optional[int] = Form(0),
    scene_id: Optional[str] = Form(""),
    camera_id: Optional[str] = Form(""),
    channel: Optional[int] = Form(None),
):
    """Receives an escalated incident from an edge device (supporting multipart form-data or JSON)."""
    clip_data = None
    clip_filename = ""

    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            body = await request.json()
            req_model = EscalationRequestModel(**body)
            esc_req = EscalationRequest(
                edge_device_id=req_model.edge_device_id,
                edge_score=req_model.edge_score,
                edge_embedding=req_model.edge_embedding,
                timestamp_ms=req_model.timestamp_ms,
                scene_id=req_model.scene_id,
                camera_id=req_model.camera_id,
                channel=req_model.channel,
                metadata=req_model.metadata,
            )
        except Exception as exc:
            raise HTTPException(400, f"Invalid JSON escalation payload: {exc}")
    else:
        if edge_device_id is None or edge_score is None:
            raise HTTPException(400, "Missing required fields: edge_device_id, edge_score")

        parsed_embedding = []
        if edge_embedding:
            try:
                parsed_embedding = json.loads(edge_embedding)
            except Exception:
                raise HTTPException(400, "edge_embedding must be a valid JSON-encoded float list")

        if file:
            clip_data = await file.read()
            clip_filename = file.filename or "incident_clip.mp4"

        esc_req = EscalationRequest(
            edge_device_id=edge_device_id,
            edge_score=edge_score,
            edge_embedding=parsed_embedding,
            clip_data=clip_data,
            clip_filename=clip_filename,
            timestamp_ms=timestamp_ms or 0,
            scene_id=scene_id or "",
            camera_id=camera_id or "",
            channel=channel,
        )

    result = await handle_escalation(esc_req)

    return EscalationResponseModel(
        escalation_id=result.escalation_id,
        edge_device_id=result.edge_device_id,
        edge_score=result.edge_score,
        cloud_score=result.cloud_score,
        is_confirmed_anomaly=result.is_confirmed_anomaly,
        confidence=result.confidence,
        incident_id=result.incident_id,
        ensemble_score=result.ensemble_score,
        temporal_boost=result.temporal_boost,
        latency_ms=result.latency_ms,
    )


@app.get("/api/v1/escalations/stats")
@app.get("/api/escalations/stats")
async def escalation_stats():
    """Returns global escalation metrics and confirmation rate."""
    return escalation_tracker.summary()


# ---------------------------------------------------------------------------
# Edge Device Management & Camera Catalog
# ---------------------------------------------------------------------------


@app.get("/api/v1/edges", response_model=List[EdgeDeviceModel])
@app.get("/api/edges", response_model=List[EdgeDeviceModel])
async def list_edges():
    """Lists all registered edge nodes and camera channels."""
    devices = edge_registry.list_all()
    return [
        EdgeDeviceModel(
            device_id=d.device_id,
            name=d.name,
            location=d.location,
            status=d.status,
            baseline_version=d.baseline_version,
            threshold=d.threshold,
            stats=d.stats(),
        )
        for d in devices
    ]


@app.post("/api/v1/edges/register", response_model=EdgeDeviceModel)
@app.post("/api/edges/register", response_model=EdgeDeviceModel)
async def register_edge(req: EdgeRegisterRequest):
    """Registers a new edge node or camera device."""
    record = edge_registry.register(
        name=req.name,
        location=req.location,
        device_id=req.device_id,
        threshold=req.threshold,
    )
    return EdgeDeviceModel(
        device_id=record.device_id,
        name=record.name,
        location=record.location,
        status=record.status,
        baseline_version=record.baseline_version,
        threshold=record.threshold,
        stats=record.stats(),
    )


@app.get("/api/v1/edges/{device_id}/stats")
@app.get("/api/edges/{device_id}/stats")
async def edge_stats_endpoint(device_id: str):
    """Retrieves operational telemetry and confirmation accuracy for a specific device."""
    device = edge_registry.get(device_id)
    if not device:
        raise HTTPException(404, f"Device '{device_id}' not found")

    return {
        **device.stats(),
        "escalation_details": escalation_tracker.device_stats(device_id),
        "ensemble_threshold": ensemble_scorer.get_device_threshold(device_id),
    }


@app.get("/api/v1/cameras", response_model=List[CameraStreamInfo])
@app.get("/api/cameras", response_model=List[CameraStreamInfo])
async def list_cameras():
    """Lists all 9 Dahua surveillance camera channels and active stream endpoints."""
    cameras = config.parse_camera_names()
    results = []
    for ch, name in cameras.items():
        substream = f"http://{config.edge_host}:{config.edge_port}/api/v1/stream/{ch}/live"
        snapshot = f"http://{config.edge_host}:{config.edge_port}/api/v1/stream/{ch}/snapshot"
        results.append(
            CameraStreamInfo(
                channel=ch,
                name=name,
                status="active",
                substream_url=substream,
                snapshot_url=snapshot,
                latest_score=0.0,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Twelve Labs Semantic Search & Video Q&A Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/search", response_model=Dict[str, Any])
@app.post("/api/search", response_model=Dict[str, Any])
async def search_endpoint(query: str = Form(...), max_clips: int = Form(10)):
    """Executes semantic natural-language video search via Twelve Labs Marengo."""
    if not twelvelabs_client.is_enabled():
        raise HTTPException(503, "TWELVE_LABS_API_KEY is not configured")

    try:
        results = twelvelabs_client.search_videos(query=query, max_clips=max_clips)
        return {
            "query": query,
            "results": [
                SearchResultModel(
                    video_id=r.video_id,
                    score=r.score,
                    start=r.start,
                    end=r.end,
                    confidence=r.confidence,
                    metadata=r.metadata,
                ).model_dump()
                for r in results
            ],
            "count": len(results),
        }
    except Exception as exc:
        raise HTTPException(500, f"Marengo search failed: {exc}")


@app.post("/api/v1/analyze", response_model=AnalysisResultModel)
@app.post("/api/analyze", response_model=AnalysisResultModel)
async def analyze_endpoint(video_id: str = Form(...), prompt: str = Form(...)):
    """Generates structured incident explanations and answers surveillance questions via Pegasus."""
    if not twelvelabs_client.is_enabled():
        raise HTTPException(503, "TWELVE_LABS_API_KEY is not configured")

    try:
        result = twelvelabs_client.analyze_video(video_id=video_id, prompt=prompt)
        return AnalysisResultModel(
            video_id=result.video_id,
            text=result.text,
            latency_ms=result.latency_ms,
        )
    except Exception as exc:
        raise HTTPException(500, f"Pegasus analysis failed: {exc}")


@app.get("/api/v1/twelvelabs/status")
@app.get("/api/twelvelabs/status")
async def twelvelabs_status():
    """Returns Twelve Labs API status and configured models."""
    return {
        "enabled": twelvelabs_client.is_enabled(),
        "marengo_model": config.marengo_model,
        "pegasus_model": config.pegasus_model,
        "marengo_index": config.marengo_index_name,
        "pegasus_index": config.pegasus_index_name,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.central_host, port=config.central_port)
