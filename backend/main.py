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
from .copilot import copilot_engine
from .edge_registry import registry as edge_registry
from .ensemble import ensemble_scorer
from .escalation import EscalationRequest, handle_escalation, tracker as escalation_tracker
from .incidents import incident_engine
from .memory import memory_governor
from .models import (
    AnalysisResultModel,
    AnomalyResultResponse,
    BaselineIndexRequest,
    CameraStreamInfo,
    CopilotChatRequest,
    CopilotChatResponse,
    CopilotMessage,
    DailyDigestRequest,
    DailyDigestResponse,
    EdgeDeviceModel,
    EdgeRegisterRequest,
    EmbedRequest,
    EnsembleResultModel,
    EscalationRequestModel,
    EscalationResponseModel,
    HealthResponse,
    IncidentRecord,
    IncidentStatusUpdateRequest,
    LoadSheddingLevelRequest,
    LoadSheddingStatus,
    MemoryStatsResponse,
    NeighborResponse,
    QuarantineItemModel,
    QuarantinePromotionRequest,
    SearchResultModel,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchResultItem,
    VLMExplanation,
)
from .search import search_engine
from .streaming import streaming_manager
from .vlm_explainer import explain_incident
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
def score_embedding_endpoint(req: EmbedRequest):
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
def index_baseline_endpoint(req: BaselineIndexRequest):
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
def qdrant_stats_endpoint():
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
def search_endpoint(query: str = Form(...), max_clips: int = Form(10)):
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
def analyze_endpoint(video_id: str = Form(...), prompt: str = Form(...)):
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


@app.post("/api/v1/embed/text")
@app.post("/api/embed/text")
def embed_text_endpoint(text: str = Form(...)):
    """Generates high-dimensional embedding vector for query text via Twelve Labs Marengo."""
    if not twelvelabs_client.is_enabled():
        raise HTTPException(503, "TWELVE_LABS_API_KEY is not configured")

    t0 = time.perf_counter()
    emb = twelvelabs_client.create_text_embedding(text)
    if emb is None:
        raise HTTPException(500, "Failed to extract embedding from Marengo")
    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "model": config.marengo_model,
        "text": text,
        "dim": len(emb),
        "embedding_preview": emb[:5],
        "latency_ms": latency_ms,
    }


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


# ---------------------------------------------------------------------------
# Phase 4: Incident Formation & Management Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/incidents", response_model=List[IncidentRecord])
@app.get("/api/incidents", response_model=List[IncidentRecord])
def list_incidents_endpoint(
    channel: Optional[int] = None,
    status: Optional[str] = None,
    min_severity: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
):
    """Lists consolidated incidents with filtering by channel, status, and severity."""
    return incident_engine.list_incidents(
        channel=channel,
        status=status,
        min_severity=min_severity,
        limit=limit,
        offset=offset,
    )


@app.get("/api/v1/incidents/active", response_model=List[IncidentRecord])
@app.get("/api/incidents/active", response_model=List[IncidentRecord])
def active_incidents_endpoint():
    """Returns currently active (OPEN) incidents across all camera feeds."""
    return incident_engine.get_active_incidents()


@app.get("/api/v1/incidents/{incident_id}", response_model=IncidentRecord)
@app.get("/api/incidents/{incident_id}", response_model=IncidentRecord)
def get_incident_endpoint(incident_id: str):
    """Retrieves single incident details including events timeline and VLM explanation."""
    inc = incident_engine.get_incident(incident_id)
    if inc is None:
        raise HTTPException(404, f"Incident '{incident_id}' not found")
    return inc


@app.post("/api/v1/incidents/{incident_id}/status", response_model=IncidentRecord)
@app.post("/api/incidents/{incident_id}/status", response_model=IncidentRecord)
def update_incident_status_endpoint(incident_id: str, req: IncidentStatusUpdateRequest):
    """Updates incident lifecycle state (OPEN, ACKNOWLEDGED, CLOSED, ARCHIVED) and notes."""
    inc = incident_engine.update_status(
        incident_id=incident_id,
        status=req.status,
        notes=req.notes,
    )
    if inc is None:
        raise HTTPException(404, f"Incident '{incident_id}' not found")
    return inc


@app.post("/api/v1/incidents/{incident_id}/explain", response_model=IncidentRecord)
@app.post("/api/incidents/{incident_id}/explain", response_model=IncidentRecord)
async def explain_incident_endpoint(
    incident_id: str,
    video_id: Optional[str] = None,
    clip_path: Optional[str] = None,
):
    """Triggers or regenerates a natural language Pegasus VLM explanation for an existing incident."""
    inc = incident_engine.get_incident(incident_id)
    if inc is None:
        raise HTTPException(404, f"Incident '{incident_id}' not found")

    if not twelvelabs_client.is_enabled():
        raise HTTPException(503, "Twelve Labs Pegasus VLM is not enabled or configured")

    try:
        explanation = await explain_incident(incident=inc, video_id=video_id, clip_path=clip_path)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"Pegasus VLM analysis failed: {exc}")

    updated = incident_engine.attach_vlm_explanation(incident_id, explanation)
    return updated or inc


# ---------------------------------------------------------------------------
# Phase 4: Memory Governor & Quarantine Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/memory/quarantine", response_model=List[QuarantineItemModel])
@app.get("/api/memory/quarantine", response_model=List[QuarantineItemModel])
def list_quarantine_endpoint(camera_id: Optional[str] = None, status: Optional[str] = None):
    """Lists candidate normal baseline vectors held in observation quarantine."""
    return memory_governor.list_quarantine(camera_id=camera_id, status=status)


@app.post("/api/v1/memory/quarantine/stage", response_model=Dict[str, Any])
@app.post("/api/memory/quarantine/stage", response_model=Dict[str, Any])
def stage_quarantine_endpoint(
    vector: List[float],
    camera_id: str,
    channel: int = 1,
    anomaly_score: float = 0.0,
):
    """Stages a newly proposed normal vector candidate into the 1-hour quarantine buffer."""
    item, reason = memory_governor.stage_to_quarantine(
        vector=vector,
        camera_id=camera_id,
        channel=channel,
        anomaly_score=anomaly_score,
    )
    if item is None:
        raise HTTPException(400, f"Candidate vector rejected: {reason}")
    return {"status": "ok", "message": reason, "item": item.model_dump()}


@app.post("/api/v1/memory/quarantine/promote", response_model=Dict[str, Any])
@app.post("/api/memory/quarantine/promote", response_model=Dict[str, Any])
def promote_quarantine_endpoint(req: QuarantinePromotionRequest):
    """Promotes matured or approved quarantined vectors into the active Qdrant baseline."""
    count = memory_governor.promote_quarantined_vectors(
        vector_ids=req.vector_ids,
        camera_id=req.camera_id,
        force=req.force,
    )
    return {"status": "ok", "promoted_count": count}


@app.post("/api/v1/memory/scrub", response_model=Dict[str, Any])
@app.post("/api/memory/scrub", response_model=Dict[str, Any])
def scrub_retention_endpoint(retention_days: Optional[int] = None):
    """Performs scheduled retention cleanup of historical vector memory and quarantine."""
    result = memory_governor.scrub_retention(retention_days=retention_days)
    return {"status": "ok", **result}


@app.get("/api/v1/memory/stats", response_model=MemoryStatsResponse)
@app.get("/api/memory/stats", response_model=MemoryStatsResponse)
def memory_stats_endpoint():
    """Returns memory governor telemetry, per-camera vector counts, and quarantine size."""
    return memory_governor.get_stats()


# ---------------------------------------------------------------------------
# Phase 4: Streaming Backpressure & Load Shedding Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/streaming/status", response_model=LoadSheddingStatus)
@app.get("/api/streaming/status", response_model=LoadSheddingStatus)
def streaming_status_endpoint():
    """Returns real-time adaptive streaming backpressure telemetry and active tier."""
    return streaming_manager.get_status()


@app.post("/api/v1/streaming/level", response_model=LoadSheddingStatus)
@app.post("/api/streaming/level", response_model=LoadSheddingStatus)
def set_streaming_level_endpoint(req: LoadSheddingLevelRequest):
    """Updates load shedding level override (0..3) or toggles automatic mode."""
    streaming_manager.set_level(level=req.level, auto_mode=req.auto_mode)
    return streaming_manager.get_status()


# ---------------------------------------------------------------------------
# Phase 5: Semantic Video Search Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/search/semantic", response_model=SemanticSearchResponse)
@app.post("/api/search/semantic", response_model=SemanticSearchResponse)
def semantic_search_endpoint(req: SemanticSearchRequest):
    """Executes multi-modal semantic search combining Marengo visual search with incident metadata filters."""
    return search_engine.search(
        query=req.query,
        channel=req.channel,
        camera_id=req.camera_id,
        start_time=req.start_time,
        end_time=req.end_time,
        min_severity=req.min_severity,
        severity_badge=req.severity_badge,
        status=req.status,
        limit=req.limit,
        offset=req.offset,
        threshold=req.threshold,
    )


# ---------------------------------------------------------------------------
# Phase 5: Conversational AI Security Copilot Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/v1/copilot/chat", response_model=CopilotChatResponse)
@app.post("/api/copilot/chat", response_model=CopilotChatResponse)
def copilot_chat_endpoint(req: CopilotChatRequest):
    """Engages the AI Security Copilot in a conversational Q&A turn grounded in surveillance events."""
    return copilot_engine.chat(
        message=req.message,
        session_id=req.session_id,
        channel=req.channel,
        camera_id=req.camera_id,
        start_time=req.start_time,
        end_time=req.end_time,
    )


@app.get("/api/v1/copilot/history/{session_id}", response_model=List[CopilotMessage])
@app.get("/api/copilot/history/{session_id}", response_model=List[CopilotMessage])
def copilot_history_endpoint(session_id: str):
    """Retrieves full conversation turn history for a given Copilot chat session."""
    return copilot_engine.get_history(session_id)


@app.delete("/api/v1/copilot/history/{session_id}")
@app.delete("/api/copilot/history/{session_id}")
def copilot_clear_session_endpoint(session_id: str):
    """Clears conversation message history for a specific session."""
    deleted = copilot_engine.clear_history(session_id)
    return {"status": "ok", "session_id": session_id, "deleted": deleted}


# ---------------------------------------------------------------------------
# Phase 5: Daily Surveillance Summary Digest Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/digest/daily", response_model=DailyDigestResponse)
@app.get("/api/digest/daily", response_model=DailyDigestResponse)
@app.post("/api/v1/digest/generate", response_model=DailyDigestResponse)
def daily_digest_endpoint(
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    channel: Optional[int] = None,
    format: str = "full",
):
    """Generates an executive 24-hour daily surveillance report across all 9 camera channels."""
    return copilot_engine.generate_daily_digest(
        start_time=start_time,
        end_time=end_time,
        channel=channel,
        format=format,
    )


@app.get("/api/v1/digest/summary")
@app.get("/api/digest/summary")
def digest_summary_endpoint():
    """Returns compact 24-hour summary telemetry for dashboard metrics cards."""
    digest = copilot_engine.generate_daily_digest(format="executive")
    return {
        "digest_id": digest.digest_id,
        "threat_level": digest.threat_level,
        "total_incidents": digest.total_incidents,
        "critical_incidents": digest.critical_incidents,
        "high_incidents": digest.high_incidents,
        "executive_summary": digest.executive_summary,
        "generated_at": digest.generated_at,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.central_host, port=config.central_port)
