"""Pydantic data models and schemas for Central Cloud API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health & Status Schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    model_loaded: bool
    twelve_labs_enabled: bool
    edge_devices_count: int = 0
    uptime_s: float = 0.0


# ---------------------------------------------------------------------------
# Anomaly Scoring & Nearest Neighbor Schemas
# ---------------------------------------------------------------------------


class NeighborResponse(BaseModel):
    clip_id: str
    similarity: float
    source_video: str = ""
    scene_id: str = ""
    camera_id: Optional[str] = ""


class AnomalyResultResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    neighbors: List[NeighborResponse] = Field(default_factory=list)
    latency_ms: Dict[str, float] = Field(default_factory=dict)
    scoring_method: str = "visual"


class EmbedRequest(BaseModel):
    """Request to score a pre-computed embedding against the normal baseline."""
    embedding: List[float]
    collection_name: Optional[str] = None
    k: int = 5
    camera_id: Optional[str] = None
    scene_id: Optional[str] = None


class BaselineIndexRequest(BaseModel):
    """Request to batch index baseline vectors into central Qdrant."""
    vectors: List[List[float]]
    metadata: Optional[List[Dict[str, Any]]] = None
    collection_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Edge Device Registration & Escalation Schemas
# ---------------------------------------------------------------------------


class EscalationRequestModel(BaseModel):
    """Escalation payload sent by an edge device."""
    edge_device_id: str
    edge_score: float
    edge_embedding: List[float]
    timestamp_ms: int = 0
    scene_id: str = ""
    camera_id: Optional[str] = ""
    channel: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EscalationResponseModel(BaseModel):
    """Central cloud evaluation response returned to the edge device."""
    escalation_id: str
    edge_device_id: str
    edge_score: float
    cloud_score: float
    is_confirmed_anomaly: bool
    confidence: float
    incident_id: Optional[str] = None
    ensemble_score: Optional[float] = None
    temporal_boost: float = 0.0
    latency_ms: float = 0.0


class EnsembleResultModel(BaseModel):
    """Multi-model ensemble fusion evaluation output."""
    edge_score: float
    cloud_score: float
    ensemble_score: float
    confidence: float
    is_anomaly: bool
    temporal_boost: float = 0.0
    weights_used: Dict[str, float] = Field(default_factory=dict)


class EdgeDeviceModel(BaseModel):
    """Edge device status and configuration schema."""
    device_id: str
    name: str
    location: str
    status: str = "online"
    baseline_version: str = ""
    threshold: float = 0.15
    stats: Optional[Dict[str, Any]] = None


class EdgeRegisterRequest(BaseModel):
    """Request schema for registering a new edge node."""
    name: str
    location: str
    device_id: Optional[str] = None
    threshold: Optional[float] = None


class CameraStreamInfo(BaseModel):
    """Information and stream endpoints for a camera channel."""
    channel: int
    name: str
    status: str = "active"
    substream_url: Optional[str] = None
    mainstream_url: Optional[str] = None
    snapshot_url: Optional[str] = None
    latest_score: Optional[float] = None


class IncidentResponse(BaseModel):
    """Incident schema for web console and alert management."""
    incident_id: str
    channel: int
    camera_name: str
    timestamp: float
    edge_score: float
    cloud_score: float
    ensemble_score: float
    severity: int
    clip_url: Optional[str] = None
    snapshot_url: Optional[str] = None
    status: str = "new"


# ---------------------------------------------------------------------------
# Phase 4: Incident Formation & VLM Explanation Schemas
# ---------------------------------------------------------------------------


class VLMExplanation(BaseModel):
    """Structured natural language incident explanation produced by Pegasus/VLM."""
    summary: str
    actors: List[str] = Field(default_factory=list)
    action: str = ""
    objects: List[str] = Field(default_factory=list)
    risk_assessment: str = "routine"  # routine | suspicious | hazard | breach
    recommended_action: str = ""
    model: str = "pegasus1.2"
    latency_ms: float = 0.0


class IncidentEvent(BaseModel):
    """Single discrete score/clip observation within a compound incident."""
    event_id: str
    timestamp: float
    edge_score: float
    cloud_score: float
    ensemble_score: float
    smoothed_score: float
    snapshot_url: Optional[str] = None
    clip_url: Optional[str] = None


class IncidentRecord(BaseModel):
    """Full lifecycle incident representation managed by IncidentFormationEngine."""
    incident_id: str
    channel: int
    camera_id: str
    camera_name: str
    start_time: float
    end_time: float
    duration_s: float
    peak_score: float
    mean_score: float
    smoothed_score: float = 0.0
    severity: int = Field(ge=0, le=100)
    severity_badge: str = "LOW"  # LOW | MODERATE | HIGH | CRITICAL
    status: str = "OPEN"  # OPEN | ACKNOWLEDGED | CLOSED | ARCHIVED
    event_count: int = 1
    events: List[IncidentEvent] = Field(default_factory=list)
    vlm_explanation: Optional[VLMExplanation] = None
    snapshot_url: Optional[str] = None
    clip_url: Optional[str] = None
    scene_id: str = ""
    created_at: float
    updated_at: float
    notes: Optional[str] = None


class IncidentStatusUpdateRequest(BaseModel):
    """Request payload to change incident status or add operator notes."""
    status: str  # OPEN | ACKNOWLEDGED | CLOSED | ARCHIVED
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Phase 4: Memory Governor & Anti-Poisoning Schemas
# ---------------------------------------------------------------------------


class QuarantineItemModel(BaseModel):
    """Vector candidate quarantined for observation before baseline promotion."""
    vector_id: str
    camera_id: str
    channel: int
    vector_preview: List[float] = Field(default_factory=list)
    created_at: float
    expires_at: float
    anomaly_score: float
    status: str = "QUARANTINED"  # QUARANTINED | PROMOTED | REJECTED | EXPIRED
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuarantinePromotionRequest(BaseModel):
    """Request to promote quarantined vector(s) to active Qdrant baseline."""
    vector_ids: Optional[List[str]] = None
    camera_id: Optional[str] = None
    force: bool = False


class MemoryStatsResponse(BaseModel):
    """Memory governor and vector capacity telemetry."""
    total_quarantined: int
    quarantine_by_camera: Dict[str, int] = Field(default_factory=dict)
    total_baseline_points: int
    baseline_by_camera: Dict[str, int] = Field(default_factory=dict)
    max_vectors_per_camera: int
    retention_days: int


# ---------------------------------------------------------------------------
# Phase 4: Streaming Backpressure & Load Shedding Schemas
# ---------------------------------------------------------------------------


class LoadSheddingStatus(BaseModel):
    """Telemetry and active level for adaptive backpressure system."""
    current_level: int  # 0: NORMAL, 1: SCORE_ONLY, 2: PASSTHROUGH, 3: SHED_LOAD
    level_name: str
    auto_mode: bool
    avg_latency_ms: float
    queue_depth: int
    total_requests: int
    shed_requests_count: int


class LoadSheddingLevelRequest(BaseModel):
    """Request to change backpressure mode or force specific level."""
    level: Optional[int] = None
    auto_mode: Optional[bool] = None


# ---------------------------------------------------------------------------
# Twelve Labs Search & Analysis Schemas
# ---------------------------------------------------------------------------


class SearchResultModel(BaseModel):
    video_id: str
    score: float
    start: float
    end: float
    confidence: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResultModel(BaseModel):
    video_id: str
    text: str
    latency_ms: float = 0.0
