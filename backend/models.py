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
