"""Central vector baseline anomaly detection via kNN search in Qdrant."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class Neighbor:
    clip_id: str
    similarity: float
    source_video: str = ""
    scene_id: str = ""
    camera_id: str = ""


@dataclass
class LatencyBreakdown:
    embed_ms: float = 0.0
    search_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class AnomalyResult:
    anomaly_score: float
    is_anomaly: bool
    neighbors: List[Neighbor] = field(default_factory=list)
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "neighbors": [
                {
                    "clip_id": n.clip_id,
                    "similarity": n.similarity,
                    "source_video": n.source_video,
                    "scene_id": n.scene_id,
                    "camera_id": n.camera_id,
                }
                for n in self.neighbors
            ],
            "latency_ms": {
                "embed": self.latency.embed_ms,
                "search": self.latency.search_ms,
                "total": self.latency.total_ms,
            },
        }


# ---------------------------------------------------------------------------
# In-Memory Baseline Index Fallback
# ---------------------------------------------------------------------------


class InMemoryCentralBaselineIndex:
    """Thread-safe pure NumPy kNN baseline index used as a failsafe when Qdrant daemon is unavailable."""

    def __init__(self):
        self._lock = threading.Lock()
        self._vectors: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []

    def clear(self):
        with self._lock:
            self._vectors.clear()
            self._metadata.clear()

    def upsert(self, vectors: List[List[float]], metadata_list: Optional[List[Dict[str, Any]]] = None) -> int:
        with self._lock:
            for i, vec in enumerate(vectors):
                v = np.array(vec, dtype=np.float32)
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
                self._vectors.append(v)
                meta = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
                if "id" not in meta:
                    meta["id"] = str(uuid.uuid4())[:8]
                self._metadata.append(meta)
            return len(self._vectors)

    def search(
        self,
        query: List[float],
        k: int = 5,
        camera_id: Optional[str] = None,
        scene_id: Optional[str] = None,
    ) -> List[Neighbor]:
        with self._lock:
            if not self._vectors:
                return []

            q = np.array(query, dtype=np.float32)
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm

            candidates = []
            for i, v in enumerate(self._vectors):
                meta = self._metadata[i]
                # Apply filter if specified
                if camera_id is not None and str(meta.get("camera_id", "")) != str(camera_id):
                    continue
                if scene_id is not None and meta.get("scene_id", "") != scene_id:
                    continue
                sim = float(np.dot(q, v))
                candidates.append((sim, meta))

            # Fallback to unfiltered if filtered matched 0 points
            if not candidates:
                for i, v in enumerate(self._vectors):
                    sim = float(np.dot(q, v))
                    candidates.append((sim, self._metadata[i]))

            candidates.sort(key=lambda x: x[0], reverse=True)
            top = candidates[:k]

            neighbors = []
            for sim, meta in top:
                neighbors.append(
                    Neighbor(
                        clip_id=str(meta.get("id", "")),
                        similarity=sim,
                        source_video=meta.get("source_video", ""),
                        scene_id=meta.get("scene_id", ""),
                        camera_id=str(meta.get("camera_id", "")),
                    )
                )
            return neighbors

    def count(self) -> int:
        with self._lock:
            return len(self._vectors)


_in_memory_index = InMemoryCentralBaselineIndex()

# ---------------------------------------------------------------------------
# Qdrant Client Management
# ---------------------------------------------------------------------------

_qdrant = None
_qdrant_lock = threading.Lock()


def get_qdrant():
    """Returns the Qdrant client singleton, initializing connection if needed."""
    global _qdrant
    if _qdrant is None:
        with _qdrant_lock:
            if _qdrant is None:
                try:
                    from qdrant_client import QdrantClient
                    _qdrant = QdrantClient(
                        url=config.qdrant_url,
                        api_key=config.qdrant_api_key,
                        timeout=5.0,
                    )
                    # Verify connectivity
                    _qdrant.get_collections()
                except Exception as exc:
                    logger.warning("Qdrant cluster unavailable at %s (%s). Using in-memory baseline fallback.", config.qdrant_url, exc)
                    _qdrant = None
    return _qdrant


def reset_qdrant_client():
    """Resets client singleton for testing purposes."""
    global _qdrant
    with _qdrant_lock:
        _qdrant = None


# ---------------------------------------------------------------------------
# Core Baseline Indexing & Scoring
# ---------------------------------------------------------------------------


def index_baseline_vectors(
    vectors: List[List[float]],
    metadata_list: Optional[List[Dict[str, Any]]] = None,
    collection_name: Optional[str] = None,
) -> int:
    """Upserts baseline vectors and metadata payloads into Central Qdrant."""
    coll = collection_name or config.collection_name
    client = get_qdrant()

    # Always keep in-memory index in sync
    _in_memory_index.upsert(vectors, metadata_list)

    if client is None:
        return _in_memory_index.count()

    try:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        # Ensure collection exists
        collections = [c.name for c in client.get_collections().collections]
        if coll not in collections:
            dim = len(vectors[0]) if vectors else 576
            client.create_collection(
                collection_name=coll,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        points = []
        for i, vec in enumerate(vectors):
            meta = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            pid = meta.get("id") or str(uuid.uuid4())
            points.append(PointStruct(id=pid, vector=vec, payload=meta))

        client.upsert(collection_name=coll, points=points)
        info = client.get_collection(coll)
        return info.points_count or len(points)
    except Exception as exc:
        logger.warning("Failed to upsert to Qdrant (%s), using in-memory baseline.", exc)
        return _in_memory_index.count()


def score_clip(
    embedding: List[float],
    collection_name: Optional[str] = None,
    k: int = 5,
    threshold: Optional[float] = None,
    scene_id: Optional[str] = None,
    camera_id: Optional[str | int] = None,
) -> AnomalyResult:
    """Scores an embedding against the normal baseline using cosine kNN distance.

    Formula:
        Anomaly Score = 1.0 - mean(cosine_sim(embedding, top_k_neighbors))
    """
    if threshold is None:
        threshold = config.cloud_anomaly_threshold

    coll = collection_name or config.collection_name
    client = get_qdrant()
    t0 = time.perf_counter()

    neighbors: List[Neighbor] = []

    if client is not None:
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            must_conditions = []
            if camera_id is not None:
                must_conditions.append(FieldCondition(key="camera_id", match=MatchValue(value=str(camera_id))))
            if scene_id:
                must_conditions.append(FieldCondition(key="scene_id", match=MatchValue(value=scene_id)))

            results = []
            if must_conditions:
                q_filter = Filter(must=must_conditions)
                results = client.query_points(
                    collection_name=coll,
                    query=embedding,
                    limit=k,
                    query_filter=q_filter,
                ).points

            # Fallback to unfiltered search
            if not results:
                results = client.query_points(
                    collection_name=coll,
                    query=embedding,
                    limit=k,
                ).points

            for r in results:
                payload = r.payload or {}
                neighbors.append(
                    Neighbor(
                        clip_id=str(r.id),
                        similarity=r.score,
                        source_video=payload.get("source_video", ""),
                        scene_id=payload.get("scene_id", ""),
                        camera_id=str(payload.get("camera_id", "")),
                    )
                )
        except Exception as exc:
            logger.warning("Qdrant query failed (%s), querying in-memory baseline.", exc)
            neighbors = _in_memory_index.search(
                query=embedding,
                k=k,
                camera_id=str(camera_id) if camera_id is not None else None,
                scene_id=scene_id,
            )
    else:
        neighbors = _in_memory_index.search(
            query=embedding,
            k=k,
            camera_id=str(camera_id) if camera_id is not None else None,
            scene_id=scene_id,
        )

    search_ms = (time.perf_counter() - t0) * 1000

    if not neighbors:
        # If no baseline exists, assume 1.0 (anomalous/unseeded)
        return AnomalyResult(
            anomaly_score=1.0,
            is_anomaly=True,
            neighbors=[],
            latency=LatencyBreakdown(search_ms=search_ms, total_ms=search_ms),
        )

    similarities = [n.similarity for n in neighbors]
    anomaly_score = max(0.0, 1.0 - (sum(similarities) / len(similarities)))

    return AnomalyResult(
        anomaly_score=anomaly_score,
        is_anomaly=anomaly_score >= threshold,
        neighbors=neighbors,
        latency=LatencyBreakdown(search_ms=search_ms, total_ms=search_ms),
    )


def get_collection_stats(collection_name: Optional[str] = None) -> Dict[str, Any]:
    """Returns statistics for the central Qdrant baseline collection."""
    coll = collection_name or config.collection_name
    client = get_qdrant()

    if client is not None:
        try:
            info = client.get_collection(coll)
            return {
                "collection": coll,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": getattr(info.status, "value", str(info.status)),
                "storage_backend": "qdrant_cluster",
            }
        except Exception as exc:
            logger.warning("Failed to get Qdrant collection stats: %s", exc)

    return {
        "collection": coll,
        "points_count": _in_memory_index.count(),
        "vectors_count": _in_memory_index.count(),
        "status": "in_memory_fallback",
        "storage_backend": "in_memory",
    }
