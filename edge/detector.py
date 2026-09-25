"""
Qdrant Edge Two-Shard Anomaly Detector.

Implements real-time edge anomaly scoring across surveillance channels using a dual-shard architecture:
  1. Immutable Baseline Shard: Pre-calibrated normal activity patterns per camera feed.
  2. Mutable Recent Context Shard: Rolling FIFO window of recent clips (~30 min context).

Computes composite anomaly score S_edge = α * d_baseline + (1 - α) * d_recent and evaluates
high-recall triage threshold for cloud escalation.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from edge.config import EdgeConfig, config as default_config

logger = logging.getLogger(__name__)

# Try importing Qdrant Client
try:
    from qdrant_client import QdrantClient  # type: ignore[import-not-found,import-untyped]
    from qdrant_client.models import (  # type: ignore[import-not-found,import-untyped]
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )
    HAS_QDRANT = True
except ImportError:
    QdrantClient = None
    HAS_QDRANT = False


class InMemoryTwoShardIndex:
    """
    High-performance pure NumPy fallback for two-shard vector storage and kNN search.
    Used when qdrant-client is not installed or during lightweight testing.
    """

    def __init__(self, embedding_dim: int = 576) -> None:
        self.embedding_dim = embedding_dim
        self._lock = threading.Lock()
        # baseline: {channel: [(id, vector, metadata), ...]}
        self.baseline: Dict[int, List[Tuple[str, np.ndarray, dict]]] = {}
        # recent: {channel: [(id, vector, metadata), ...]}
        self.recent: Dict[int, List[Tuple[str, np.ndarray, dict]]] = {}

    def insert_baseline(self, channel: int, vectors: List[np.ndarray], metas: Optional[List[dict]] = None) -> int:
        with self._lock:
            if channel not in self.baseline:
                self.baseline[channel] = []
            for i, vec in enumerate(vectors):
                pt_id = str(uuid.uuid4())
                meta = metas[i] if metas and i < len(metas) else {}
                self.baseline[channel].append((pt_id, vec.astype(np.float32), meta))
            return len(vectors)

    def insert_recent(self, channel: int, pt_id: str, vector: np.ndarray, meta: dict, max_points: int = 200) -> None:
        with self._lock:
            if channel not in self.recent:
                self.recent[channel] = []
            self.recent[channel].append((pt_id, vector.astype(np.float32), meta))
            # FIFO eviction if capacity exceeded
            if len(self.recent[channel]) > max_points:
                excess = len(self.recent[channel]) - max_points
                self.recent[channel] = self.recent[channel][excess:]

    def search_knn(self, channel: int, query_vec: np.ndarray, shard: str, top_k: int = 5) -> List[Tuple[float, dict]]:
        """Compute cosine distances to top-k vectors in the shard for channel."""
        with self._lock:
            target = self.baseline.get(channel, []) if shard == "baseline" else self.recent.get(channel, [])
            if not target:
                return []

            vecs = np.stack([item[1] for item in target], axis=0)  # (N, D)
            # Query is L2 normalized, vecs are L2 normalized -> cosine sim = vecs @ query
            q = query_vec.astype(np.float32)
            q_norm = np.linalg.norm(q)
            if q_norm > 1e-6:
                q = q / q_norm

            sims = vecs @ q
            # Cosine distance = 1 - cosine similarity
            dists = np.clip(1.0 - sims, 0.0, 2.0)

            # Sort indices ascending by distance
            sorted_indices = np.argsort(dists)[:top_k]
            results = []
            for idx in sorted_indices:
                results.append((float(dists[idx]), target[idx][2]))
            return results

    def get_stats(self) -> dict:
        with self._lock:
            baseline_count = sum(len(v) for v in self.baseline.values())
            recent_count = sum(len(v) for v in self.recent.values())
            return {
                "engine": "in_memory_numpy",
                "baseline_points": baseline_count,
                "recent_points": recent_count,
                "channels": list(set(list(self.baseline.keys()) + list(self.recent.keys()))),
            }


class QdrantEdgeDetector:
    """
    Two-Shard Anomaly Detector powered by Qdrant Edge (with NumPy in-memory fallback).
    """

    BASELINE_COLLECTION = "edge_baseline"
    RECENT_COLLECTION = "edge_recent_context"

    def __init__(
        self,
        qdrant_path: Optional[Union[str, Path]] = None,
        qdrant_url: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        triage_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        alpha_baseline: Optional[float] = None,
        max_recent_points: Optional[int] = None,
        cfg: Optional[EdgeConfig] = None,
    ) -> None:
        self.config = cfg or default_config
        self.embedding_dim = embedding_dim or self.config.embedding_dim
        self.triage_threshold = triage_threshold or self.config.edge_triage_threshold
        self.top_k = top_k or self.config.edge_top_k
        self.alpha = alpha_baseline or self.config.edge_alpha_baseline
        self.max_recent_points = max_recent_points or self.config.edge_mutable_max_points

        raw_path = qdrant_path or self.config.qdrant_edge_path
        self.qdrant_path = Path(raw_path) if raw_path else None
        self.qdrant_url = qdrant_url or self.config.qdrant_edge_url

        self.client: Optional[QdrantClient] = None
        self.use_fallback = False
        self._fallback_index = InMemoryTwoShardIndex(embedding_dim=self.embedding_dim)
        self._lock = threading.Lock()

        # Channel recent context tracking for FIFO pruning
        self._recent_point_ids: Dict[int, List[str]] = {}

        self._init_client()

    def _init_client(self) -> None:
        """Initialize Qdrant client connection and collections."""
        if not HAS_QDRANT:
            logger.info("qdrant-client not installed; using pure-NumPy Two-Shard engine.")
            self.use_fallback = True
            return

        try:
            if self.qdrant_url:
                logger.info("Connecting to Qdrant Edge via URL: %s", self.qdrant_url)
                self.client = QdrantClient(url=self.qdrant_url, timeout=5.0)
            elif self.qdrant_path:
                if str(self.qdrant_path) == ":memory:":
                    self.client = QdrantClient(":memory:")
                else:
                    self.qdrant_path.mkdir(parents=True, exist_ok=True)
                    self.client = QdrantClient(path=str(self.qdrant_path))
                logger.info("Connected to local embedded Qdrant at: %s", self.qdrant_path)
            else:
                self.client = QdrantClient(":memory:")

            self._ensure_collections()
            self.use_fallback = False
        except Exception as e:
            logger.warning("Could not initialize QdrantClient (%s). Falling back to InMemoryTwoShardIndex.", e)
            self.use_fallback = True
            self.client = None

    def _ensure_collections(self) -> None:
        """Ensure both baseline and recent collections exist in Qdrant with Cosine distance."""
        if not self.client:
            return

        existing = [c.name for c in self.client.get_collections().collections]

        for col_name in [self.BASELINE_COLLECTION, self.RECENT_COLLECTION]:
            if col_name not in existing:
                self.client.create_collection(
                    collection_name=col_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection: %s (dim=%d)", col_name, self.embedding_dim)

    # ----------------------------------------------------------------------
    # Baseline Shard Management (Immutable Normal Routine)
    # ----------------------------------------------------------------------

    def seed_baseline(
        self,
        channel: int,
        vectors: List[np.ndarray],
        metadata: Optional[List[dict]] = None,
    ) -> int:
        """
        Index normal baseline vectors for a camera channel into the immutable baseline shard.
        """
        if not vectors:
            return 0

        if self.use_fallback or not self.client:
            return self._fallback_index.insert_baseline(channel, vectors, metadata)

        points = []
        for i, vec in enumerate(vectors):
            pt_id = str(uuid.uuid4())
            meta = metadata[i] if metadata and i < len(metadata) else {}
            meta["channel"] = channel
            meta["shard"] = "baseline"
            meta["indexed_at"] = time.time()

            points.append(
                PointStruct(
                    id=pt_id,
                    vector=vec.tolist(),
                    payload=meta,
                )
            )

        self.client.upsert(
            collection_name=self.BASELINE_COLLECTION,
            points=points,
        )
        logger.info("Seeded %d baseline vectors for channel %d in Qdrant Edge", len(points), channel)
        return len(points)

    # ----------------------------------------------------------------------
    # Anomaly Scoring & Recent Context Insertion
    # ----------------------------------------------------------------------

    def score_clip(
        self,
        channel: int,
        embedding: np.ndarray,
        clip_id: str,
        timestamp: float,
    ) -> Tuple[float, dict]:
        """
        Score an incoming video clip embedding against baseline and recent shards.

        Returns:
            (anomaly_score, details_dict)
            anomaly_score is normalized in [0.0, 1.0].
        """
        vec = embedding.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm

        d_baseline, d_recent = 0.0, 0.0
        baseline_matches: List[dict] = []
        recent_matches: List[dict] = []

        if self.use_fallback or not self.client:
            b_results = self._fallback_index.search_knn(channel, vec, shard="baseline", top_k=self.top_k)
            r_results = self._fallback_index.search_knn(channel, vec, shard="recent", top_k=self.top_k)

            if b_results:
                d_baseline = float(np.mean([d for d, _ in b_results]))
                baseline_matches = [m for _, m in b_results]
            if r_results:
                d_recent = float(np.mean([d for d, _ in r_results]))
                recent_matches = [m for _, m in r_results]

            # Insert into fallback recent shard
            self._fallback_index.insert_recent(
                channel=channel,
                pt_id=clip_id,
                vector=vec,
                meta={"channel": channel, "clip_id": clip_id, "timestamp": timestamp},
                max_points=self.max_recent_points,
            )
        else:
            # Query Qdrant
            ch_filter = Filter(must=[FieldCondition(key="channel", match=MatchValue(value=channel))])

            # 1. Baseline Shard Query
            b_search = []
            if hasattr(self.client, "query_points"):
                b_search = self.client.query_points(
                    collection_name=self.BASELINE_COLLECTION,
                    query=vec.tolist(),
                    query_filter=ch_filter,
                    limit=self.top_k,
                ).points
            elif hasattr(self.client, "search"):
                b_search = self.client.search(
                    collection_name=self.BASELINE_COLLECTION,
                    query_vector=vec.tolist(),
                    query_filter=ch_filter,
                    limit=self.top_k,
                )
            if b_search:
                # In Qdrant COSINE distance mode: score = 1 - cosine_distance, so distance = 1 - score
                # or score is cosine similarity directly (depending on Qdrant version)
                # Qdrant cosine similarity returns scores in [-1, 1], distance = 1.0 - score
                dists = [max(0.0, 1.0 - hit.score) for hit in b_search]
                d_baseline = float(np.mean(dists))
                baseline_matches = [hit.payload for hit in b_search]

            # 2. Recent Context Shard Query
            r_search = []
            if hasattr(self.client, "query_points"):
                r_search = self.client.query_points(
                    collection_name=self.RECENT_COLLECTION,
                    query=vec.tolist(),
                    query_filter=ch_filter,
                    limit=self.top_k,
                ).points
            elif hasattr(self.client, "search"):
                r_search = self.client.search(
                    collection_name=self.RECENT_COLLECTION,
                    query_vector=vec.tolist(),
                    query_filter=ch_filter,
                    limit=self.top_k,
                )
            if r_search:
                dists = [max(0.0, 1.0 - hit.score) for hit in r_search]
                d_recent = float(np.mean(dists))
                recent_matches = [hit.payload for hit in r_search]

            # 3. Insert into Recent Context Shard
            pt_id = str(uuid.uuid4())
            self.client.upsert(
                collection_name=self.RECENT_COLLECTION,
                points=[
                    PointStruct(
                        id=pt_id,
                        vector=vec.tolist(),
                        payload={
                            "channel": channel,
                            "clip_id": clip_id,
                            "timestamp": timestamp,
                            "shard": "recent",
                        },
                    )
                ],
            )

            # Enforce mutable capacity via FIFO pruning
            with self._lock:
                if channel not in self._recent_point_ids:
                    self._recent_point_ids[channel] = []
                self._recent_point_ids[channel].append(pt_id)
                if len(self._recent_point_ids[channel]) > self.max_recent_points:
                    old_id = self._recent_point_ids[channel].pop(0)
                    try:
                        self.client.delete(
                            collection_name=self.RECENT_COLLECTION,
                            points_selector=[old_id],
                        )
                    except Exception as e:
                        logger.debug("Failed to prune old point %s: %s", old_id, e)

        # Composite anomaly calculation
        has_baseline = bool(baseline_matches)
        has_recent = bool(recent_matches)

        if has_baseline and has_recent:
            composite_score = self.alpha * d_baseline + (1.0 - self.alpha) * d_recent
        elif has_baseline:
            composite_score = d_baseline
        elif has_recent:
            composite_score = d_recent
        else:
            # Unseeded cold start: return 0.0 baseline distance
            composite_score = 0.0

        # Normalization to [0.0, 1.0]
        score_normalized = float(np.clip(composite_score, 0.0, 1.0))
        is_escalated = self.is_escalation(score_normalized)

        details = {
            "channel": channel,
            "clip_id": clip_id,
            "timestamp": timestamp,
            "anomaly_score": score_normalized,
            "is_escalated": is_escalated,
            "triage_threshold": self.triage_threshold,
            "d_baseline": d_baseline,
            "d_recent": d_recent,
            "has_baseline": has_baseline,
            "has_recent": has_recent,
            "engine": "qdrant" if not self.use_fallback else "numpy_fallback",
        }

        return score_normalized, details

    def is_escalation(self, score: float) -> bool:
        """Check if score meets or exceeds the edge high-recall triage threshold."""
        return score >= self.triage_threshold

    def get_stats(self) -> dict:
        """Get aggregate telemetry for both shards."""
        if self.use_fallback or not self.client:
            return self._fallback_index.get_stats()

        try:
            b_info = self.client.get_collection(self.BASELINE_COLLECTION)
            r_info = self.client.get_collection(self.RECENT_COLLECTION)
            return {
                "engine": "qdrant_edge",
                "baseline_points": b_info.points_count,
                "recent_points": r_info.points_count,
                "triage_threshold": self.triage_threshold,
                "top_k": self.top_k,
                "alpha_baseline": self.alpha,
            }
        except Exception as e:
            return {
                "engine": "qdrant_edge",
                "error": str(e),
                "triage_threshold": self.triage_threshold,
            }

    def close(self) -> None:
        """Close Qdrant client connection if open."""
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
