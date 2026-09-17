"""Memory Governor & Anti-Poisoning Engine for Vector Baseline Management with Qdrant Staging."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .anomaly import get_collection_stats, get_qdrant, index_baseline_vectors
from .config import config
from .models import MemoryStatsResponse, QuarantineItemModel

logger = logging.getLogger(__name__)


def _to_point_id(vector_id: str) -> str:
    """Deterministically converts string vector_id to a valid UUID string for Qdrant."""
    try:
        uuid.UUID(str(vector_id))
        return str(vector_id)
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(vector_id)))


class MemoryGovernor:
    """Safeguards the normal vector baseline from environmental concept drift and data poisoning.

    Uses a dedicated staging collection in Qdrant for persistent quarantine storage across
    restarts, maintaining an in-memory cache for fast access and standalone fallback.
    """

    def __init__(
        self,
        quarantine_collection_name: Optional[str] = None,
        quarantine_duration_s: Optional[int] = None,
        max_vectors_per_camera: Optional[int] = None,
        retention_days: Optional[int] = None,
        anti_poisoning_threshold: Optional[float] = None,
    ):
        self.quarantine_collection_name = (
            quarantine_collection_name
            if quarantine_collection_name is not None
            else config.quarantine_collection_name
        )
        self.quarantine_duration_s = (
            quarantine_duration_s
            if quarantine_duration_s is not None
            else config.quarantine_duration_s
        )
        self.max_vectors_per_camera = (
            max_vectors_per_camera
            if max_vectors_per_camera is not None
            else config.max_vectors_per_camera
        )
        self.retention_days = (
            retention_days
            if retention_days is not None
            else config.baseline_retention_days
        )
        self.anti_poisoning_threshold = (
            anti_poisoning_threshold
            if anti_poisoning_threshold is not None
            else config.anti_poisoning_threshold
        )

        self._quarantine: Dict[str, QuarantineItemModel] = {}
        self._raw_vectors: Dict[str, List[float]] = {}
        self._camera_vector_counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def _ensure_quarantine_collection(self, dim: int = 576):
        """Ensures the Qdrant staging quarantine collection exists."""
        client = get_qdrant()
        if client is None:
            return
        try:
            from qdrant_client.models import Distance, VectorParams
            collections = [c.name for c in client.get_collections().collections]
            if self.quarantine_collection_name not in collections:
                client.create_collection(
                    collection_name=self.quarantine_collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        except Exception as exc:
            logger.warning("Failed to ensure quarantine collection in Qdrant: %s", exc)

    def validate_candidate(
        self,
        vector: List[float],
        camera_id: str,
        anomaly_score: float,
    ) -> Tuple[bool, str]:
        """Validates a candidate vector to prevent poisoning the baseline with anomalous frames."""
        if not vector or len(vector) == 0:
            return False, "Candidate vector is empty"

        arr = np.array(vector, dtype=np.float32)
        if np.isnan(arr).any() or np.isinf(arr).any():
            return False, "Candidate vector contains NaN or Inf values"

        norm = np.linalg.norm(arr)
        if norm < 1e-6:
            return False, "Candidate vector norm is zero"

        if anomaly_score >= self.anti_poisoning_threshold:
            return (
                False,
                f"Anti-poisoning reject: anomaly score ({anomaly_score:.3f}) >= threshold ({self.anti_poisoning_threshold:.3f})",
            )

        return True, "Valid candidate"

    def stage_to_quarantine(
        self,
        vector: List[float],
        camera_id: str,
        channel: int,
        anomaly_score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[QuarantineItemModel], str]:
        """Stages a validated candidate vector into the persistent Qdrant quarantine buffer."""
        valid, reason = self.validate_candidate(vector, camera_id, anomaly_score)
        if not valid:
            logger.warning("Rejected baseline candidate for %s: %s", camera_id, reason)
            return None, reason

        now = time.time()
        vec_id = f"qvec-{uuid.uuid4().hex[:12]}"
        item = QuarantineItemModel(
            vector_id=vec_id,
            camera_id=camera_id,
            channel=channel,
            vector_preview=vector[:5] if len(vector) >= 5 else vector,
            created_at=now,
            expires_at=now + self.quarantine_duration_s,
            anomaly_score=anomaly_score,
            status="QUARANTINED",
            metadata=metadata or {},
        )

        with self._lock:
            self._quarantine[vec_id] = item
            self._raw_vectors[vec_id] = vector

        # Upsert point to Qdrant quarantine collection
        client = get_qdrant()
        if client is not None:
            try:
                from qdrant_client.models import PointStruct
                self._ensure_quarantine_collection(dim=len(vector))
                point_id = _to_point_id(vec_id)
                payload = {
                    "vector_id": vec_id,
                    "camera_id": camera_id,
                    "channel": channel,
                    "created_at": now,
                    "expires_at": item.expires_at,
                    "anomaly_score": anomaly_score,
                    "status": "QUARANTINED",
                    "metadata": metadata or {},
                }
                client.upsert(
                    collection_name=self.quarantine_collection_name,
                    points=[PointStruct(id=point_id, vector=vector, payload=payload)],
                )
            except Exception as exc:
                logger.warning("Failed to persist quarantined vector to Qdrant (%s), cached in RAM.", exc)

        logger.info(
            "Staged vector [%s] to quarantine for %s in collection '%s' (expires in %ds)",
            vec_id, camera_id, self.quarantine_collection_name, self.quarantine_duration_s,
        )
        return item, "Staged to quarantine"

    def list_quarantine(
        self,
        camera_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[QuarantineItemModel]:
        """Lists currently quarantined candidate vectors from Qdrant or in-memory cache."""
        client = get_qdrant()
        if client is not None:
            try:
                from qdrant_client.models import FieldCondition, Filter, MatchValue
                must_conditions = []
                if camera_id:
                    must_conditions.append(FieldCondition(key="camera_id", match=MatchValue(value=str(camera_id))))
                if status:
                    must_conditions.append(FieldCondition(key="status", match=MatchValue(value=status.upper())))

                q_filter = Filter(must=must_conditions) if must_conditions else None

                records, _ = client.scroll(
                    collection_name=self.quarantine_collection_name,
                    scroll_filter=q_filter,
                    limit=1000,
                    with_payload=True,
                    with_vectors=True,
                )
                items = []
                for rec in records:
                    p = rec.payload or {}
                    vec = rec.vector if isinstance(rec.vector, list) else []
                    vec_id = p.get("vector_id", str(rec.id))
                    item = QuarantineItemModel(
                        vector_id=vec_id,
                        camera_id=p.get("camera_id", ""),
                        channel=p.get("channel", 1),
                        vector_preview=vec[:5] if len(vec) >= 5 else vec,
                        created_at=p.get("created_at", 0.0),
                        expires_at=p.get("expires_at", 0.0),
                        anomaly_score=p.get("anomaly_score", 0.0),
                        status=p.get("status", "QUARANTINED"),
                        metadata=p.get("metadata", {}),
                    )
                    items.append(item)
                    with self._lock:
                        self._quarantine[vec_id] = item
                        if vec:
                            self._raw_vectors[vec_id] = vec

                items.sort(key=lambda x: x.created_at, reverse=True)
                return items
            except Exception as exc:
                logger.debug("Qdrant quarantine query failed (%s), using in-memory cache.", exc)

        with self._lock:
            items = list(self._quarantine.values())

        if camera_id:
            items = [i for i in items if i.camera_id == camera_id]
        if status:
            s_up = status.upper()
            items = [i for i in items if i.status.upper() == s_up]

        items.sort(key=lambda x: x.created_at, reverse=True)
        return items

    def promote_quarantined_vectors(
        self,
        vector_ids: Optional[List[str]] = None,
        camera_id: Optional[str] = None,
        force: bool = False,
    ) -> int:
        """Promotes matured or manually approved quarantined vectors into the active baseline index."""
        now = time.time()

        # Ensure in-memory cache is hydrated from Qdrant if available
        self.list_quarantine(camera_id=camera_id)

        to_promote: List[QuarantineItemModel] = []
        vectors_to_index: List[List[float]] = []
        meta_to_index: List[Dict[str, Any]] = []

        with self._lock:
            for vec_id, item in list(self._quarantine.items()):
                if item.status != "QUARANTINED":
                    continue

                if vector_ids and vec_id not in vector_ids:
                    continue

                if camera_id and item.camera_id != camera_id:
                    continue

                # Matured or forced promotion
                if force or now >= item.expires_at:
                    raw_vec = self._raw_vectors.get(vec_id)
                    if raw_vec:
                        to_promote.append(item)
                        vectors_to_index.append(raw_vec)
                        meta_to_index.append({
                            "camera_id": item.camera_id,
                            "channel": item.channel,
                            "quarantine_id": vec_id,
                            "promoted_at": now,
                            **item.metadata,
                        })

            if not to_promote:
                return 0

            # Mark promoted in memory
            for item in to_promote:
                item.status = "PROMOTED"

        # Index vectors into central Qdrant active baseline collection
        try:
            total_indexed = index_baseline_vectors(
                vectors=vectors_to_index,
                metadata_list=meta_to_index,
                collection_name=config.collection_name,
            )

            # Update status in Qdrant quarantine collection
            client = get_qdrant()
            if client is not None:
                try:
                    for item in to_promote:
                        pid = _to_point_id(item.vector_id)
                        client.set_payload(
                            collection_name=self.quarantine_collection_name,
                            payload={"status": "PROMOTED", "promoted_at": now},
                            points=[pid],
                        )
                except Exception as exc:
                    logger.warning("Failed to update status in Qdrant quarantine collection: %s", exc)

            with self._lock:
                for item in to_promote:
                    cam = item.camera_id
                    self._camera_vector_counts[cam] = self._camera_vector_counts.get(cam, 0) + 1

            logger.info("Successfully promoted %d quarantined vectors to active baseline", total_indexed)
            return total_indexed
        except Exception as exc:
            logger.error("Failed to index promoted baseline vectors: %s", exc)
            with self._lock:
                for item in to_promote:
                    item.status = "QUARANTINED"
            raise

    def scrub_retention(self, retention_days: Optional[int] = None) -> Dict[str, int]:
        """Scrubs expired or obsolete quarantine records older than retention period."""
        days = retention_days if retention_days is not None else self.retention_days
        cutoff = time.time() - (days * 86400.0)

        purged_quarantine = 0
        purged_ids: List[str] = []

        with self._lock:
            for vec_id, item in list(self._quarantine.items()):
                if item.created_at < cutoff or (item.status in ("PROMOTED", "REJECTED", "EXPIRED") and item.created_at < cutoff):
                    self._quarantine.pop(vec_id, None)
                    self._raw_vectors.pop(vec_id, None)
                    purged_ids.append(vec_id)
                    purged_quarantine += 1

        # Delete purged points from Qdrant quarantine collection
        client = get_qdrant()
        if client is not None and purged_ids:
            try:
                from qdrant_client.models import PointIdsList
                point_ids = [_to_point_id(vid) for vid in purged_ids]
                client.delete(
                    collection_name=self.quarantine_collection_name,
                    points_selector=PointIdsList(points=point_ids),
                )
            except Exception as exc:
                logger.warning("Failed to purge scrubbed points from Qdrant quarantine collection: %s", exc)

        logger.info("Retention scrub completed: purged %d items older than %d days", purged_quarantine, days)
        return {
            "purged_quarantine_items": purged_quarantine,
            "retention_days": days,
        }

    def get_stats(self) -> MemoryStatsResponse:
        """Returns baseline memory telemetry, quarantine status, and per-camera counts."""
        # Ensure latest points from Qdrant are loaded
        items = self.list_quarantine()

        total_quarantined = sum(1 for i in items if i.status == "QUARANTINED")
        quarantine_by_cam: Dict[str, int] = {}
        for i in items:
            if i.status == "QUARANTINED":
                quarantine_by_cam[i.camera_id] = quarantine_by_cam.get(i.camera_id, 0) + 1

        with self._lock:
            baseline_counts = dict(self._camera_vector_counts)

        qdrant_stats = get_collection_stats()
        total_baseline = qdrant_stats.get("points_count", 0)

        return MemoryStatsResponse(
            total_quarantined=total_quarantined,
            quarantine_by_camera=quarantine_by_cam,
            total_baseline_points=total_baseline,
            baseline_by_camera=baseline_counts,
            max_vectors_per_camera=self.max_vectors_per_camera,
            retention_days=self.retention_days,
        )

    def clear(self):
        """Clears memory governor state and resets test collections."""
        with self._lock:
            self._quarantine.clear()
            self._raw_vectors.clear()
            self._camera_vector_counts.clear()

        client = get_qdrant()
        if client is not None:
            try:
                collections = [c.name for c in client.get_collections().collections]
                if self.quarantine_collection_name in collections:
                    client.delete_collection(self.quarantine_collection_name)
            except Exception as exc:
                logger.debug("Failed to delete Qdrant quarantine collection during clear: %s", exc)


# Global singleton instance
memory_governor = MemoryGovernor()

