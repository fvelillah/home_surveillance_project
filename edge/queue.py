"""
Crash-Resilient SQLite Offline Queue & Auto-Drain Worker.

Persists escalated anomaly incidents locally to disk using SQLite with Write-Ahead Logging (WAL)
to survive power loss, process crashes, and internet outages. Runs an asynchronous background drain
worker that forwards queued incidents to the Central Analytics Backend as soon as network connectivity
is available.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

from edge.config import EdgeConfig, config as default_config

logger = logging.getLogger(__name__)


@dataclass
class IncidentItem:
    """An escalated anomaly incident payload stored in the offline queue."""
    channel: int
    camera_name: str
    timestamp: float
    score: float
    clip_path: str
    incident_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    snapshot_path: Optional[str] = None
    embedding: Optional[List[float]] = None
    status: str = "PENDING"  # PENDING, SYNCING, SYNCED, FAILED
    retry_count: int = 0
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    synced_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class SQLiteOfflineQueue:
    """
    Crash-resilient, thread-safe SQLite persistence queue for escalated incidents.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        cfg: Optional[EdgeConfig] = None,
    ) -> None:
        self.config = cfg or default_config
        raw_path = db_path or self.config.sqlite_queue_path
        self.db_path = Path(raw_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        # Enable Write-Ahead Logging for high concurrency & crash durability
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Create database tables and indices if not already present."""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS offline_incidents (
                            incident_id TEXT PRIMARY KEY,
                            channel INTEGER NOT NULL,
                            camera_name TEXT NOT NULL,
                            timestamp REAL NOT NULL,
                            score REAL NOT NULL,
                            clip_path TEXT NOT NULL,
                            snapshot_path TEXT,
                            embedding_json TEXT,
                            status TEXT DEFAULT 'PENDING',
                            retry_count INTEGER DEFAULT 0,
                            error_message TEXT,
                            created_at REAL NOT NULL,
                            synced_at REAL
                        )
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_status_created ON offline_incidents(status, created_at);"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_channel ON offline_incidents(channel);"
                    )
            finally:
                conn.close()
            logger.info("Initialized SQLite Offline Queue at %s", self.db_path)

    # ----------------------------------------------------------------------
    # Enqueue & State Transitions
    # ----------------------------------------------------------------------

    def enqueue(self, item: IncidentItem) -> bool:
        """Insert a newly escalated incident into the durable queue."""
        emb_json = json.dumps(item.embedding) if item.embedding is not None else None
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO offline_incidents (
                            incident_id, channel, camera_name, timestamp, score,
                            clip_path, snapshot_path, embedding_json, status,
                            retry_count, error_message, created_at, synced_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item.incident_id,
                            item.channel,
                            item.camera_name,
                            item.timestamp,
                            item.score,
                            item.clip_path,
                            item.snapshot_path,
                            emb_json,
                            item.status,
                            item.retry_count,
                            item.error_message,
                            item.created_at,
                            item.synced_at,
                        ),
                    )
                logger.info(
                    "Enqueued incident %s (Ch %d, score=%.3f) into SQLite offline queue",
                    item.incident_id,
                    item.channel,
                    item.score,
                )
                return True
            except sqlite3.IntegrityError:
                logger.warning("Incident %s already in offline queue", item.incident_id)
                return False
            except Exception as e:
                logger.error("Failed to enqueue incident %s: %s", item.incident_id, e)
                return False
            finally:
                conn.close()

    def peek_pending(self, limit: int = 10) -> List[IncidentItem]:
        """Fetch oldest pending incidents ready for cloud synchronization."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    """
                    SELECT * FROM offline_incidents
                    WHERE status = 'PENDING'
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
                items: List[IncidentItem] = []
                for r in rows:
                    emb = json.loads(r["embedding_json"]) if r["embedding_json"] else None
                    items.append(
                        IncidentItem(
                            incident_id=r["incident_id"],
                            channel=r["channel"],
                            camera_name=r["camera_name"],
                            timestamp=r["timestamp"],
                            score=r["score"],
                            clip_path=r["clip_path"],
                            snapshot_path=r["snapshot_path"],
                            embedding=emb,
                            status=r["status"],
                            retry_count=r["retry_count"],
                            error_message=r["error_message"],
                            created_at=r["created_at"],
                            synced_at=r["synced_at"],
                        )
                    )
                return items
            finally:
                conn.close()

    def mark_syncing(self, incident_ids: List[str]) -> None:
        """Mark incidents as currently in-flight."""
        if not incident_ids:
            return
        placeholders = ",".join("?" for _ in incident_ids)
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        f"UPDATE offline_incidents SET status = 'SYNCING' WHERE incident_id IN ({placeholders})",
                        incident_ids,
                    )
            finally:
                conn.close()

    def mark_synced(self, incident_id: str) -> None:
        """Mark incident as successfully acknowledged and synchronized by cloud backend."""
        now = time.time()
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        """
                        UPDATE offline_incidents
                        SET status = 'SYNCED', synced_at = ?, error_message = NULL
                        WHERE incident_id = ?
                        """,
                        (now, incident_id),
                    )
            finally:
                conn.close()
        logger.info("Marked incident %s as SYNCED", incident_id)

    def mark_failed(self, incident_id: str, error_msg: str, max_retries: int = 5) -> None:
        """Mark incident synchronization attempt as failed, incrementing retry counter."""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute(
                        """
                        UPDATE offline_incidents
                        SET retry_count = retry_count + 1,
                            status = CASE WHEN retry_count + 1 >= ? THEN 'FAILED' ELSE 'PENDING' END,
                            error_message = ?
                        WHERE incident_id = ?
                        """,
                        (max_retries, error_msg, incident_id),
                    )
            finally:
                conn.close()
        logger.warning("Incident %s sync failed: %s", incident_id, error_msg)

    def get_stats(self) -> Dict[str, int]:
        """Return counts of incidents categorized by status."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    """
                    SELECT status, count(*) as count
                    FROM offline_incidents
                    GROUP BY status
                    """
                )
                counts = {r["status"]: r["count"] for r in cursor.fetchall()}
                total = sum(counts.values())
                return {
                    "total": total,
                    "pending": counts.get("PENDING", 0),
                    "syncing": counts.get("SYNCING", 0),
                    "synced": counts.get("SYNCED", 0),
                    "failed": counts.get("FAILED", 0),
                }
            finally:
                conn.close()

    def purge_synced(self, older_than_days: int = 7) -> int:
        """Purge successfully synced incidents older than the specified retention window."""
        cutoff_ts = time.time() - (older_than_days * 86400)
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    cur = conn.execute(
                        "DELETE FROM offline_incidents WHERE status = 'SYNCED' AND synced_at < ?",
                        (cutoff_ts,),
                    )
                    return cur.rowcount
            finally:
                conn.close()


# ----------------------------------------------------------------------
# Auto-Drain Background Sync Worker
# ----------------------------------------------------------------------

class AutoDrainWorker:
    """
    Background worker thread that automatically drains the SQLite offline queue
    and forwards incidents to the Central Cloud Backend.
    """

    def __init__(
        self,
        queue: SQLiteOfflineQueue,
        central_url: Optional[str] = None,
        sync_interval_s: Optional[float] = None,
        max_retries: Optional[int] = None,
        cfg: Optional[EdgeConfig] = None,
    ) -> None:
        self.config = cfg or default_config
        self.queue = queue
        self.central_url = (central_url or self.config.central_backend_url).rstrip("/")
        self.sync_interval_s = sync_interval_s or self.config.drain_sync_interval_s
        self.max_retries = max_retries or self.config.drain_max_retries

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the background auto-drain worker."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._drain_loop,
            name="AutoDrainWorker",
            daemon=True,
        )
        self._thread.start()
        logger.info("AutoDrainWorker started (target: %s, interval=%.1fs)", self.central_url, self.sync_interval_s)

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the background worker."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("AutoDrainWorker stopped")

    def _drain_loop(self) -> None:
        """Continuous polling and synchronization loop."""
        while self._running and not self._stop_event.is_set():
            try:
                self.drain_once()
            except Exception as e:
                logger.error("Unexpected error in AutoDrainWorker loop: %s", e)

            self._stop_event.wait(self.sync_interval_s)

    def drain_once(self) -> int:
        """Attempt to synchronize one batch of pending incidents to cloud backend."""
        pending = self.queue.peek_pending(limit=5)
        if not pending:
            return 0

        synced_count = 0
        endpoint = f"{self.central_url}/api/v1/escalate"

        for item in pending:
            if self._stop_event.is_set():
                break

            self.queue.mark_syncing([item.incident_id])

            payload = {
                "incident_id": item.incident_id,
                "channel": item.channel,
                "camera_name": item.camera_name,
                "timestamp": item.timestamp,
                "anomaly_score": item.score,
                "clip_path": item.clip_path,
                "snapshot_path": item.snapshot_path,
                "embedding": item.embedding,
                "edge_created_at": item.created_at,
            }

            try:
                resp = requests.post(endpoint, json=payload, timeout=5.0)
                if resp.status_code in (200, 201, 202):
                    self.queue.mark_synced(item.incident_id)
                    synced_count += 1
                else:
                    err = f"Central backend returned HTTP {resp.status_code}: {resp.text[:100]}"
                    self.queue.mark_failed(item.incident_id, err, max_retries=self.max_retries)
            except requests.exceptions.RequestException as e:
                err = f"Network connection failed: {e}"
                self.queue.mark_failed(item.incident_id, err, max_retries=self.max_retries)

        return synced_count
