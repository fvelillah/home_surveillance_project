"""Incident Formation Engine with EMA smoothing, hysteresis, and cooldown merging."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from .config import config
from .models import IncidentEvent, IncidentRecord, VLMExplanation

logger = logging.getLogger(__name__)


def compute_severity(peak_score: float, mean_score: float, event_count: int) -> tuple[int, str]:
    """Computes normalized 0-100 severity integer and discrete badge."""
    # Peak (60%), Mean (25%), Frequency/Duration boost (15%)
    freq_factor = min(1.0, event_count / 5.0)
    raw_sev = (0.60 * peak_score + 0.25 * mean_score + 0.15 * freq_factor) * 100.0
    sev_int = int(round(max(0.0, min(100.0, raw_sev))))

    if sev_int >= 75:
        badge = "CRITICAL"
    elif sev_int >= 50:
        badge = "HIGH"
    elif sev_int >= 25:
        badge = "MODERATE"
    else:
        badge = "LOW"

    return sev_int, badge


class IncidentFormationEngine:
    """Orchestrates incident state transitions, score smoothing, and event consolidation."""

    def __init__(
        self,
        ema_alpha: Optional[float] = None,
        start_threshold: Optional[float] = None,
        end_threshold: Optional[float] = None,
        cooldown_window_s: Optional[float] = None,
    ):
        self.ema_alpha = ema_alpha if ema_alpha is not None else config.ema_alpha
        self.start_threshold = start_threshold if start_threshold is not None else config.incident_start_threshold
        self.end_threshold = end_threshold if end_threshold is not None else config.incident_end_threshold
        self.cooldown_window_s = cooldown_window_s if cooldown_window_s is not None else config.cooldown_window_s
        
        self._ema_scores: Dict[str, float] = {}
        self._active_incidents: Dict[str, IncidentRecord] = {}
        self._incidents: List[IncidentRecord] = []
        self._incident_by_id: Dict[str, IncidentRecord] = {}
        self._last_event_times: Dict[str, float] = {}
        self._lock = threading.Lock()

    def update_ema(self, camera_key: str, raw_score: float) -> float:
        """Calculates and updates the exponential moving average anomaly score."""
        with self._lock:
            prev = self._ema_scores.get(camera_key)
            if prev is None:
                smoothed = float(raw_score)
            else:
                smoothed = (self.ema_alpha * raw_score) + ((1.0 - self.ema_alpha) * prev)
            self._ema_scores[camera_key] = smoothed
            return smoothed

    def get_smoothed_score(self, camera_key: str) -> float:
        """Retrieves latest smoothed EMA score for a given camera channel."""
        with self._lock:
            return self._ema_scores.get(camera_key, 0.0)

    def process_escalation(
        self,
        channel: int,
        camera_id: str,
        camera_name: str,
        edge_score: float,
        cloud_score: float,
        ensemble_score: float,
        timestamp: Optional[float] = None,
        snapshot_url: Optional[str] = None,
        clip_url: Optional[str] = None,
        scene_id: str = "",
    ) -> Optional[IncidentRecord]:
        """Processes a new escalation event, applying EMA smoothing, hysteresis, and cooldown merging."""
        ts = timestamp if (timestamp is not None and timestamp > 0) else time.time()
        cam_key = f"cam-{channel}" if channel else camera_id

        smoothed = self.update_ema(cam_key, ensemble_score)

        with self._lock:
            active = self._active_incidents.get(cam_key)
            last_ts = self._last_event_times.get(cam_key, 0.0)

            # Check if active incident can be merged (within cooldown window)
            can_merge = False
            target_incident: Optional[IncidentRecord] = None

            if active is not None:
                can_merge = True
                target_incident = active
            elif last_ts > 0 and (ts - last_ts) <= self.cooldown_window_s:
                # Find most recent incident for this camera
                for inc in reversed(self._incidents):
                    if inc.channel == channel or inc.camera_id == camera_id:
                        if (ts - inc.end_time) <= self.cooldown_window_s:
                            can_merge = True
                            target_incident = inc
                            break

            event_item = IncidentEvent(
                event_id=f"evt-{uuid.uuid4().hex[:8]}",
                timestamp=ts,
                edge_score=edge_score,
                cloud_score=cloud_score,
                ensemble_score=ensemble_score,
                smoothed_score=smoothed,
                snapshot_url=snapshot_url,
                clip_url=clip_url,
            )

            # Condition 1: Merge into existing incident
            if can_merge and target_incident is not None:
                target_incident.events.append(event_item)
                target_incident.event_count = len(target_incident.events)
                target_incident.end_time = max(target_incident.end_time, ts)
                target_incident.duration_s = max(0.0, target_incident.end_time - target_incident.start_time)
                target_incident.peak_score = max(target_incident.peak_score, ensemble_score)
                target_incident.smoothed_score = smoothed

                all_scores = [e.ensemble_score for e in target_incident.events]
                target_incident.mean_score = sum(all_scores) / len(all_scores)

                sev, badge = compute_severity(
                    target_incident.peak_score,
                    target_incident.mean_score,
                    target_incident.event_count,
                )
                target_incident.severity = sev
                target_incident.severity_badge = badge
                target_incident.updated_at = time.time()
                if snapshot_url and not target_incident.snapshot_url:
                    target_incident.snapshot_url = snapshot_url
                if clip_url and not target_incident.clip_url:
                    target_incident.clip_url = clip_url

                # Check if smoothed score dropped below end_threshold to mark closing/closed
                if smoothed < self.end_threshold:
                    target_incident.status = "CLOSED"
                    self._active_incidents.pop(cam_key, None)
                else:
                    target_incident.status = "OPEN"
                    self._active_incidents[cam_key] = target_incident

                self._last_event_times[cam_key] = ts
                return target_incident

            # Condition 2: Trigger new incident if smoothed score meets or exceeds start threshold
            if smoothed >= self.start_threshold:
                incident_id = f"inc-{uuid.uuid4().hex[:10]}"
                sev, badge = compute_severity(ensemble_score, ensemble_score, 1)

                new_inc = IncidentRecord(
                    incident_id=incident_id,
                    channel=channel,
                    camera_id=camera_id,
                    camera_name=camera_name,
                    start_time=ts,
                    end_time=ts,
                    duration_s=0.0,
                    peak_score=ensemble_score,
                    mean_score=ensemble_score,
                    smoothed_score=smoothed,
                    severity=sev,
                    severity_badge=badge,
                    status="OPEN",
                    event_count=1,
                    events=[event_item],
                    snapshot_url=snapshot_url,
                    clip_url=clip_url,
                    scene_id=scene_id,
                    created_at=time.time(),
                    updated_at=time.time(),
                )

                self._active_incidents[cam_key] = new_inc
                self._incidents.append(new_inc)
                self._incident_by_id[incident_id] = new_inc
                self._last_event_times[cam_key] = ts
                logger.info(
                    "New incident formed [%s] on Ch%d (%s) - Severity: %d (%s), Score: %.3f",
                    incident_id, channel, camera_name, sev, badge, smoothed,
                )
                return new_inc

            # Score is below threshold and no active incident
            self._last_event_times[cam_key] = ts
            return None

    def get_incident(self, incident_id: str) -> Optional[IncidentRecord]:
        """Retrieves a single incident by unique ID."""
        with self._lock:
            return self._incident_by_id.get(incident_id)

    def list_incidents(
        self,
        channel: Optional[int] = None,
        status: Optional[str] = None,
        min_severity: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[IncidentRecord]:
        """Returns filtered list of recorded incidents sorted newest first."""
        with self._lock:
            res = list(self._incidents)

        if channel is not None:
            res = [r for r in res if r.channel == channel]
        if status is not None:
            s_up = status.upper()
            res = [r for r in res if r.status.upper() == s_up]
        if min_severity is not None:
            res = [r for r in res if r.severity >= min_severity]

        # Newest first
        res.sort(key=lambda x: x.created_at, reverse=True)
        return res[offset : offset + limit]

    def update_status(
        self,
        incident_id: str,
        status: str,
        notes: Optional[str] = None,
    ) -> Optional[IncidentRecord]:
        """Updates the status and optional operator notes for an incident."""
        with self._lock:
            inc = self._incident_by_id.get(incident_id)
            if inc is None:
                return None
            inc.status = status.upper()
            if notes is not None:
                inc.notes = notes
            inc.updated_at = time.time()

            # If closed or archived, remove from active dictionary
            if inc.status in ("CLOSED", "ARCHIVED"):
                cam_key = f"cam-{inc.channel}"
                if self._active_incidents.get(cam_key) == inc:
                    self._active_incidents.pop(cam_key, None)
            return inc

    def attach_vlm_explanation(
        self,
        incident_id: str,
        explanation: VLMExplanation,
    ) -> Optional[IncidentRecord]:
        """Attaches natural language VLM explanation to an incident."""
        with self._lock:
            inc = self._incident_by_id.get(incident_id)
            if inc is None:
                return None
            inc.vlm_explanation = explanation
            inc.updated_at = time.time()
            return inc

    def get_active_incidents(self) -> List[IncidentRecord]:
        """Returns currently active (OPEN) incidents across all channels."""
        with self._lock:
            return list(self._active_incidents.values())

    def reset_channel_ema(self, camera_key: str):
        """Resets the EMA score tracking for a camera channel."""
        with self._lock:
            self._ema_scores.pop(camera_key, None)

    def clear(self):
        """Clears all stored incidents and state (useful for tests)."""
        with self._lock:
            self._ema_scores.clear()
            self._active_incidents.clear()
            self._incidents.clear()
            self._incident_by_id.clear()
            self._last_event_times.clear()


# Global singleton instance
incident_engine = IncidentFormationEngine()
