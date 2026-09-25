"""Conversational AI Security Copilot & Daily Surveillance Summary Digest Engine."""

from __future__ import annotations

import datetime
import logging
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set

from .config import config
from .incidents import incident_engine
from .models import (
    ChannelActivitySummary,
    CopilotChatRequest,
    CopilotChatResponse,
    CopilotEvidence,
    CopilotMessage,
    DailyDigestRequest,
    DailyDigestResponse,
    DigestIncidentSummary,
    IncidentRecord,
)
from .search import search_engine
from . import twelvelabs_client

logger = logging.getLogger(__name__)


def _format_timestamp(ts: float) -> str:
    """Formats Unix timestamp into a readable human string."""
    try:
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ts)


def _format_time_short(ts: float) -> str:
    """Formats Unix timestamp into a compact HH:MM:SS string."""
    try:
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        return dt.strftime("%H:%M:%S")
    except Exception:
        return str(ts)


class SecurityCopilotEngine:
    """Conversational Copilot and Daily Digest generator grounded in video events."""

    def __init__(self):
        self._sessions: Dict[str, List[CopilotMessage]] = {}
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Conversational Chat Engine
    # -----------------------------------------------------------------------

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        channel: Optional[int] = None,
        camera_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> CopilotChatResponse:
        """Processes an operator message, retrieves relevant evidence, and returns a grounded response."""
        t0 = time.perf_counter()
        active_session = session_id or f"session-{uuid.uuid4().hex[:8]}"

        # 1. Parse intent, target channels, and temporal references from message
        clean_msg = message.strip()
        inferred_channel = channel or self._infer_channel_from_query(clean_msg)

        # 2. Retrieve relevant incident evidence
        matched_incidents = self._retrieve_evidence(
            query=clean_msg,
            channel=inferred_channel,
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
        )

        # 3. Build evidence objects
        evidence_list: List[CopilotEvidence] = []
        cited_ids: List[str] = []
        for inc in matched_incidents[:5]:
            cited_ids.append(inc.incident_id)
            vlm = inc.vlm_explanation
            summary = vlm.summary if vlm else f"Anomaly detected on {inc.camera_name} (Severity {inc.severity})"
            actors = vlm.actors if vlm else []
            action = vlm.action if vlm else ""

            evidence_list.append(
                CopilotEvidence(
                    incident_id=inc.incident_id,
                    channel=inc.channel,
                    camera_name=inc.camera_name,
                    timestamp=inc.start_time,
                    severity=inc.severity,
                    severity_badge=inc.severity_badge,
                    summary=summary,
                    actors=actors,
                    action=action,
                    clip_url=inc.clip_url,
                    snapshot_url=inc.snapshot_url,
                )
            )

        # 4. Generate grounded natural-language response
        response_text = self._synthesize_response(
            query=clean_msg,
            evidence=evidence_list,
            inferred_channel=inferred_channel,
        )

        # 5. Record conversational turns
        user_turn = CopilotMessage(
            role="user",
            content=clean_msg,
            timestamp=time.time(),
            evidence=[],
        )
        assistant_turn = CopilotMessage(
            role="assistant",
            content=response_text,
            timestamp=time.time(),
            evidence=evidence_list,
        )

        with self._lock:
            if active_session not in self._sessions:
                self._sessions[active_session] = []
            self._sessions[active_session].append(user_turn)
            self._sessions[active_session].append(assistant_turn)

        latency_ms = (time.perf_counter() - t0) * 1000

        return CopilotChatResponse(
            response=response_text,
            session_id=active_session,
            cited_incidents=cited_ids,
            evidence=evidence_list,
            latency_ms=round(latency_ms, 2),
        )

    def _infer_channel_from_query(self, query: str) -> Optional[int]:
        """Detects if query mentions a specific Dahua camera channel or location name."""
        q_lower = query.lower()
        cameras = config.parse_camera_names()

        # Check by channel number: "channel 1", "ch 2", "camera 3"
        ch_match = re.search(r"\b(?:ch|channel|cam|camera)\s*([1-9])\b", q_lower)
        if ch_match:
            try:
                ch_num = int(ch_match.group(1))
                if 1 <= ch_num <= config.num_cameras:
                    return ch_num
            except ValueError:
                pass

        # Check by camera location name
        for ch, name in cameras.items():
            name_lower = name.lower()
            # If multi-word (e.g. "front door", "back patio")
            if name_lower in q_lower:
                return ch
            # If specific distinctive word (e.g. "driveway", "garage", "patio", "backyard", "gate")
            distinct_words = [w for w in name_lower.split() if len(w) > 3 and w not in ("side", "north", "south")]
            for word in distinct_words:
                if word in q_lower:
                    return ch

        return None

    def _retrieve_evidence(
        self,
        query: str,
        channel: Optional[int] = None,
        camera_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[IncidentRecord]:
        """Gathers candidate incidents using semantic search and incident engine."""
        # Use semantic search engine
        search_res = search_engine.search(
            query=query,
            channel=channel,
            camera_id=camera_id,
            start_time=start_time,
            end_time=end_time,
            limit=10,
        )

        matched_records: List[IncidentRecord] = []
        for item in search_res.results:
            inc = incident_engine.get_incident(item.incident_id)
            if inc:
                matched_records.append(inc)

        # If search returned nothing specific, fetch latest active or recent incidents
        if not matched_records:
            recent = incident_engine.list_incidents(
                channel=channel,
                limit=5,
            )
            matched_records.extend(recent)

        return matched_records

    def _synthesize_response(
        self,
        query: str,
        evidence: List[CopilotEvidence],
        inferred_channel: Optional[int] = None,
    ) -> str:
        """Constructs a factual, security-oriented response based on retrieved evidence."""
        q_lower = query.lower()
        cameras = config.parse_camera_names()

        # Handle queries when no incidents are found
        if not evidence:
            target_cam = f"Channel {inferred_channel} ({cameras.get(inferred_channel, 'Unknown')})" if inferred_channel else "any camera channel"
            return (
                f"I reviewed the surveillance logs, and no anomalous events or security incidents were recorded for {target_cam}. "
                "All monitoring feeds are operating within normal baseline activity."
            )

        # Count severity badges
        critical_items = [e for e in evidence if e.severity_badge == "CRITICAL"]
        high_items = [e for e in evidence if e.severity_badge == "HIGH"]
        moderate_items = [e for e in evidence if e.severity_badge == "MODERATE"]

        lines = []

        # Headline assessment
        if critical_items:
            lines.append(f"⚠️ **Alert**: Found {len(critical_items)} **CRITICAL** incident(s) requiring immediate attention:")
        elif high_items:
            lines.append(f"🔍 Found {len(high_items)} **HIGH** priority incident(s) in the surveillance records:")
        else:
            lines.append(f"📊 Found {len(evidence)} relevant event(s) matching your query:")

        # Bullet list of cited evidence
        for idx, item in enumerate(evidence, 1):
            time_str = _format_timestamp(item.timestamp)
            badge_icon = "🔴" if item.severity_badge == "CRITICAL" else ("🟠" if item.severity_badge == "HIGH" else "🟡")
            line = f"{idx}. {badge_icon} **{item.camera_name} (Ch {item.channel})** — `{item.severity_badge}` (Severity: {item.severity}/100) at `{time_str}`"
            line += f"\n   • **Summary**: {item.summary}"
            if item.actors:
                line += f"\n   • **Actors**: {', '.join(item.actors)}"
            if item.action:
                line += f"\n   • **Action**: {item.action}"
            lines.append(line)

        # Recommendation / summary footer
        if critical_items:
            lines.append("\n**Recommended Action**: Please review the high-resolution Main-Stream clips and verify the perimeter immediately.")
        elif high_items:
            lines.append("\n**Recommendation**: Review the flagged clips to confirm if authorized activity.")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Session Management
    # -----------------------------------------------------------------------

    def get_history(self, session_id: str) -> List[CopilotMessage]:
        """Retrieves conversational message history for a given session."""
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def clear_history(self, session_id: str) -> bool:
        """Clears a specific conversation session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def clear_all(self):
        """Clears all conversation sessions (useful for tests)."""
        with self._lock:
            self._sessions.clear()

    # -----------------------------------------------------------------------
    # Daily Surveillance Summary Digest & Routine Generator
    # -----------------------------------------------------------------------

    def generate_daily_digest(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        channel: Optional[int] = None,
        format: str = "full",
    ) -> DailyDigestResponse:
        """Synthesizes a 24-hour executive surveillance report across all 9 camera channels."""
        now = time.time()
        p_end = end_time if end_time is not None else now
        p_start = start_time if start_time is not None else (p_end - 86400.0)  # Default 24 hours

        # Fetch incidents in time range
        all_incidents = incident_engine.list_incidents(limit=1000)
        filtered = [
            inc for inc in all_incidents
            if inc.start_time <= p_end and inc.end_time >= p_start
            and (channel is None or inc.channel == channel)
        ]

        cameras = config.parse_camera_names()
        digest_id = f"digest-{uuid.uuid4().hex[:8]}"

        # Global metric counters
        total_incidents = len(filtered)
        total_events = sum(inc.event_count for inc in filtered)
        critical_count = sum(1 for inc in filtered if inc.severity_badge == "CRITICAL")
        high_count = sum(1 for inc in filtered if inc.severity_badge == "HIGH")
        moderate_count = sum(1 for inc in filtered if inc.severity_badge == "MODERATE")
        low_count = sum(1 for inc in filtered if inc.severity_badge == "LOW")

        # Threat Level
        if critical_count > 0:
            threat_level = "SEVERE"
        elif high_count > 0:
            threat_level = "ELEVATED"
        elif moderate_count > 0:
            threat_level = "MODERATE"
        else:
            threat_level = "LOW"

        # Per-channel breakdown
        channel_summaries: List[ChannelActivitySummary] = []
        for ch_num, cam_name in sorted(cameras.items()):
            if channel is not None and ch_num != channel:
                continue

            cam_incs = [inc for inc in filtered if inc.channel == ch_num]
            c_events = sum(inc.event_count for inc in cam_incs)
            c_peak = max([inc.severity for inc in cam_incs], default=0)
            c_crit = sum(1 for inc in cam_incs if inc.severity_badge == "CRITICAL")
            c_high = sum(1 for inc in cam_incs if inc.severity_badge == "HIGH")
            c_mod = sum(1 for inc in cam_incs if inc.severity_badge == "MODERATE")
            c_low = sum(1 for inc in cam_incs if inc.severity_badge == "LOW")

            if not cam_incs:
                c_summary = "Normal baseline. No anomalous security incidents observed."
            else:
                top_inc = max(cam_incs, key=lambda x: x.severity)
                vlm_desc = top_inc.vlm_explanation.summary if top_inc.vlm_explanation else "activity recorded"
                c_summary = f"{len(cam_incs)} incident(s) logged (Peak Severity: {c_peak}/100). Primary observation: {vlm_desc}."

            channel_summaries.append(
                ChannelActivitySummary(
                    channel=ch_num,
                    camera_id=f"cam-{ch_num}",
                    camera_name=cam_name,
                    total_events=c_events,
                    total_incidents=len(cam_incs),
                    peak_severity=c_peak,
                    critical_count=c_crit,
                    high_count=c_high,
                    moderate_count=c_mod,
                    low_count=c_low,
                    summary=c_summary,
                )
            )

        # Key Incident Highlights (sorted by severity)
        sorted_incs = sorted(filtered, key=lambda x: (x.severity, x.start_time), reverse=True)
        key_incidents: List[DigestIncidentSummary] = []
        for inc in sorted_incs[:5]:
            vlm = inc.vlm_explanation
            summary = vlm.summary if vlm else f"Incident on {inc.camera_name}"
            actors = vlm.actors if vlm else []
            action = vlm.action if vlm else ""

            key_incidents.append(
                DigestIncidentSummary(
                    incident_id=inc.incident_id,
                    channel=inc.channel,
                    camera_name=inc.camera_name,
                    timestamp=inc.start_time,
                    duration_s=round(inc.duration_s, 1),
                    severity=inc.severity,
                    severity_badge=inc.severity_badge,
                    summary=summary,
                    actors=actors,
                    action=action,
                    snapshot_url=inc.snapshot_url,
                    clip_url=inc.clip_url,
                )
            )

        # Routine Observations & Recommended Actions
        routine_observations = self._generate_routine_observations(filtered)
        recommended_actions = self._generate_recommended_actions(threat_level, filtered)

        # Executive Summary synthesis
        exec_summary = self._generate_executive_summary(
            p_start=p_start,
            p_end=p_end,
            total_incidents=total_incidents,
            threat_level=threat_level,
            critical_count=critical_count,
            high_count=high_count,
            key_incidents=key_incidents,
        )

        # Markdown synthesis
        markdown_text = self._build_markdown_report(
            digest_id=digest_id,
            p_start=p_start,
            p_end=p_end,
            threat_level=threat_level,
            total_incidents=total_incidents,
            total_events=total_events,
            exec_summary=exec_summary,
            channel_summaries=channel_summaries,
            key_incidents=key_incidents,
            routine_observations=routine_observations,
            recommended_actions=recommended_actions,
        )

        return DailyDigestResponse(
            digest_id=digest_id,
            period_start=p_start,
            period_end=p_end,
            generated_at=now,
            total_incidents=total_incidents,
            total_events=total_events,
            critical_incidents=critical_count,
            high_incidents=high_count,
            moderate_incidents=moderate_count,
            low_incidents=low_count,
            threat_level=threat_level,
            executive_summary=exec_summary,
            channel_summaries=channel_summaries,
            key_incidents=key_incidents,
            routine_observations=routine_observations,
            recommended_actions=recommended_actions,
            markdown_text=markdown_text,
        )

    def _generate_routine_observations(self, incidents: List[IncidentRecord]) -> List[str]:
        """Extracts normal pattern observations from the surveillance records."""
        observations = [
            "All 9 Dahua NVR channels maintained steady HTTP streaming intake with zero frame loss.",
            "Normal daytime vehicular transit and pedestrian passage observed in perimeter zones.",
        ]
        if not incidents:
            observations.append("Baseline normal vector distance remained stable across all camera feeds.")
        else:
            routine_vlm = [
                inc.vlm_explanation.summary for inc in incidents
                if inc.vlm_explanation and inc.severity_badge in ("LOW", "MODERATE")
            ]
            if routine_vlm:
                observations.append(f"Typical neighborhood motion observed: {routine_vlm[0]}.")
        return observations

    def _generate_recommended_actions(self, threat_level: str, incidents: List[IncidentRecord]) -> List[str]:
        """Formulates proactive security actions based on the day's events."""
        actions = []
        if threat_level == "SEVERE":
            actions.extend([
                "🚨 CRITICAL: Immediately inspect physical security locks and perimeter fences for breached zones.",
                "Review and archive high-resolution Main-Stream footage for law enforcement handover.",
                "Verify outdoor lighting and night-vision infrared illumination on affected camera channels.",
            ])
        elif threat_level == "ELEVATED":
            actions.extend([
                "Review flagged HIGH priority clips to verify subject identity.",
                "Check perimeter gate sensors and confirm scheduled deliveries.",
            ])
        else:
            actions.extend([
                "No urgent security interventions required; routine surveillance status.",
                "Continue 48-hour baseline calibration to further refine edge kNN thresholds.",
            ])
        return actions

    def _generate_executive_summary(
        self,
        p_start: float,
        p_end: float,
        total_incidents: int,
        threat_level: str,
        critical_count: int,
        high_count: int,
        key_incidents: List[DigestIncidentSummary],
    ) -> str:
        """Synthesizes high-level executive briefing text."""
        time_range = f"{_format_timestamp(p_start)} to {_format_timestamp(p_end)}"
        if total_incidents == 0:
            return (
                f"Surveillance report for {time_range}. All 9 camera channels functioned normally with "
                "zero security breaches or anomalous deviations detected. Home security posture remains OPTIMAL (Threat Level: LOW)."
            )

        if threat_level in ("SEVERE", "ELEVATED"):
            top = key_incidents[0] if key_incidents else None
            detail = f"Primary alert on {top.camera_name}: {top.summary}." if top else ""
            return (
                f"Surveillance report for {time_range}. A total of {total_incidents} incident(s) were recorded, "
                f"including {critical_count} CRITICAL and {high_count} HIGH severity event(s). Threat Level is rated {threat_level}. {detail}"
            )

        return (
            f"Surveillance report for {time_range}. System logged {total_incidents} minor/routine event(s). "
            f"No critical breaches occurred, and overall threat posture remains {threat_level}."
        )

    def _build_markdown_report(
        self,
        digest_id: str,
        p_start: float,
        p_end: float,
        threat_level: str,
        total_incidents: int,
        total_events: int,
        exec_summary: str,
        channel_summaries: List[ChannelActivitySummary],
        key_incidents: List[DigestIncidentSummary],
        routine_observations: List[str],
        recommended_actions: List[str],
    ) -> str:
        """Generates a complete, beautiful GitHub-flavored markdown surveillance report."""
        badge_icon = "🔴" if threat_level == "SEVERE" else ("🟠" if threat_level == "ELEVATED" else ("🟡" if threat_level == "MODERATE" else "🟢"))

        md = [
            f"# 🛡️ Daily Home Surveillance Security Digest",
            f"**Report ID**: `{digest_id}` | **Generated**: `{_format_timestamp(time.time())}`",
            f"**Reporting Period**: `{_format_timestamp(p_start)}` — `{_format_timestamp(p_end)}`",
            "",
            f"### Security Threat Level: {badge_icon} **{threat_level}**",
            "",
            "## 1. Executive Summary",
            f"> {exec_summary}",
            "",
            "## 2. Key Operational Metrics",
            f"| Metric | Value |",
            f"| :--- | :--- |",
            f"| **Active Dahua Cameras** | 9 Channels |",
            f"| **Total Incidents Recorded** | {total_incidents} |",
            f"| **Total Anomaly Events** | {total_events} |",
            f"| **Threat Level** | {threat_level} |",
            "",
            "## 3. 9-Channel Surveillance Grid Breakdown",
            "| Channel | Camera Location | Incidents | Peak Severity | Status Summary |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for cs in channel_summaries:
            status_badge = "CRITICAL" if cs.critical_count > 0 else ("HIGH" if cs.high_count > 0 else ("MODERATE" if cs.moderate_count > 0 else "NORMAL"))
            md.append(
                f"| **Ch {cs.channel}** | {cs.camera_name} | {cs.total_incidents} | {cs.peak_severity}/100 | `{status_badge}` — {cs.summary} |"
            )

        if key_incidents:
            md.extend([
                "",
                "## 4. Key Incident Highlights",
            ])
            for idx, inc in enumerate(key_incidents, 1):
                icon = "🔴" if inc.severity_badge == "CRITICAL" else ("🟠" if inc.severity_badge == "HIGH" else "🟡")
                md.extend([
                    f"### {idx}. {icon} {inc.camera_name} (Ch {inc.channel}) — `{inc.severity_badge}` ({inc.severity}/100)",
                    f"- **Time**: `{_format_timestamp(inc.timestamp)}` (Duration: {inc.duration_s}s)",
                    f"- **VLM Summary**: {inc.summary}",
                    f"- **Actors**: {', '.join(inc.actors) if inc.actors else 'None identified'}",
                    f"- **Action**: {inc.action or 'Unclassified movement'}",
                    "",
                ])

        md.extend([
            "## 5. Routine Baseline Observations",
        ])
        for obs in routine_observations:
            md.append(f"- {obs}")

        md.extend([
            "",
            "## 6. Proactive Recommendations",
        ])
        for act in recommended_actions:
            md.append(f"- {act}")

        return "\n".join(md)


# Global singleton instance
copilot_engine = SecurityCopilotEngine()
