"""Multi-Modal Semantic Video Search Engine combining Twelve Labs Marengo & Incident Metadata."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set

from .config import config
from .incidents import incident_engine
from .models import (
    IncidentRecord,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchResultItem,
)
from . import twelvelabs_client

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> Set[str]:
    """Extracts lowercase alpha-numeric tokens from a string."""
    return set(re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower()))


def _compute_text_similarity(query_tokens: Set[str], incident: IncidentRecord) -> tuple[float, str, List[str]]:
    """Computes lexical relevance score between query tokens and incident metadata."""
    if not query_tokens:
        return 0.1, "baseline_match", []

    matched_elements: List[str] = []
    score = 0.0

    # Camera Name match
    cam_tokens = _tokenize(incident.camera_name)
    cam_overlap = query_tokens.intersection(cam_tokens)
    if cam_overlap:
        score += 0.25 * (len(cam_overlap) / len(query_tokens))
        matched_elements.append(f"camera({incident.camera_name})")

    # Scene ID match
    if incident.scene_id:
        scene_tokens = _tokenize(incident.scene_id)
        scene_overlap = query_tokens.intersection(scene_tokens)
        if scene_overlap:
            score += 0.15 * (len(scene_overlap) / len(query_tokens))
            matched_elements.append(f"scene({incident.scene_id})")

    # VLM Explanation match
    if incident.vlm_explanation is not None:
        vlm = incident.vlm_explanation

        # Summary tokens
        sum_tokens = _tokenize(vlm.summary)
        sum_overlap = query_tokens.intersection(sum_tokens)
        if sum_overlap:
            score += 0.35 * (len(sum_overlap) / len(query_tokens))
            matched_elements.append("vlm_summary")

        # Actors tokens (high weight)
        actor_text = " ".join(vlm.actors)
        actor_tokens = _tokenize(actor_text)
        actor_overlap = query_tokens.intersection(actor_tokens)
        if actor_overlap:
            score += 0.40 * (len(actor_overlap) / max(1, len(query_tokens)))
            matched_elements.append(f"actors({', '.join(actor_overlap)})")

        # Action tokens
        act_tokens = _tokenize(vlm.action)
        act_overlap = query_tokens.intersection(act_tokens)
        if act_overlap:
            score += 0.30 * (len(act_overlap) / max(1, len(query_tokens)))
            matched_elements.append(f"action({vlm.action})")

        # Objects tokens
        obj_text = " ".join(vlm.objects)
        obj_tokens = _tokenize(obj_text)
        obj_overlap = query_tokens.intersection(obj_tokens)
        if obj_overlap:
            score += 0.20 * (len(obj_overlap) / max(1, len(query_tokens)))
            matched_elements.append(f"objects({', '.join(obj_overlap)})")

    # Operator notes match
    if incident.notes:
        notes_tokens = _tokenize(incident.notes)
        notes_overlap = query_tokens.intersection(notes_tokens)
        if notes_overlap:
            score += 0.20 * (len(notes_overlap) / max(1, len(query_tokens)))
            matched_elements.append("operator_notes")

    # Add small severity bonus (0.00 to 0.10) to favor confirmed high-threat events
    score += (incident.severity / 1000.0)

    # Normalize score to 0.0 - 1.0 range
    final_score = min(1.0, max(0.0, score))
    reason = ", ".join(matched_elements) if matched_elements else "no_keyword_overlap"
    return final_score, reason, matched_elements


class SemanticSearchEngine:
    """Multi-modal search engine uniting Twelve Labs Marengo 3.0 and metadata indexing."""

    def __init__(self):
        pass

    def search(
        self,
        query: str,
        channel: Optional[int] = None,
        camera_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        min_severity: Optional[int] = None,
        severity_badge: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        threshold: str = "medium",
    ) -> SemanticSearchResponse:
        """Searches surveillance footage and incidents using semantic text and filters."""
        t0 = time.perf_counter()
        query_clean = query.strip()
        query_tokens = _tokenize(query_clean)

        # 1. Fetch all candidate incidents
        all_incidents = incident_engine.list_incidents(limit=1000)

        # 2. Try Twelve Labs Marengo search if enabled
        marengo_matches: Dict[str, float] = {}
        if twelvelabs_client.is_enabled() and query_clean:
            try:
                marengo_results = twelvelabs_client.search_videos(
                    query=query_clean,
                    max_clips=max(50, limit * 2),
                    threshold=threshold,
                )
                for mr in marengo_results:
                    marengo_matches[mr.video_id] = mr.score
            except Exception as exc:
                logger.warning("Twelve Labs Marengo search error (falling back to lexical): %s", exc)

        # 3. Score and Filter candidates
        scored_candidates: List[tuple[IncidentRecord, float, str, str]] = []

        filters_applied = {
            "query": query_clean,
            "channel": channel,
            "camera_id": camera_id,
            "start_time": start_time,
            "end_time": end_time,
            "min_severity": min_severity,
            "severity_badge": severity_badge,
            "status": status,
            "threshold": threshold,
        }
        filters_applied = {k: v for k, v in filters_applied.items() if v is not None}

        for inc in all_incidents:
            # Apply Metadata Filters
            if channel is not None and inc.channel != channel:
                continue
            if camera_id is not None and inc.camera_id != camera_id:
                continue
            if start_time is not None and inc.end_time < start_time:
                continue
            if end_time is not None and inc.start_time > end_time:
                continue
            if min_severity is not None and inc.severity < min_severity:
                continue
            if severity_badge is not None and inc.severity_badge.upper() != severity_badge.upper():
                continue
            if status is not None and inc.status.upper() != status.upper():
                continue

            # Compute Relevance
            lex_score, reason, matched_items = _compute_text_similarity(query_tokens, inc)

            # Check if incident is linked to a Marengo video match
            marengo_score = 0.0
            for vid_id, vscore in marengo_matches.items():
                if not vid_id:
                    continue
                clip_url_str = inc.clip_url or ""
                notes_str = inc.notes or ""
                vid_matched = (
                    vid_id in clip_url_str
                    or vid_id in notes_str
                    or inc.incident_id in vid_id
                    or (inc.scene_id and inc.scene_id in vid_id)
                )
                if not vid_matched and inc.events:
                    for ev in inc.events:
                        if ev.clip_url and vid_id in ev.clip_url:
                            vid_matched = True
                            break
                if vid_matched:
                    marengo_score = max(marengo_score, vscore)

            if marengo_score > 0:
                # Weighted fusion of Marengo visual search + lexical metadata
                final_score = (0.65 * marengo_score) + (0.35 * lex_score)
                match_reason = f"marengo_visual({round(marengo_score, 2)}) + {reason}"
                confidence = "high" if final_score >= 0.70 else "medium"
            else:
                final_score = lex_score
                match_reason = f"metadata_match({reason})"
                confidence = "high" if final_score >= 0.60 else ("medium" if final_score >= 0.30 else "low")

            # If a query is provided, skip matches that have 0 keyword overlap and no visual match
            if query_clean and final_score <= 0.05 and not matched_items and marengo_score == 0:
                continue

            scored_candidates.append((inc, final_score, match_reason, confidence))

        # 4. Sort by relevance descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        total_matches = len(scored_candidates)
        paginated = scored_candidates[offset : offset + limit]

        # 5. Format results
        result_items: List[SemanticSearchResultItem] = []
        for inc, score, reason, confidence in paginated:
            vlm = inc.vlm_explanation
            summary = vlm.summary if vlm else f"Incident on {inc.camera_name} (Severity: {inc.severity})"
            actors = vlm.actors if vlm else []
            action = vlm.action if vlm else ""
            objects = vlm.objects if vlm else []

            result_items.append(
                SemanticSearchResultItem(
                    incident_id=inc.incident_id,
                    channel=inc.channel,
                    camera_id=inc.camera_id,
                    camera_name=inc.camera_name,
                    score=round(score, 3),
                    confidence=confidence,
                    timestamp=inc.start_time,
                    duration_s=round(inc.duration_s, 1),
                    severity=inc.severity,
                    severity_badge=inc.severity_badge,
                    status=inc.status,
                    summary=summary,
                    actors=actors,
                    action=action,
                    objects=objects,
                    snapshot_url=inc.snapshot_url,
                    clip_url=inc.clip_url,
                    match_reason=reason,
                )
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        return SemanticSearchResponse(
            query=query,
            total_matches=total_matches,
            results=result_items,
            latency_ms=round(latency_ms, 2),
            filters_applied=filters_applied,
        )


# Global singleton instance
search_engine = SemanticSearchEngine()
