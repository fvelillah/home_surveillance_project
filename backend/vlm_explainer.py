"""Natural Language VLM Scene Explainer using Twelve Labs Pegasus."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from .config import config
from .models import IncidentRecord, VLMExplanation
from . import twelvelabs_client

logger = logging.getLogger(__name__)

VLM_ANALYSIS_PROMPT = """Analyze this home surveillance incident video as an AI security analyst.
Output a valid JSON object strictly matching this schema:
{
  "summary": "Concise one-sentence description of the event observed in the video.",
  "actors": ["List of subjects involved, e.g. delivery driver, visitor, intruder, vehicle, pet"],
  "action": "Description of the specific activity occurring, e.g. dropped parcel on porch, approached front door",
  "objects": ["Key items visible, e.g. cardboard box, backpack, vehicle, gate"],
  "risk_assessment": "routine | suspicious | hazard | breach",
  "recommended_action": "Clear recommendation for homeowner or security operator"
}
Output only the JSON object without any additional markdown formatting or preamble."""


def _parse_vlm_text_response(raw_text: str) -> Dict[str, Any]:
    """Attempts to extract JSON dictionary from Pegasus LLM output."""
    raw_text = raw_text.strip()
    # Strip markdown json code block fences if present
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw_text = "\n".join(lines).strip()

    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Regex search for JSON block
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Fallback to plain text encapsulation
    return {
        "summary": raw_text[:200],
        "actors": ["unknown subject"],
        "action": "movement observed",
        "objects": [],
        "risk_assessment": "suspicious",
        "recommended_action": "Review video clip manually",
    }


async def explain_incident(
    incident: IncidentRecord,
    video_id: Optional[str] = None,
    clip_path: Optional[str] = None,
) -> VLMExplanation:
    """Generates a structured natural language VLM explanation exclusively using Twelve Labs Pegasus."""
    t0 = time.perf_counter()

    if not twelvelabs_client.is_enabled():
        raise RuntimeError("Twelve Labs Pegasus VLM is not enabled or configured")

    target_video_id = video_id
    if not target_video_id and clip_path:
        try:
            res = twelvelabs_client.upload_video(clip_path, index_type="pegasus")
            target_video_id = res.get("pegasus_video_id")
        except Exception as exc:
            logger.error("Failed to upload clip to Pegasus index: %s", exc)
            raise RuntimeError(f"Failed to upload clip to Pegasus index: {exc}") from exc

    if not target_video_id:
        if clip_path:
            raise RuntimeError(
                f"Cannot perform Pegasus VLM explanation for incident '{incident.incident_id}': "
                f"failed to obtain pegasus_video_id for clip '{clip_path}'"
            )
        raise ValueError(
            f"Cannot perform Pegasus VLM explanation for incident '{incident.incident_id}': "
            "no video_id or clip_path provided"
        )

    try:
        analysis = twelvelabs_client.analyze_video(
            video_id=target_video_id,
            prompt=VLM_ANALYSIS_PROMPT,
        )
    except Exception as exc:
        logger.error("Pegasus VLM analysis failed for video_id '%s': %s", target_video_id, exc)
        raise RuntimeError(f"Twelve Labs Pegasus VLM analysis failed: {exc}") from exc

    parsed = _parse_vlm_text_response(analysis.text)
    latency_ms = (time.perf_counter() - t0) * 1000

    return VLMExplanation(
        summary=parsed.get("summary", f"Incident detected on {incident.camera_name}"),
        actors=parsed.get("actors", ["unidentified subject"]),
        action=parsed.get("action", "activity in camera zone"),
        objects=parsed.get("objects", []),
        risk_assessment=parsed.get("risk_assessment", "suspicious"),
        recommended_action=parsed.get("recommended_action", f"Inspect {incident.camera_name}"),
        model=config.pegasus_model,
        latency_ms=latency_ms,
    )
