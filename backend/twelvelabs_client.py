"""Twelve Labs API client for video embedding, semantic search, and VLM analysis.

Wraps the Twelve Labs Python SDK with connection caching, index creation,
and robust error handling.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    video_id: str
    score: float
    start: float
    end: float
    confidence: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    text: str
    video_id: str
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Client Singleton & Helpers
# ---------------------------------------------------------------------------

_client = None
_marengo_index_id: Optional[str] = None
_pegasus_index_id: Optional[str] = None


def is_enabled() -> bool:
    """Returns True if a valid Twelve Labs API key is configured."""
    return bool(config.twelve_labs_api_key)


def get_client():
    """Returns or creates the Twelve Labs client singleton."""
    global _client
    if _client is None:
        if not is_enabled():
            raise RuntimeError("TWELVE_LABS_API_KEY is not configured")
        try:
            from twelvelabs import TwelveLabs
            _client = TwelveLabs(api_key=config.twelve_labs_api_key)
        except ImportError:
            raise RuntimeError("twelvelabs package is not installed. Install via `pip install twelvelabs`")
    return _client


def _ensure_index(index_name: str, models: List[Dict[str, Any]]) -> str:
    """Gets an existing index by name or creates a new one, returning its index ID."""
    client = get_client()
    try:
        indexes = client.index.list()
        for idx in indexes:
            if idx.name == index_name:
                return idx.id
    except Exception as exc:
        logger.warning("Failed to list Twelve Labs indexes: %s", exc)

    # Create new index
    idx = client.index.create(
        name=index_name,
        models=models,
    )
    logger.info("Created Twelve Labs index '%s' (id=%s)", index_name, idx.id)
    return idx.id


def get_marengo_index_id() -> str:
    """Returns the index ID for Marengo video search and embeddings."""
    global _marengo_index_id
    if _marengo_index_id is None:
        _marengo_index_id = _ensure_index(
            config.marengo_index_name,
            [{"name": config.marengo_model, "options": ["visual", "audio"]}],
        )
    return _marengo_index_id


def get_pegasus_index_id() -> str:
    """Returns the index ID for Pegasus VLM text generation."""
    global _pegasus_index_id
    if _pegasus_index_id is None:
        _pegasus_index_id = _ensure_index(
            config.pegasus_index_name,
            [{"name": config.pegasus_model, "options": ["visual", "audio"]}],
        )
    return _pegasus_index_id


# ---------------------------------------------------------------------------
# Core Operations: Upload, Search, Analyze, Embed
# ---------------------------------------------------------------------------


def upload_video(file_path: str | Path, index_type: str = "both") -> Dict[str, str]:
    """Uploads a video clip to Twelve Labs for Marengo / Pegasus indexing.

    Args:
        file_path: Path to the MP4 video clip.
        index_type: "marengo", "pegasus", or "both".

    Returns:
        Dict containing indexed video IDs: {"marengo_video_id": ..., "pegasus_video_id": ...}
    """
    client = get_client()
    result: Dict[str, str] = {}
    path_str = str(file_path)

    if index_type in ("marengo", "both"):
        try:
            idx_id = get_marengo_index_id()
            task = client.task.create(
                index_id=idx_id,
                file=path_str,
            )
            task.wait_for_done(timeout=config.twelve_labs_upload_timeout)
            if task.status == "ready":
                result["marengo_video_id"] = task.video_id
                logger.info("Uploaded to Marengo index: video_id=%s", task.video_id)
            else:
                logger.error("Marengo upload task status: %s", task.status)
        except Exception as exc:
            logger.error("Failed to upload to Marengo: %s", exc)

    if index_type in ("pegasus", "both"):
        try:
            idx_id = get_pegasus_index_id()
            task = client.task.create(
                index_id=idx_id,
                file=path_str,
            )
            task.wait_for_done(timeout=config.twelve_labs_upload_timeout)
            if task.status == "ready":
                result["pegasus_video_id"] = task.video_id
                logger.info("Uploaded to Pegasus index: video_id=%s", task.video_id)
            else:
                logger.error("Pegasus upload task status: %s", task.status)
        except Exception as exc:
            logger.error("Failed to upload to Pegasus: %s", exc)

    return result


def search_videos(
    query: str,
    max_clips: Optional[int] = None,
    threshold: str = "medium",
    group_by: str = "clip",
) -> List[SearchResult]:
    """Executes semantic text-to-video search across indexed footage via Marengo."""
    client = get_client()
    idx_id = get_marengo_index_id()

    search_results = client.search.query(
        index_id=idx_id,
        search_options=["visual", "audio"],
        query_text=query,
        group_by=group_by,
        threshold=threshold,
        page_limit=max_clips or config.twelve_labs_max_clips,
        sort_option="score",
    )

    results: List[SearchResult] = []
    for group in getattr(search_results, "data", []):
        for clip in getattr(group, "clips", []):
            results.append(
                SearchResult(
                    video_id=clip.video_id,
                    score=clip.score,
                    start=clip.start,
                    end=clip.end,
                    confidence=getattr(clip, "confidence", ""),
                    metadata={"module_type": getattr(clip, "module_type", "visual")},
                )
            )
    return results


def analyze_video(video_id: str, prompt: str) -> AnalysisResult:
    """Queries Pegasus VLM to generate a natural-language description or answer questions."""
    client = get_client()
    t0 = time.perf_counter()

    response = client.generate.text(
        video_id=video_id,
        prompt=prompt,
        temperature=0.2,
    )

    latency_ms = (time.perf_counter() - t0) * 1000
    text_output = response.data if hasattr(response, "data") else str(response)

    return AnalysisResult(
        text=text_output,
        video_id=video_id,
        latency_ms=latency_ms,
    )


def get_video_embedding(video_id: str) -> Optional[List[float]]:
    """Retrieves high-dimensional video embedding vector from Twelve Labs Marengo."""
    client = get_client()
    try:
        response = client.embed.create(
            model_name=config.marengo_model,
            video_id=video_id,
        )
        if response.video_embedding and response.video_embedding.values:
            return response.video_embedding.values
    except Exception as exc:
        logger.warning("Failed to retrieve Marengo embedding for video %s: %s", video_id, exc)
    return None
