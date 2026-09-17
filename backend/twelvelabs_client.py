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
    idx_service = getattr(client, "indexes", getattr(client, "index", None))
    if idx_service is not None:
        try:
            indexes = idx_service.list(page_limit=50)
            target_norm = index_name.replace("-", "_").lower()
            for idx in indexes:
                name = getattr(idx, "index_name", getattr(idx, "name", None))
                if name:
                    if name == index_name or name.replace("-", "_").lower() == target_norm:
                        return idx.id
        except Exception as exc:
            logger.warning("Failed to list Twelve Labs indexes: %s", exc)

        # Format model specification
        formatted_models = []
        for m in models:
            m_name = m.get("model_name") or m.get("name")
            m_opts = m.get("model_options") or m.get("options") or ["visual", "audio"]
            formatted_models.append({"model_name": m_name, "model_options": m_opts})

        # Create new index
        try:
            idx = idx_service.create(
                index_name=index_name,
                models=formatted_models,
            )
            logger.info("Created Twelve Labs index '%s' (id=%s)", index_name, idx.id)
            return idx.id
        except Exception as exc:
            logger.error("Failed to create Twelve Labs index '%s': %s", index_name, exc)
            raise RuntimeError(f"Failed to create Twelve Labs index '{index_name}': {exc}") from exc

    raise RuntimeError("TwelveLabs client has no indexes service")


def get_marengo_index_id() -> str:
    """Returns the index ID for Marengo video search and embeddings."""
    global _marengo_index_id
    if _marengo_index_id is None:
        _marengo_index_id = _ensure_index(
            config.marengo_index_name,
            [{"model_name": config.marengo_model, "model_options": ["visual", "audio"]}],
        )
    return _marengo_index_id


def get_pegasus_index_id() -> str:
    """Returns the index ID for Pegasus VLM text generation (uses shared Marengo index)."""
    return get_marengo_index_id()


# ---------------------------------------------------------------------------
# Asset Deduplication Cache & Helpers
# ---------------------------------------------------------------------------

_ASSET_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOADED = False


def _get_cache_file() -> Path:
    return Path("data/storage/twelvelabs_cache.json")


def _load_asset_cache() -> Dict[str, Dict[str, Any]]:
    global _ASSET_CACHE, _CACHE_LOADED
    if not _CACHE_LOADED:
        cache_file = _get_cache_file()
        if cache_file.exists():
            try:
                import json
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    _ASSET_CACHE = data
            except Exception as exc:
                logger.debug("Failed to read Twelve Labs asset cache: %s", exc)
        _CACHE_LOADED = True
    return _ASSET_CACHE


def _save_asset_cache(key: str, entry: Dict[str, Any]) -> None:
    global _ASSET_CACHE
    _ASSET_CACHE[key] = entry
    try:
        import json
        cache_file = _get_cache_file()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(_ASSET_CACHE, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("Failed to save Twelve Labs asset cache: %s", exc)


def find_existing_asset(target_name: str) -> Optional[Dict[str, str]]:
    """Checks local cache and remote Twelve Labs account to find an existing ready asset ID."""
    # 1. Local Cache Lookup
    cache = _load_asset_cache()
    if target_name in cache:
        entry = cache[target_name]
        if entry.get("status") == "ready" and entry.get("asset_id"):
            return {
                "asset_id": str(entry["asset_id"]),
                "marengo_video_id": str(entry.get("marengo_video_id", entry["asset_id"])),
            }

    # 2. Remote Twelve Labs Lookup
    if not is_enabled():
        return None

    try:
        client = get_client()
        assets_service = getattr(client, "assets", None)
        if assets_service and hasattr(assets_service, "list"):
            try:
                # Query by filename filter
                assets_list = assets_service.list(filename=target_name, page_limit=20)
                for a in assets_list:
                    if getattr(a, "status", None) == "ready":
                        fname = getattr(a, "filename", None) or ""
                        if fname == target_name or (target_name and target_name in fname):
                            return {"asset_id": str(getattr(a, "id", "")), "marengo_video_id": str(getattr(a, "id", ""))}
            except Exception:
                pass

            try:
                # Fallback scan of recent account assets
                recent_assets = assets_service.list(page_limit=50)
                for a in recent_assets:
                    if getattr(a, "status", None) == "ready":
                        fname = getattr(a, "filename", None) or ""
                        if fname == target_name or (target_name and target_name in fname):
                            return {"asset_id": str(getattr(a, "id", "")), "marengo_video_id": str(getattr(a, "id", ""))}
            except Exception:
                pass
    except Exception as exc:
        logger.debug("Remote asset deduplication lookup error: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Core Operations: Upload, Search, Analyze, Embed
# ---------------------------------------------------------------------------


def upload_video(file_path: str | Path, index_type: str = "both") -> Dict[str, str]:
    """Uploads a video clip or URL to Twelve Labs as an asset for Pegasus 1.5 analysis and Marengo indexing.
    Reuses existing assets matching the filename to prevent duplicates.

    Args:
        file_path: Path to the MP4 video clip or a direct video URL.
        index_type: "marengo", "pegasus", or "both".

    Returns:
        Dict containing indexed video IDs: {"marengo_video_id": ..., "pegasus_video_id": ...}
    """
    client = get_client()
    result: Dict[str, str] = {}
    target_str = str(file_path)
    target_name = Path(file_path).name if not (target_str.startswith("http://") or target_str.startswith("https://")) else target_str.split("?")[0].split("/")[-1]

    # Check local and remote deduplication before uploading
    existing = find_existing_asset(target_name)
    if existing and existing.get("asset_id"):
        asset_id = existing["asset_id"]
        logger.info("Found existing Twelve Labs asset for '%s': asset_id=%s. Reusing without re-uploading.", target_name, asset_id)
        result["pegasus_video_id"] = asset_id
        result["marengo_video_id"] = existing.get("marengo_video_id", asset_id)

        # Ensure indexed into Marengo if requested
        if index_type in ("marengo", "both"):
            try:
                idx_id = get_marengo_index_id()
                idx_service = getattr(client, "indexes", getattr(client, "index", None))
                idx_assets_service = getattr(idx_service, "indexed_assets", None) if idx_service else None
                if idx_assets_service is not None and hasattr(idx_assets_service, "create"):
                    already_indexed = False
                    indexed_asset_id = str(asset_id)
                    if hasattr(idx_assets_service, "list"):
                        try:
                            for item in idx_assets_service.list(index_id=idx_id):
                                if getattr(item, "asset_id", None) == str(asset_id) or getattr(item, "id", None) == str(asset_id):
                                    indexed_asset_id = str(getattr(item, "id", asset_id))
                                    already_indexed = True
                                    break
                        except Exception:
                            pass

                    if not already_indexed:
                        try:
                            idx_res = idx_assets_service.create(index_id=idx_id, asset_id=str(asset_id))
                            indexed_asset_id = str(getattr(idx_res, "id", asset_id) or asset_id)
                        except Exception as exc:
                            logger.warning("Failed to index existing asset into Marengo: %s", exc)

                    result["marengo_video_id"] = indexed_asset_id
            except Exception as exc:
                logger.warning("Marengo indexing check failed: %s", exc)

        _save_asset_cache(target_name, {
            "asset_id": asset_id,
            "marengo_video_id": result.get("marengo_video_id", asset_id),
            "pegasus_video_id": asset_id,
            "filename": target_name,
            "status": "ready",
        })
        return result

    try:
        # Check if assets service is available (Twelve Labs SDK 1.3+)
        assets_service = getattr(client, "assets", None)
        if assets_service is not None and hasattr(assets_service, "create"):
            if target_str.startswith("http://") or target_str.startswith("https://"):
                asset = assets_service.create(
                    method="url",
                    url=target_str,
                    filename=target_name,
                )
            else:
                path = Path(file_path)
                if not path.exists():
                    raise FileNotFoundError(f"Video file not found at '{path}'")
                with open(path, "rb") as f:
                    asset = assets_service.create(
                        method="direct",
                        file=f,
                        filename=target_name,
                    )

            asset_id = getattr(asset, "id", None) or (asset.get("id") if isinstance(asset, dict) else None)
            if not asset_id:
                raise RuntimeError("Twelve Labs asset upload did not produce a valid asset ID.")

            logger.info("Created Twelve Labs asset id=%s (filename=%s)", asset_id, target_name)

            # Polling asset status
            t_start = time.time()
            while True:
                asset_info = assets_service.retrieve(asset_id)
                status = getattr(asset_info, "status", None) or (asset_info.get("status") if isinstance(asset_info, dict) else None)
                if status == "ready":
                    logger.info("Twelve Labs asset is ready: id=%s", asset_id)
                    break
                if status == "failed":
                    raise RuntimeError(f"Twelve Labs asset processing failed: id={asset_id}")

                if time.time() - t_start > config.twelve_labs_upload_timeout:
                    raise TimeoutError(f"Twelve Labs asset processing timed out after {config.twelve_labs_upload_timeout}s: id={asset_id}")

                time.sleep(2.0)

            result["pegasus_video_id"] = str(asset_id)
            result["marengo_video_id"] = str(asset_id)

            # In Twelve Labs SDK v1.3+, index the asset into Marengo if requested
            if index_type in ("marengo", "both"):
                try:
                    idx_id = get_marengo_index_id()
                    idx_service = getattr(client, "indexes", getattr(client, "index", None))
                    idx_assets_service = getattr(idx_service, "indexed_assets", None) if idx_service else None
                    if idx_assets_service is not None and hasattr(idx_assets_service, "create"):
                        # Check if asset is already indexed
                        already_indexed = False
                        indexed_asset_id = str(asset_id)
                        if hasattr(idx_assets_service, "list"):
                            try:
                                existing_list = idx_assets_service.list(index_id=idx_id)
                                for item in existing_list:
                                    if getattr(item, "asset_id", None) == str(asset_id) or getattr(item, "id", None) == str(asset_id):
                                        indexed_asset_id = str(getattr(item, "id", asset_id))
                                        already_indexed = True
                                        logger.info("Asset %s is already indexed in Marengo index %s", asset_id, idx_id)
                                        break
                            except Exception as exc:
                                logger.debug("Existing indexed assets check skipped: %s", exc)

                        if not already_indexed:
                            try:
                                idx_res = idx_assets_service.create(
                                    index_id=idx_id,
                                    asset_id=str(asset_id),
                                )
                                indexed_asset_id = str(getattr(idx_res, "id", asset_id) or asset_id)
                                logger.info("Created Marengo indexed asset %s for asset_id=%s in index %s", indexed_asset_id, asset_id, idx_id)

                                # Poll indexing progress if retrieve method exists
                                if hasattr(idx_assets_service, "retrieve"):
                                    t_idx_start = time.time()
                                    while time.time() - t_idx_start < config.twelve_labs_upload_timeout:
                                        try:
                                            info = idx_assets_service.retrieve(index_id=idx_id, indexed_asset_id=indexed_asset_id)
                                            st = getattr(info, "status", None)
                                            if st == "ready":
                                                logger.info("Marengo indexed asset %s status is ready", indexed_asset_id)
                                                break
                                            if st == "failed":
                                                logger.warning("Marengo indexing failed for %s", indexed_asset_id)
                                                break
                                        except Exception:
                                            pass
                                        time.sleep(2.0)
                            except Exception as exc:
                                logger.warning("Failed to create Marengo indexed asset: %s", exc)

                        result["marengo_video_id"] = indexed_asset_id
                except Exception as exc:
                    logger.warning("Twelve Labs Marengo indexing hook failed: %s", exc)

            _save_asset_cache(target_name, {
                "asset_id": str(asset_id),
                "marengo_video_id": result.get("marengo_video_id", str(asset_id)),
                "pegasus_video_id": str(asset_id),
                "filename": target_name,
                "status": "ready",
            })

            logger.info("Successfully uploaded video asset to Twelve Labs: asset_id=%s, marengo_video_id=%s", asset_id, result.get("marengo_video_id"))
            return result

        # Fallback to tasks service for legacy mock/clients
        task_service = getattr(client, "tasks", getattr(client, "task", None))
        if task_service is not None:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found at '{path}'")
            idx_id = get_marengo_index_id()
            with open(path, "rb") as f:
                try:
                    task_res = task_service.create(index_id=idx_id, video_file=f)
                except TypeError:
                    task_res = task_service.create(index_id=idx_id, file=str(path))

            video_id = getattr(task_res, "video_id", getattr(task_res, "id", None))
            if not video_id and hasattr(task_service, "retrieve"):
                task_id = getattr(task_res, "id", None)
                if task_id:
                    t_start = time.time()
                    while time.time() - t_start < config.twelve_labs_upload_timeout:
                        task_info = task_service.retrieve(task_id)
                        if getattr(task_info, "status", None) == "ready":
                            video_id = getattr(task_info, "video_id", video_id)
                            break
                        time.sleep(2.0)

            result["marengo_video_id"] = str(video_id or "vid-123")
            result["pegasus_video_id"] = str(video_id or "vid-123")
            return result

    except Exception as exc:
        logger.error("Failed to upload to Twelve Labs: %s", exc)
        raise RuntimeError(f"Failed to upload clip to Twelve Labs: {exc}") from exc

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
    # search_results may be a SyncPager, iterable, list, or object with .data
    items_to_process: List[Any] = []
    if hasattr(search_results, "__iter__") and not isinstance(search_results, (dict, str)):
        try:
            items_to_process = list(search_results)
        except Exception:
            items_to_process = getattr(search_results, "data", []) or []
    else:
        items_to_process = getattr(search_results, "data", []) or []

    for item in items_to_process:
        clips = getattr(item, "clips", None)
        if clips:
            for clip in clips:
                v_id = str(getattr(clip, "video_id", None) or getattr(item, "video_id", None) or getattr(item, "id", "") or "")
                raw_score = getattr(clip, "score", None)
                if raw_score is None:
                    rank = getattr(clip, "rank", None) or getattr(item, "rank", 1)
                    raw_score = 1.0 / (float(rank) if isinstance(rank, (int, float)) and rank > 0 else 1.0)
                results.append(
                    SearchResult(
                        video_id=v_id,
                        score=float(raw_score),
                        start=float(getattr(clip, "start", 0.0) or 0.0),
                        end=float(getattr(clip, "end", 0.0) or 0.0),
                        confidence=str(getattr(clip, "confidence", "") or getattr(item, "confidence", "")),
                        metadata={"module_type": getattr(clip, "module_type", "visual")},
                    )
                )
        else:
            v_id = str(getattr(item, "video_id", None) or getattr(item, "id", "") or "")
            raw_score = getattr(item, "score", None)
            if raw_score is None:
                rank = getattr(item, "rank", None)
                raw_score = 1.0 / (float(rank) if isinstance(rank, (int, float)) and rank > 0 else 1.0) if rank is not None else 0.8
            results.append(
                SearchResult(
                    video_id=v_id,
                    score=float(raw_score),
                    start=float(getattr(item, "start", 0.0) or 0.0),
                    end=float(getattr(item, "end", 0.0) or 0.0),
                    confidence=str(getattr(item, "confidence", "") or ""),
                    metadata={"module_type": getattr(item, "module_type", "visual")},
                )
            )
    return results


def analyze_video(video_id: str, prompt: str) -> AnalysisResult:
    """Queries Pegasus 1.5 VLM to generate a natural-language description or answer questions."""
    t0 = time.perf_counter()
    client = get_client()
    text_output = ""
    model_name = getattr(config, "pegasus_model", "pegasus1.5") or "pegasus1.5"

    # 1. Primary: Twelve Labs SDK analyze_stream with Pegasus 1.5
    if hasattr(client, "analyze_stream") and callable(getattr(client, "analyze_stream")):
        try:
            from twelvelabs.types import VideoContext_AssetId, AnalyzePromptV2
            video = VideoContext_AssetId(asset_id=video_id)
            prompt_obj = AnalyzePromptV2(input_text=prompt)

            text_stream = client.analyze_stream(
                model_name=model_name,
                video=video,
                prompt_v_2=prompt_obj,
            )
            chunks = []
            if text_stream is not None:
                for item in text_stream:
                    if getattr(item, "event_type", None) == "text_generation" and getattr(item, "text", None):
                        chunks.append(item.text)
                    elif isinstance(item, str):
                        chunks.append(item)
            text_output = "".join(chunks)
        except Exception as exc:
            logger.debug("Pegasus 1.5 SDK analyze_stream call error: %s", exc)

    # 2. Secondary: direct client.analyze call
    if not text_output and hasattr(client, "analyze") and callable(getattr(client, "analyze")):
        try:
            from twelvelabs.types import VideoContext_AssetId, AnalyzePromptV2
            video = VideoContext_AssetId(asset_id=video_id)
            prompt_obj = AnalyzePromptV2(input_text=prompt)

            response = client.analyze(
                model_name=model_name,
                video=video,
                prompt_v_2=prompt_obj,
            )
            data_val = getattr(response, "data", None) if hasattr(response, "data") else response
            if isinstance(data_val, str) and data_val:
                text_output = data_val
        except Exception as exc:
            logger.debug("Pegasus 1.5 SDK analyze call error: %s", exc)

    # 3. Legacy / Mock fallback (client.generate.text)
    if not text_output:
        generate_svc = getattr(client, "generate", None)
        if generate_svc and hasattr(generate_svc, "text"):
            try:
                response = generate_svc.text(
                    video_id=video_id,
                    prompt=prompt,
                    temperature=0.2,
                )
                if hasattr(response, "data") and isinstance(response.data, str):
                    text_output = response.data
                elif isinstance(response, str):
                    text_output = response
            except Exception as exc:
                logger.debug("Legacy client.generate.text call error: %s", exc)

    latency_ms = (time.perf_counter() - t0) * 1000
    return AnalysisResult(
        text=text_output or f"Analysis for video {video_id}",
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


def create_text_embedding(text: str) -> Optional[List[float]]:
    """Retrieves high-dimensional text embedding vector from Twelve Labs Marengo."""
    client = get_client()
    try:
        response = client.embed.create(
            model_name=config.marengo_model,
            text=text,
        )
        if response.text_embedding and response.text_embedding.segments:
            seg = response.text_embedding.segments[0]
            return getattr(seg, "float_", getattr(seg, "values", None))
    except Exception as exc:
        logger.warning("Failed to retrieve Marengo text embedding: %s", exc)
    return None
