#!/usr/bin/env python3
"""
scripts/embed_and_index_baseline.py - Phase 7.1 & 7.2: Direct USB Baseline Extraction & Indexing.

Direct in-memory video discovery, stream decoding, PyTorch feature extraction,
MiniBatchKMeans clustering, and dual Qdrant Cloud / Edge batch upserting.
Designed specifically for mounted external USB storage containing Dahua NVR exports
(e.g., /mnt/e/NVR/<backup_date>/ or E:\\NVR\\<backup_date>\\) without copying video
files to the local hard drive.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Ensure project root is in sys.path and remove external ROS paths that shadow project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path = [p for p in sys.path if not p.startswith("/opt/ros")]

from backend.anomaly import index_baseline_vectors
from backend.config import config as central_config
from edge.config import config as edge_config
from edge.detector import QdrantEdgeDetector
from edge.model import EdgeFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("embed_and_index_baseline")


# ---------------------------------------------------------------------------
# 1. Dahua Video Discovery & ASF Filename Parser (Phase 7.1)
# ---------------------------------------------------------------------------

@dataclass
class DahuaVideoFile:
    """Represents a discovered Dahua NVR exported video file."""
    path: Path
    filename: str
    channel: int
    stream_tier: str  # "extra1" (sub-stream), "main", or "unknown"
    start_time_str: Optional[str] = None
    end_time_str: Optional[str] = None
    file_size_mb: float = 0.0

    @property
    def is_substream(self) -> bool:
        return self.stream_tier == "extra1" or "sub" in self.stream_tier.lower()


# Dahua NVR naming patterns:
# 1. Standard: NVR_ch1_extra1_20260919120000_20260919130000.asf
# 2. Short prefix: ch1_extra1_20260919120000_20260919130000.asf
# 3. Extra/Main stream variations: NVR_ch02_main_..., ch03_sub_...
# 4. Fallback: channel in directory or filename: .../ch1/... or cam1_...
DAHUA_PATTERN_PRIMARY = re.compile(
    r"(?:NVR_)?ch(?:annel)?[-_]?0*([1-9]\d*)[-_](extra1|main|sub\d*)[-_](\d+)[-_](\d+)",
    re.IGNORECASE,
)
DAHUA_PATTERN_SECONDARY = re.compile(
    r"(?:NVR_)?ch(?:annel)?[-_]?0*([1-9]\d*)",
    re.IGNORECASE,
)
SUPPORTED_EXTENSIONS = {".asf", ".dav", ".mp4", ".mkv", ".avi", ".mov", ".m4v"}


def parse_dahua_filename(file_path: Path | str) -> Optional[DahuaVideoFile]:
    """
    Parses a Dahua NVR video file path into structured metadata.
    Extracts channel (1-9), stream tier (extra1 vs main), and timestamps.
    """
    path = Path(file_path)
    name = path.name
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return None

    # Try primary pattern (full Dahua schema with timestamps and stream tier)
    match = DAHUA_PATTERN_PRIMARY.search(name)
    if match:
        ch = int(match.group(1))
        tier = match.group(2).lower()
        start_ts = match.group(3)
        end_ts = match.group(4)
        size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
        return DahuaVideoFile(
            path=path,
            filename=name,
            channel=ch,
            stream_tier=tier,
            start_time_str=start_ts,
            end_time_str=end_ts,
            file_size_mb=round(size_mb, 2),
        )

    # Fallback pattern (detect channel from filename or parent directory)
    match_sec = DAHUA_PATTERN_SECONDARY.search(name) or DAHUA_PATTERN_SECONDARY.search(str(path.parent))
    if match_sec:
        ch = int(match_sec.group(1))
        # Infer stream tier
        name_lower = name.lower()
        if "extra1" in name_lower or "sub" in name_lower:
            tier = "extra1"
        elif "main" in name_lower:
            tier = "main"
        else:
            tier = "unknown"

        size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
        return DahuaVideoFile(
            path=path,
            filename=name,
            channel=ch,
            stream_tier=tier,
            file_size_mb=round(size_mb, 2),
        )

    # Default fallback for test videos (e.g. test1.mp4 -> channel 1)
    test_match = re.search(r"test0*([1-9]\d*)", name, re.IGNORECASE)
    if test_match:
        ch = int(test_match.group(1))
        size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
        return DahuaVideoFile(
            path=path,
            filename=name,
            channel=ch,
            stream_tier="extra1",
            file_size_mb=round(size_mb, 2),
        )

    return None


def discover_dahua_videos(
    video_dir: Union[Path, str, List[Union[Path, str]]],
    channels: Optional[List[int]] = None,
    stream_tier: Optional[str] = "extra1",
) -> List[DahuaVideoFile]:
    """
    Recursively scans directory or directories for Dahua surveillance video files.
    Supports single directory, comma-separated string, or list of directory paths.
    Optionally filters by channel and stream tier.
    """
    if isinstance(video_dir, str) and "," in video_dir:
        dirs = [Path(d.strip()) for d in video_dir.split(",") if d.strip()]
    elif isinstance(video_dir, list):
        dirs = [Path(d) for d in video_dir]
    else:
        dirs = [Path(video_dir)]

    discovered_map: Dict[str, DahuaVideoFile] = {}
    for dir_path in dirs:
        if not dir_path.exists():
            logger.warning("Video directory does not exist: %s", dir_path)
            continue

        for item in dir_path.rglob("*"):
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                meta = parse_dahua_filename(item)
                if meta:
                    # Apply channel filter if requested
                    if channels and meta.channel not in channels:
                        continue
                    # Apply stream tier filter if requested (extra1 is priority for baseline)
                    if stream_tier and stream_tier != "any":
                        if stream_tier == "extra1" and not meta.is_substream:
                            continue
                        elif stream_tier != "extra1" and meta.stream_tier != stream_tier:
                            continue
                    canonical_key = str(meta.path.resolve())
                    discovered_map[canonical_key] = meta

    discovered = list(discovered_map.values())
    # Sort by channel then path
    discovered.sort(key=lambda x: (x.channel, str(x.path)))
    return discovered


# ---------------------------------------------------------------------------
# 2. In-RAM Window Decoding & PyTorch Feature Extraction (Phase 7.2)
# ---------------------------------------------------------------------------

def stream_decode_windows(
    video_path: Path | str,
    window_duration_s: float = 10.0,
    sample_frames: int = 16,
    target_width: int = 224,
    target_height: int = 224,
    max_windows: Optional[int] = None,
    window_stride_s: Optional[float] = None,
) -> Generator[Tuple[List[np.ndarray], float, float], None, None]:
    """
    Stream-decodes a video directly into RAM window-by-window using OpenCV.
    Never copies video files to disk; processes frame chunks in memory.
    Uses selective grab for non-sampled frames to maximize throughput.

    Yields:
        (frames_rgb, start_time_sec, end_time_sec)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Failed to open video stream: %s", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0.0 or np.isnan(fps):
        fps = 25.0  # Default Dahua Sub-Stream framerate

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_window = int(round(window_duration_s * fps))
    if frames_per_window < 1:
        frames_per_window = 1

    stride_s = window_stride_s if window_stride_s is not None else window_duration_s
    stride_frames = max(frames_per_window, int(round(stride_s * fps)))

    sample_indices = set(np.linspace(0, frames_per_window - 1, min(sample_frames, frames_per_window), dtype=int))
    current_frame_idx = 0
    window_count = 0

    while True:
        if max_windows and window_count >= max_windows:
            break

        window_frames: List[np.ndarray] = []
        frames_read = 0
        for f_idx in range(frames_per_window):
            if f_idx in sample_indices:
                ret, frame = cap.read()
                if not ret:
                    break
                resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                window_frames.append(rgb)
            else:
                ret = cap.grab()
                if not ret:
                    break
            frames_read += 1

        if not window_frames:
            break

        # Replicate last frame if fewer than sample_frames
        while len(window_frames) < sample_frames:
            window_frames.append(window_frames[-1])

        start_sec = current_frame_idx / fps
        end_sec = (current_frame_idx + frames_read) / fps
        current_frame_idx += frames_read
        window_count += 1

        yield window_frames, start_sec, end_sec

        if frames_read < frames_per_window:
            # End of video reached
            break

        # If stride is larger than window, grab excess frames
        excess = stride_frames - frames_per_window
        if excess > 0:
            for _ in range(excess):
                if not cap.grab():
                    break
            current_frame_idx += excess

    cap.release()


def extract_embeddings_for_channel(
    videos: List[DahuaVideoFile],
    extractor: EdgeFeatureExtractor,
    window_duration_s: float = 10.0,
    sample_frames: int = 16,
    max_windows_per_video: Optional[int] = None,
    window_stride_s: Optional[float] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Extracts PyTorch embeddings for all discovered video files of a single channel.
    Returns array of unit-normalized embeddings of shape (N, embedding_dim) and metadata.
    """
    embeddings_list: List[np.ndarray] = []
    metadata_list: List[Dict[str, Any]] = []

    for vid in videos:
        logger.info("  Decoding '%s' (%.2f MB)...", vid.filename, vid.file_size_mb)
        win_idx = 0
        for frames_rgb, start_s, end_s in stream_decode_windows(
            video_path=vid.path,
            window_duration_s=window_duration_s,
            sample_frames=sample_frames,
            max_windows=max_windows_per_video,
            window_stride_s=window_stride_s,
        ):
            # Extract unit L2-normalized embedding
            emb = extractor.extract(frames_rgb)
            embeddings_list.append(emb)

            meta = {
                "source_video": vid.filename,
                "channel": vid.channel,
                "camera_id": f"cam-{vid.channel}",
                "stream_tier": vid.stream_tier,
                "window_idx": win_idx,
                "start_time_s": round(start_s, 2),
                "end_time_s": round(end_s, 2),
                "timestamp_ms": int(time.time() * 1000),
            }
            metadata_list.append(meta)
            win_idx += 1

    if not embeddings_list:
        return np.empty((0, extractor.embedding_dim), dtype=np.float32), []

    embeddings_arr = np.stack(embeddings_list, axis=0)
    return embeddings_arr, metadata_list


# ---------------------------------------------------------------------------
# 3. MiniBatchKMeans Clustering (Phase 7.2)
# ---------------------------------------------------------------------------

def cluster_channel_embeddings(
    embeddings: np.ndarray,
    metadata_list: List[Dict[str, Any]],
    n_clusters: int = 500,
    batch_size: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Condenses dense surveillance routine vectors into representative centroids using MiniBatchKMeans.
    Normalizes fitted centroids to unit L2 norm for cosine distance calculation in Qdrant.

    Returns:
        (normalized_centroids, centroid_metadata_list)
    """
    num_samples, dim = embeddings.shape
    if num_samples == 0:
        return np.empty((0, dim), dtype=np.float32), []

    # If fewer samples than requested clusters, retain all samples as centroids
    if num_samples <= n_clusters:
        logger.info("  Sample count (%d) <= target clusters (%d). Retaining all samples.", num_samples, n_clusters)
        centroids = embeddings.copy()
        centroid_metas = []
        for i, meta in enumerate(metadata_list):
            c_meta = dict(meta)
            c_meta["is_centroid"] = True
            c_meta["cluster_id"] = i
            c_meta["samples_in_cluster"] = 1
            centroid_metas.append(c_meta)
        return centroids, centroid_metas

    logger.info("  Fitting MiniBatchKMeans: %d vectors -> %d centroids (batch_size=%d)...", num_samples, n_clusters, batch_size)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(batch_size, num_samples),
        random_state=random_state,
        n_init="auto",
    )
    labels = kmeans.fit_predict(embeddings)
    raw_centroids = kmeans.cluster_centers_.astype(np.float32)

    # L2-normalize centroids to lie on the unit hypersphere
    norms = np.linalg.norm(raw_centroids, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    normalized_centroids = raw_centroids / norms

    # Build representative metadata for each centroid
    cluster_counts = np.bincount(labels, minlength=n_clusters)
    centroid_metas: List[Dict[str, Any]] = []

    ch = metadata_list[0].get("channel", 1) if metadata_list else 1
    for c_idx in range(n_clusters):
        meta = {
            "channel": ch,
            "camera_id": f"cam-{ch}",
            "is_centroid": True,
            "cluster_id": c_idx,
            "samples_in_cluster": int(cluster_counts[c_idx]),
            "indexed_at": time.time(),
        }
        centroid_metas.append(meta)

    return normalized_centroids, centroid_metas


# ---------------------------------------------------------------------------
# 4. Dual Batch Upsert Pipeline (Phase 7.2)
# ---------------------------------------------------------------------------

def index_channel_centroids(
    channel: int,
    centroids: np.ndarray,
    metadata_list: List[Dict[str, Any]],
    upload_cloud: bool = True,
    upload_edge: bool = True,
    edge_detector: Optional[QdrantEdgeDetector] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, int]:
    """
    Batch upserts condensed baseline centroids into Central Qdrant Cloud and local Qdrant Edge shard.
    """
    results = {"cloud_points": 0, "edge_points": 0}
    if centroids.shape[0] == 0:
        return results

    # 1. Central Qdrant Upsert
    if upload_cloud:
        coll = collection_name or central_config.collection_name
        vectors = centroids.tolist()
        try:
            cloud_count = index_baseline_vectors(
                vectors=vectors,
                metadata_list=metadata_list,
                collection_name=coll,
            )
            results["cloud_points"] = cloud_count
            logger.info("  ✓ Central Qdrant: %d centroids indexed into '%s'", len(vectors), coll)
        except Exception as exc:
            logger.error("  ✗ Central Qdrant indexing failed: %s", exc)

    # 2. Local Qdrant Edge Upsert
    if upload_edge:
        try:
            detector = edge_detector or QdrantEdgeDetector(
                qdrant_path=edge_config.qdrant_edge_path,
                embedding_dim=centroids.shape[1],
            )
            edge_count = detector.seed_baseline(
                channel=channel,
                vectors=[centroids[i] for i in range(centroids.shape[0])],
                metadata=metadata_list,
            )
            results["edge_points"] = edge_count
            logger.info("  ✓ Qdrant Edge: %d centroids seeded into channel %d baseline", edge_count, channel)
        except Exception as exc:
            logger.error("  ✗ Qdrant Edge indexing failed: %s", exc)

    return results


def clear_vector_databases(
    clear_cloud: bool = True,
    clear_edge: bool = True,
    collection_name: Optional[str] = None,
    qdrant_edge_path: Optional[Path | str] = None,
    embedding_dim: int = 576,
) -> Dict[str, Any]:
    """
    Clears pre-existing baseline vectors from Qdrant Cloud and local Qdrant Edge databases.
    """
    status = {"cloud_cleared": False, "edge_cleared": False}
    coll = collection_name or central_config.collection_name

    # 1. Clear Qdrant Cloud collection
    if clear_cloud and central_config.qdrant_url:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            client = QdrantClient(
                url=central_config.qdrant_url,
                api_key=central_config.qdrant_api_key,
                timeout=15.0,
            )
            colls = [c.name for c in client.get_collections().collections]
            if coll in colls:
                logger.info("Deleting pre-existing Qdrant Cloud collection '%s'...", coll)
                client.delete_collection(collection_name=coll)
            logger.info("Recreating empty Qdrant Cloud collection '%s' (dim=%d, distance=COSINE)...", coll, embedding_dim)
            client.create_collection(
                collection_name=coll,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            )
            status["cloud_cleared"] = True
            print(f"  ✓ Qdrant Cloud collection '{coll}' deleted and recreated cleanly.")
        except Exception as exc:
            logger.error("Failed to clear Qdrant Cloud collection '%s': %s", coll, exc)
            print(f"  ✗ Qdrant Cloud clear failed: {exc}")

    # 2. Clear Qdrant Edge local database
    if clear_edge:
        edge_path = Path(qdrant_edge_path or edge_config.qdrant_edge_path)
        try:
            import shutil
            if edge_path.exists():
                logger.info("Deleting pre-existing Qdrant Edge database at: %s", edge_path)
                shutil.rmtree(edge_path, ignore_errors=True)
            edge_path.mkdir(parents=True, exist_ok=True)
            _ = QdrantEdgeDetector(qdrant_path=edge_path, embedding_dim=embedding_dim)
            status["edge_cleared"] = True
            print(f"  ✓ Local Qdrant Edge database at '{edge_path}' wiped and re-initialized cleanly.")
        except Exception as exc:
            logger.error("Failed to clear Qdrant Edge: %s", exc)
            print(f"  ✗ Qdrant Edge clear failed: {exc}")

    return status


# ---------------------------------------------------------------------------
# 5. CLI Execution & Orchestrator
# ---------------------------------------------------------------------------

def run_baseline_pipeline(
    video_dir: Union[Path, str, List[Union[Path, str]]],
    clusters_per_camera: int = 500,
    channels: Optional[List[int]] = None,
    stream_tier: str = "extra1",
    dry_run: bool = False,
    upload_cloud: bool = True,
    upload_edge: bool = True,
    model_type: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    max_windows_per_video: Optional[int] = None,
    window_stride_s: Optional[float] = None,
    clear_db: bool = False,
) -> Dict[str, Any]:
    """
    Executes end-to-end baseline calibration and indexing across all discovered channels.
    """
    t0 = time.time()
    print("\n" + "=" * 70)
    print("  DAHUA SURVEILLANCE: IN-RAM BASELINE EXTRACTION & QDRANT INDEXING")
    print("=" * 70)
    print(f"  Target Video Directory: {video_dir}")
    print(f"  Stream Tier Filter:     {stream_tier}")
    print(f"  Clusters Per Camera:    {clusters_per_camera}")
    print(f"  Window Stride:          {window_stride_s if window_stride_s is not None else 10.0}s")
    print(f"  Clear Pre-existing DB:  {clear_db and not dry_run}")
    print(f"  Mode:                   {'DRY-RUN (No Writes)' if dry_run else 'LIVE UPSERT'}")
    print(f"  Cloud Ingestion:        {upload_cloud and not dry_run}")
    print(f"  Edge Ingestion:         {upload_edge and not dry_run}")
    print("=" * 70 + "\n")

    # 1. Video Discovery
    videos = discover_dahua_videos(video_dir, channels=channels, stream_tier=stream_tier)
    if not videos:
        # If no Sub-Stream files found with strict filter, fallback to any supported video
        if stream_tier != "any":
            logger.warning("No '%s' files found; scanning for any video files...", stream_tier)
            videos = discover_dahua_videos(video_dir, channels=channels, stream_tier="any")

    if not videos:
        print(f"✗ No Dahua surveillance video files found in '{video_dir}'.")
        return {"status": "no_videos_found", "channels_indexed": 0, "total_centroids": 0}

    # Group discovered videos by channel
    by_channel: Dict[int, List[DahuaVideoFile]] = {}
    for v in videos:
        by_channel.setdefault(v.channel, []).append(v)

    total_size_mb = sum(v.file_size_mb for v in videos)
    print(f"Found {len(videos)} video file(s) across {len(by_channel)} channel(s) ({total_size_mb:.2f} MB total):\n")
    for ch, v_list in sorted(by_channel.items()):
        ch_name = central_config.parse_camera_names().get(ch, f"Channel {ch}")
        ch_size = sum(f.file_size_mb for f in v_list)
        print(f"  • Channel {ch:02d} ({ch_name}): {len(v_list)} file(s), {ch_size:.2f} MB")

    # 2. Initialize PyTorch Feature Extractor
    chosen_model = model_type or edge_config.model_type
    chosen_dim = embedding_dim or edge_config.embedding_dim
    print(f"\nInitializing PyTorch Feature Extractor ({chosen_model}, dim={chosen_dim})...")
    extractor = EdgeFeatureExtractor(model_type=chosen_model, embedding_dim=chosen_dim)

    # Clear databases before uploading if requested
    if clear_db and not dry_run:
        print("\nClearing Pre-Existing Vectors from Vector Databases...")
        clear_vector_databases(
            clear_cloud=upload_cloud,
            clear_edge=upload_edge,
            embedding_dim=chosen_dim,
        )
        print()

    # Initialize Edge Detector if edge ingestion enabled
    edge_detector = None
    if upload_edge and not dry_run:
        edge_detector = QdrantEdgeDetector(
            qdrant_path=edge_config.qdrant_edge_path,
            embedding_dim=chosen_dim,
        )

    summary: Dict[str, Any] = {
        "status": "success",
        "channels_processed": len(by_channel),
        "total_windows": 0,
        "total_centroids": 0,
        "channel_results": {},
        "elapsed_s": 0.0,
    }

    # 3. Process Each Channel Sequentially
    print("\nProcessing Channels (In-RAM Decoding & MiniBatchKMeans)...")
    for ch, ch_videos in sorted(by_channel.items()):
        ch_name = central_config.parse_camera_names().get(ch, f"Channel {ch}")
        print(f"\n--- Channel {ch:02d} ({ch_name}) ---")

        t_ch0 = time.time()
        # Stream-decode & extract embeddings
        embeddings, metadata = extract_embeddings_for_channel(
            videos=ch_videos,
            extractor=extractor,
            window_duration_s=edge_config.clip_duration_s,
            sample_frames=edge_config.segment_sample_frames,
            max_windows_per_video=max_windows_per_video,
            window_stride_s=window_stride_s,
        )
        print(f"  Extracted {embeddings.shape[0]} windows in {time.time() - t_ch0:.2f}s")
        summary["total_windows"] += embeddings.shape[0]

        if embeddings.shape[0] == 0:
            print("  ⚠ No frames decoded from channel videos.")
            continue

        # Condense with MiniBatchKMeans
        t_km0 = time.time()
        centroids, centroid_metas = cluster_channel_embeddings(
            embeddings=embeddings,
            metadata_list=metadata,
            n_clusters=clusters_per_camera,
            batch_size=100,
        )
        print(f"  Fitted {centroids.shape[0]} normal centroids in {time.time() - t_km0:.2f}s")
        summary["total_centroids"] += centroids.shape[0]

        # Upsert if not dry-run
        upsert_res = {"cloud_points": 0, "edge_points": 0}
        if not dry_run:
            upsert_res = index_channel_centroids(
                channel=ch,
                centroids=centroids,
                metadata_list=centroid_metas,
                upload_cloud=upload_cloud,
                upload_edge=upload_edge,
                edge_detector=edge_detector,
            )

        summary["channel_results"][ch] = {
            "windows": embeddings.shape[0],
            "centroids": centroids.shape[0],
            "cloud_points": upsert_res["cloud_points"],
            "edge_points": upsert_res["edge_points"],
        }

    summary["elapsed_s"] = round(time.time() - t0, 2)

    # 4. Print Summary Matrix
    print("\n" + "=" * 70)
    print("  BASELINE EXTRACTION SUMMARY MATRIX")
    print("=" * 70)
    print(f"  Channels Processed: {summary['channels_processed']}")
    print(f"  Total Windows:      {summary['total_windows']}")
    print(f"  Total Centroids:    {summary['total_centroids']}")
    print(f"  Total Elapsed Time: {summary['elapsed_s']:.2f}s")
    print("-" * 70)
    for ch, res in summary["channel_results"].items():
        ch_name = central_config.parse_camera_names().get(ch, f"Cam {ch}")
        print(f"  CH {ch:02d} ({ch_name:<16}): {res['windows']:>4} wins -> {res['centroids']:>4} centroids | Cloud: {res['cloud_points']} | Edge: {res['edge_points']}")
    print("=" * 70 + "\n")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 7.1/7.2: Direct USB Video Baseline Extraction & Qdrant Indexing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="/mnt/e/NVR",
        help="Path to mounted USB storage or directory containing Dahua NVR video files (can be comma-separated)",
    )
    parser.add_argument(
        "--clusters-per-camera",
        type=int,
        default=500,
        help="Target number of MiniBatchKMeans centroid clusters per camera channel",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Comma-separated channel filter (e.g. '1,2,3')",
    )
    parser.add_argument(
        "--stream-tier",
        type=str,
        default="extra1",
        choices=["extra1", "main", "any"],
        help="Filter stream tier ('extra1' for Sub-Stream, 'main', or 'any')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform in-RAM decoding and clustering without writing to Qdrant databases",
    )
    parser.add_argument(
        "--skip-cloud",
        action="store_true",
        help="Skip upserting centroids to Central Qdrant Cloud",
    )
    parser.add_argument(
        "--skip-edge",
        action="store_true",
        help="Skip seeding centroids to local Qdrant Edge shard",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Maximum windows to decode per video (useful for testing/benchmarking)",
    )
    parser.add_argument(
        "--window-stride-s",
        type=float,
        default=20.0,
        help="Stride between consecutive 10s evaluation windows in seconds",
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Wipe all pre-existing baseline vectors from Qdrant Cloud and Qdrant Edge before uploading",
    )

    args = parser.parse_args()

    channels_filter = None
    if args.channels:
        try:
            channels_filter = [int(c.strip()) for c in args.channels.split(",") if c.strip()]
        except ValueError:
            logger.error("Invalid --channels format. Expected integers, e.g. '1,2,3'")
            return 1

    # Check if video_dir exists; if default does not exist, check local data/tests/
    target_path: Union[str, Path] = args.video_dir
    if isinstance(target_path, str) and "," not in target_path:
        p = Path(target_path)
        if not p.exists() and target_path == "/mnt/e/NVR":
            fallback_path = PROJECT_ROOT / "data" / "tests"
            if fallback_path.exists():
                logger.info("Default USB mount '/mnt/e/NVR' not found. Falling back to '%s'", fallback_path)
                target_path = fallback_path

    res = run_baseline_pipeline(
        video_dir=target_path,
        clusters_per_camera=args.clusters_per_camera,
        channels=channels_filter,
        stream_tier=args.stream_tier,
        dry_run=args.dry_run,
        upload_cloud=not args.skip_cloud,
        upload_edge=not args.skip_edge,
        max_windows_per_video=args.max_windows,
        window_stride_s=args.window_stride_s,
        clear_db=args.clear_db,
    )

    return 0 if res.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
