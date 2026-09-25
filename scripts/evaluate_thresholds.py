#!/usr/bin/env python3
"""
scripts/evaluate_thresholds.py - Phase 7.3: Empirical Threshold Calibration & ROC Evaluation.

Computes empirical kNN cosine distance distributions (50th, 90th, 95th, 99th, and max percentiles)
on normal surveillance footage to systematically tune EDGE_TRIAGE_THRESHOLD (~0.060) and
CLOUD_ANOMALY_THRESHOLD (~0.150).
Optionally evaluates discrimination against known anomalous clips or synthetic perturbations
to produce ROC-AUC, precision-recall metrics, and automated .env threshold updates.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path and remove external ROS paths that shadow project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path = [p for p in sys.path if not p.startswith("/opt/ros")]

from backend.anomaly import score_clip
from backend.config import config as central_config
from edge.config import config as edge_config
from edge.detector import QdrantEdgeDetector
from edge.model import EdgeFeatureExtractor
from scripts.embed_and_index_baseline import discover_dahua_videos, stream_decode_windows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_thresholds")


@dataclass
class DistributionMetrics:
    """Statistical summary of empirical kNN cosine distances."""
    sample_count: int
    mean: float
    std: float
    min: float
    p50: float  # Median
    p90: float
    p95: float
    p99: float
    max: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ROCMetrics:
    """ROC curve and discrimination performance metrics."""
    auc_roc: float
    best_f1: float
    best_threshold: float
    tpr_at_best_f1: float
    fpr_at_best_f1: float
    precision_at_best_f1: float


def compute_distribution(scores: List[float]) -> DistributionMetrics:
    """Calculates distribution statistics and percentiles for a list of cosine distances."""
    if not scores:
        return DistributionMetrics(
            sample_count=0,
            mean=0.0,
            std=0.0,
            min=0.0,
            p50=0.0,
            p90=0.0,
            p95=0.0,
            p99=0.0,
            max=0.0,
        )

    arr = np.array(scores, dtype=np.float32)
    return DistributionMetrics(
        sample_count=len(arr),
        mean=float(round(float(np.mean(arr)), 4)),
        std=float(round(float(np.std(arr)), 4)),
        min=float(round(float(np.min(arr)), 4)),
        p50=float(round(float(np.percentile(arr, 50)), 4)),
        p90=float(round(float(np.percentile(arr, 90)), 4)),
        p95=float(round(float(np.percentile(arr, 95)), 4)),
        p99=float(round(float(np.percentile(arr, 99)), 4)),
        max=float(round(float(np.max(arr)), 4)),
    )


def compute_roc_metrics(
    normal_scores: List[float],
    anomaly_scores: List[float],
    steps: int = 200,
) -> ROCMetrics:
    """
    Computes empirical ROC curve and F1 metrics across a sweep of distance thresholds.
    """
    if not normal_scores or not anomaly_scores:
        return ROCMetrics(
            auc_roc=1.0,
            best_f1=1.0,
            best_threshold=0.10,
            tpr_at_best_f1=1.0,
            fpr_at_best_f1=0.0,
            precision_at_best_f1=1.0,
        )

    min_val = min(min(normal_scores), min(anomaly_scores))
    max_val = max(max(normal_scores), max(anomaly_scores))
    thresholds = np.linspace(max(0.0, min_val - 0.05), max_val + 0.05, steps)

    n_normal = len(normal_scores)
    n_anomaly = len(anomaly_scores)

    norm_arr = np.array(normal_scores)
    anom_arr = np.array(anomaly_scores)

    tprs: List[float] = []
    fprs: List[float] = []
    f1s: List[float] = []
    precisions: List[float] = []

    best_f1 = -1.0
    best_th = 0.10
    best_tpr = 0.0
    best_fpr = 0.0
    best_prec = 0.0

    for th in thresholds:
        tp = np.sum(anom_arr >= th)
        fp = np.sum(norm_arr >= th)
        fn = np.sum(anom_arr < th)
        tn = np.sum(norm_arr < th)

        tpr = float(tp / n_anomaly) if n_anomaly > 0 else 0.0
        fpr = float(fp / n_normal) if n_normal > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        f1 = float(2 * (prec * tpr) / (prec + tpr)) if (prec + tpr) > 0 else 0.0

        tprs.append(tpr)
        fprs.append(fpr)
        f1s.append(f1)
        precisions.append(prec)

        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
            best_tpr = tpr
            best_fpr = fpr
            best_prec = prec

    # Approximate AUC-ROC via trapezoidal rule
    # Sort points by increasing FPR
    sorted_pairs = sorted(zip(fprs, tprs), key=lambda x: x[0])
    s_fprs = [p[0] for p in sorted_pairs]
    s_tprs = [p[1] for p in sorted_pairs]
    auc = float(np.abs(np.trapezoid(s_tprs, s_fprs))) if hasattr(np, "trapezoid") else float(np.abs(np.trapz(s_tprs, s_fprs)))
    auc = min(1.0, max(0.0, auc))

    return ROCMetrics(
        auc_roc=round(auc, 4),
        best_f1=round(best_f1, 4),
        best_threshold=round(best_th, 4),
        tpr_at_best_f1=round(best_tpr, 4),
        fpr_at_best_f1=round(best_fpr, 4),
        precision_at_best_f1=round(best_prec, 4),
    )


def extract_clip_embeddings(
    video_dir: Path | str,
    extractor: EdgeFeatureExtractor,
    channels: Optional[List[int]] = None,
    stream_tier: str = "any",
    window_duration_s: float = 10.0,
    sample_frames: int = 16,
    max_windows_per_video: Optional[int] = None,
) -> Dict[int, List[Tuple[np.ndarray, str, float]]]:
    """
    Extracts embeddings for discovered video files, grouped by camera channel.
    Returns: {channel: [(embedding, filename, start_sec), ...]}
    """
    v_dir = Path(video_dir)
    videos = discover_dahua_videos(v_dir, channels=channels, stream_tier=stream_tier)
    if not videos and stream_tier != "any":
        videos = discover_dahua_videos(v_dir, channels=channels, stream_tier="any")

    by_channel: Dict[int, List[Tuple[np.ndarray, str, float]]] = {}
    for vid in videos:
        logger.info("  Scanning clip: %s (ch=%d)...", vid.filename, vid.channel)
        for frames_rgb, start_s, _ in stream_decode_windows(
            video_path=vid.path,
            window_duration_s=window_duration_s,
            sample_frames=sample_frames,
            max_windows=max_windows_per_video,
        ):
            emb = extractor.extract(frames_rgb)
            by_channel.setdefault(vid.channel, []).append((emb, vid.filename, start_s))

    return by_channel


def score_embeddings_against_baseline(
    channel_embeddings: Dict[int, List[Tuple[np.ndarray, str, float]]],
    edge_detector: Optional[QdrantEdgeDetector] = None,
    k: int = 5,
) -> Dict[int, List[float]]:
    """
    Scores embeddings against the normal baseline using Qdrant Edge or Central Qdrant.
    Returns: {channel: [anomaly_score_1, anomaly_score_2, ...]}
    """
    detector = edge_detector or QdrantEdgeDetector(
        qdrant_path=edge_config.qdrant_edge_path,
        embedding_dim=edge_config.embedding_dim,
    )

    scores_by_channel: Dict[int, List[float]] = {}
    for ch, items in channel_embeddings.items():
        ch_scores: List[float] = []
        for i, (emb, _, _) in enumerate(items):
            clip_id = f"eval-ch{ch}-{i}-{uuid.uuid4().hex[:6]}"
            score, details = detector.score_clip(
                channel=ch,
                embedding=emb,
                clip_id=clip_id,
                timestamp=time.time(),
            )
            dist = float(details.get("d_baseline", score))
            ch_scores.append(dist)
        scores_by_channel[ch] = ch_scores

    return scores_by_channel


def update_env_thresholds(
    env_file_path: Path | str,
    edge_triage_th: float,
    cloud_anomaly_th: float,
) -> bool:
    """
    Updates EDGE_TRIAGE_THRESHOLD and CLOUD_ANOMALY_THRESHOLD in the target .env file.
    """
    p = Path(env_file_path)
    if not p.exists():
        logger.error(".env file not found at: %s", p)
        return False

    content = p.read_text(encoding="utf-8")

    # Replace or append EDGE_TRIAGE_THRESHOLD
    if re.search(r"^EDGE_TRIAGE_THRESHOLD\s*=", content, re.MULTILINE):
        content = re.sub(
            r"^EDGE_TRIAGE_THRESHOLD\s*=.*$",
            f"EDGE_TRIAGE_THRESHOLD={edge_triage_th:.3f}",
            content,
            flags=re.MULTILINE,
        )
    else:
        content += f"\nEDGE_TRIAGE_THRESHOLD={edge_triage_th:.3f}\n"

    # Replace or append CLOUD_ANOMALY_THRESHOLD
    if re.search(r"^CLOUD_ANOMALY_THRESHOLD\s*=", content, re.MULTILINE):
        content = re.sub(
            r"^CLOUD_ANOMALY_THRESHOLD\s*=.*$",
            f"CLOUD_ANOMALY_THRESHOLD={cloud_anomaly_th:.3f}",
            content,
            flags=re.MULTILINE,
        )
    else:
        content += f"\nCLOUD_ANOMALY_THRESHOLD={cloud_anomaly_th:.3f}\n"

    p.write_text(content, encoding="utf-8")
    logger.info("Successfully updated .env with EDGE_TRIAGE_THRESHOLD=%.3f, CLOUD_ANOMALY_THRESHOLD=%.3f", edge_triage_th, cloud_anomaly_th)
    return True


def run_threshold_evaluation(
    normal_video_dir: Path | str,
    anomaly_video_dir: Optional[Path | str] = None,
    channels: Optional[List[int]] = None,
    apply_env: bool = False,
    json_output_path: Optional[Path | str] = None,
    env_path: Optional[Path | str] = None,
    max_windows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes threshold calibration and ROC evaluation across all channels.
    """
    t0 = time.time()
    n_dir = Path(normal_video_dir)
    print("\n" + "=" * 70)
    print("  DAHUA SURVEILLANCE: THRESHOLD CALIBRATION & ROC EVALUATION")
    print("=" * 70)
    print(f"  Normal Video Directory:   {n_dir.resolve()}")
    print(f"  Anomaly Video Directory:  {Path(anomaly_video_dir).resolve() if anomaly_video_dir else 'Synthetic Perturbation Probes'}")
    print(f"  Apply to .env:            {apply_env}")
    print("=" * 70 + "\n")

    extractor = EdgeFeatureExtractor(
        model_type=edge_config.model_type,
        embedding_dim=edge_config.embedding_dim,
    )
    detector = QdrantEdgeDetector(
        qdrant_path=edge_config.qdrant_edge_path,
        embedding_dim=edge_config.embedding_dim,
    )

    # 1. Extract and Score Normal Calibration Clips
    print("Extracting & Scoring Normal Calibration Footage...")
    normal_channel_embeddings = extract_clip_embeddings(
        video_dir=n_dir,
        extractor=extractor,
        channels=channels,
        max_windows_per_video=max_windows,
    )
    normal_scores_by_ch = score_embeddings_against_baseline(
        channel_embeddings=normal_channel_embeddings,
        edge_detector=detector,
    )

    all_normal_scores: List[float] = []
    for scores in normal_scores_by_ch.values():
        all_normal_scores.extend(scores)

    if not all_normal_scores:
        print("✗ No normal evaluation scores computed. Ensure baseline is seeded or directory contains clips.")
        return {"status": "no_normal_scores"}

    # 2. Extract and Score Anomaly Clips (or generate synthetic perturbations)
    all_anomaly_scores: List[float] = []
    if anomaly_video_dir and Path(anomaly_video_dir).exists():
        print("Extracting & Scoring Anomalous Validation Clips...")
        anom_channel_embeddings = extract_clip_embeddings(
            video_dir=anomaly_video_dir,
            extractor=extractor,
            channels=channels,
            max_windows_per_video=max_windows,
        )
        anom_scores_by_ch = score_embeddings_against_baseline(
            channel_embeddings=anom_channel_embeddings,
            edge_detector=detector,
        )
        for scores in anom_scores_by_ch.values():
            all_anomaly_scores.extend(scores)
    else:
        # Generate synthetic orthogonal noise perturbation probes
        print("Generating Synthetic Perturbation Anomaly Probes for ROC sweep...")
        for ch, items in normal_channel_embeddings.items():
            for emb, _, _ in items:
                # Add heavy noise or orthogonal vector
                noise = np.random.randn(*emb.shape).astype(np.float32)
                perturbed = 0.1 * emb + 0.9 * (noise / np.linalg.norm(noise))
                synth_clip_id = f"synth-eval-{uuid.uuid4().hex[:6]}"
                score, details = detector.score_clip(
                    channel=ch,
                    embedding=perturbed,
                    clip_id=synth_clip_id,
                    timestamp=time.time(),
                )
                dist = float(details.get("d_baseline", score))
                all_anomaly_scores.append(dist)

    # 3. Compute Distribution Metrics
    overall_dist = compute_distribution(all_normal_scores)
    per_channel_dist: Dict[int, DistributionMetrics] = {
        ch: compute_distribution(scores) for ch, scores in normal_scores_by_ch.items()
    }

    # 4. Compute ROC Metrics
    roc_metrics = compute_roc_metrics(all_normal_scores, all_anomaly_scores)

    # 5. Threshold Recommendations
    # Edge Triage Threshold: Target 95th percentile of normal routine (~0.060)
    # Cloud Anomaly Threshold: Target 99th percentile of normal routine (~0.150)
    recommended_edge_th = max(0.040, min(0.120, round(overall_dist.p95, 3)))
    recommended_cloud_th = max(0.080, min(0.300, round(overall_dist.p99, 3)))

    # Display Calibration Matrix
    print("\n" + "=" * 70)
    print("  EMPIRICAL DISTANCE DISTRIBUTION MATRIX (NORMAL ROUTINE)")
    print("=" * 70)
    print("  Channel | Samples | Mean   | Std    | p50    | p90    | p95    | p99    | Max")
    print("-" * 70)
    for ch, d in sorted(per_channel_dist.items()):
        ch_name = central_config.parse_camera_names().get(ch, f"Cam {ch}")
        print(f"  CH {ch:02d}   | {d.sample_count:>7} | {d.mean:.4f} | {d.std:.4f} | {d.p50:.4f} | {d.p90:.4f} | {d.p95:.4f} | {d.p99:.4f} | {d.max:.4f}")
    print("-" * 70)
    print(f"  GLOBAL  | {overall_dist.sample_count:>7} | {overall_dist.mean:.4f} | {overall_dist.std:.4f} | {overall_dist.p50:.4f} | {overall_dist.p90:.4f} | {overall_dist.p95:.4f} | {overall_dist.p99:.4f} | {overall_dist.max:.4f}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("  RECOMMENDED SURVEILLANCE THRESHOLDS")
    print("=" * 70)
    print(f"  EDGE_TRIAGE_THRESHOLD:     {recommended_edge_th:.3f}  (Empirical 95th Percentile: {overall_dist.p95:.4f})")
    print(f"  CLOUD_ANOMALY_THRESHOLD:   {recommended_cloud_th:.3f}  (Empirical 99th Percentile: {overall_dist.p99:.4f})")
    print(f"  ROC-AUC Score:             {roc_metrics.auc_roc:.4f}")
    print(f"  Optimal F1 Score:          {roc_metrics.best_f1:.4f}  (at T = {roc_metrics.best_threshold:.3f})")
    print(f"  TPR at Optimal F1:         {roc_metrics.tpr_at_best_f1:.2%}")
    print(f"  FPR at Optimal F1:         {roc_metrics.fpr_at_best_f1:.2%}")
    print("=" * 70 + "\n")

    # 6. Apply to .env if requested
    if apply_env:
        target_env = env_path or (PROJECT_ROOT / ".env")
        update_env_thresholds(
            env_file_path=target_env,
            edge_triage_th=recommended_edge_th,
            cloud_anomaly_th=recommended_cloud_th,
        )

    results_data: Dict[str, Any] = {
        "status": "success",
        "elapsed_s": round(time.time() - t0, 2),
        "overall_distribution": overall_dist.to_dict(),
        "per_channel_distribution": {ch: d.to_dict() for ch, d in per_channel_dist.items()},
        "roc_metrics": asdict(roc_metrics),
        "recommended_thresholds": {
            "edge_triage_threshold": recommended_edge_th,
            "cloud_anomaly_threshold": recommended_cloud_th,
        },
    }

    # 7. Export JSON if requested
    if json_output_path:
        out_p = Path(json_output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
        logger.info("Exported evaluation metrics to: %s", out_p)

    return results_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 7.3: Empirical Threshold Calibration & ROC Evaluation Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="data/tests",
        help="Path to directory containing normal surveillance validation footage",
    )
    parser.add_argument(
        "--anomaly-dir",
        type=str,
        default=None,
        help="Optional path to directory containing known anomalous test clips",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Comma-separated channel filter (e.g. '1,2,3')",
    )
    parser.add_argument(
        "--apply-env",
        action="store_true",
        help="Automatically update EDGE_TRIAGE_THRESHOLD and CLOUD_ANOMALY_THRESHOLD in .env",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Path to write JSON evaluation metrics report",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Limit windows per video for faster calibration runs",
    )

    args = parser.parse_args()

    channels_filter = None
    if args.channels:
        try:
            channels_filter = [int(c.strip()) for c in args.channels.split(",") if c.strip()]
        except ValueError:
            logger.error("Invalid --channels format. Expected integers, e.g. '1,2,3'")
            return 1

    target_video_dir = Path(args.video_dir)
    if not target_video_dir.exists():
        fallback = PROJECT_ROOT / "data" / "tests"
        if fallback.exists():
            logger.info("Specified video dir '%s' not found; using fallback '%s'", target_video_dir, fallback)
            target_video_dir = fallback

    res = run_threshold_evaluation(
        normal_video_dir=target_video_dir,
        anomaly_video_dir=args.anomaly_dir,
        channels=channels_filter,
        apply_env=args.apply_env,
        json_output_path=args.json_output,
        max_windows=args.max_windows,
    )

    return 0 if res.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
