#!/usr/bin/env python3
"""
CLI Evaluation & Search Tool for Semantic Video Search Engine (backend/search.py).

Indexes surveillance video clips from data/tests/ into Twelve Labs Marengo and
Incident Metadata Store, allowing natural-language semantic video search and ranking.

Features:
  - Scans data/tests/ for surveillance clips (.mp4, .avi, .mov, .mkv, .webm).
  - Automatically indexes discovered clips into Marengo & Incident Memory.
  - Generates synthetic test clips on demand (--generate-sample).
  - Multi-modal query search combining Marengo visual search and VLM metadata.
  - Rich ANSI colorized terminal match cards with match reasons & danger badges.
  - Interactive search query shell (--interactive).
  - Structured JSON output mode (--json) and file export (--output).
  - Offline / CI mock mode (--mock) to test pipeline integration without API keys.

Usage:
  # Search test clips in data/tests/ for a query
  python scripts/evaluate_search.py --query "person striking window with tool"

  # Launch interactive search shell over indexed videos
  python scripts/evaluate_search.py --interactive

  # Automatically index and query with filters
  python scripts/evaluate_search.py --query "delivery" --channel 1 --min-severity 25

  # Generate sample clips and search immediately
  python scripts/evaluate_search.py --generate-sample --query "intruder"

  # Run in offline mock mode without Twelve Labs API keys
  python scripts/evaluate_search.py --query "intruder near fence" --mock

  # Output search results as JSON
  python scripts/evaluate_search.py --query "delivery van" --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import config
from backend.incidents import incident_engine
from backend.models import IncidentRecord, SemanticSearchResponse, VLMExplanation
from backend.search import search_engine
from backend.vlm_explainer import explain_incident
from backend import twelvelabs_client

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_search")

# ---------------------------------------------------------------------------
# ANSI Terminal Colors & Styling
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
WHITE = "\033[97m"
BG_RED = "\033[41m\033[97m"
BG_YELLOW = "\033[43m\033[30m"
BG_GREEN = "\033[42m\033[30m"
BG_CYAN = "\033[46m\033[30m"
BG_BLUE = "\033[44m\033[97m"


def color_score(score: float, confidence: str) -> str:
    """Format similarity score with color badge."""
    pct = round(score * 100.0, 1)
    conf_str = f"[{confidence.upper()}]"
    if score >= 0.70:
        return f"{BG_GREEN} {pct}% MATCH {conf_str} {RESET}"
    if score >= 0.40:
        return f"{BG_CYAN} {pct}% MATCH {conf_str} {RESET}"
    if score >= 0.20:
        return f"{BG_YELLOW} {pct}% MATCH {conf_str} {RESET}"
    return f"{DIM}{pct}% MATCH {conf_str}{RESET}"


def color_severity(badge: str, severity: int) -> str:
    """Format severity badge with color."""
    b = badge.upper()
    if b == "CRITICAL" or severity >= 75:
        return f"{RED}{BOLD}{b} ({severity}/100){RESET}"
    if b == "HIGH" or severity >= 50:
        return f"{MAGENTA}{BOLD}{b} ({severity}/100){RESET}"
    if b == "MODERATE" or severity >= 25:
        return f"{YELLOW}{b} ({severity}/100){RESET}"
    return f"{GREEN}{b} ({severity}/100){RESET}"


# ---------------------------------------------------------------------------
# Video Utilities & Synthetic Sample Generator
# ---------------------------------------------------------------------------

def extract_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Extracts resolution, frame rate, duration, and file size using OpenCV."""
    meta = {
        "file_path": str(video_path),
        "file_name": video_path.name,
        "size_bytes": video_path.stat().st_size if video_path.exists() else 0,
        "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2) if video_path.exists() else 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_s": 0.0,
    }

    if not video_path.exists():
        return meta

    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        meta["fps"] = round(cap.get(cv2.CAP_PROP_FPS), 2) or 25.0
        meta["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if meta["fps"] > 0 and meta["frame_count"] > 0:
            meta["duration_s"] = round(meta["frame_count"] / meta["fps"], 2)
        cap.release()

    return meta


def generate_synthetic_video(
    output_path: Path,
    duration_s: float = 5.0,
    fps: int = 25,
    width: int = 704,
    height: int = 480,
    scene_type: str = "intrusion",
) -> Path:
    """Generates a synthetic surveillance test MP4 video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = int(duration_s * fps)
    for f_idx in range(total_frames):
        t = f_idx / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if scene_type == "intrusion":
            # Dark night yard with intruder
            frame[:, :, 0] = 20
            frame[:, :, 1] = 25
            frame[:, :, 2] = 30
            # Draw house wall & window
            cv2.rectangle(frame, (80, 60), (width - 80, height - 40), (60, 60, 80), 2)
            cv2.rectangle(frame, (120, 100), (220, 220), (180, 180, 120), -1)
            # Draw moving subject with tool
            actor_x = int(140 + 40 * math.sin(t * 2.0))
            actor_y = int(160 + 20 * math.cos(t * 2.0))
            cv2.rectangle(frame, (actor_x, actor_y), (actor_x + 50, actor_y + 110), (0, 0, 200), -1)
            cv2.line(frame, (actor_x + 45, actor_y + 40), (actor_x + 100, actor_y - 10), (0, 255, 255), 4)
            cv2.putText(frame, "INTRUDER (TOOL DETECTED)", (actor_x - 10, actor_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        elif scene_type == "delivery":
            # Daytime porch with courier
            frame[:, :, 0] = 120
            frame[:, :, 1] = 140
            frame[:, :, 2] = 130
            # Door & porch
            cv2.rectangle(frame, (200, 80), (320, height - 40), (80, 40, 20), -1)
            # Courier with package box
            actor_x = int(150 + (width - 300) * (0.5 + 0.5 * math.sin(t * 1.0)))
            actor_y = 200
            cv2.rectangle(frame, (actor_x, actor_y), (actor_x + 40, actor_y + 100), (180, 100, 50), -1)
            cv2.rectangle(frame, (actor_x + 10, actor_y + 40), (actor_x + 50, actor_y + 70), (40, 100, 180), -1)
            cv2.putText(frame, "DELIVERY COURIER (PARCEL)", (actor_x - 10, actor_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        else:
            # Routine driveway
            frame[:, :, 0] = 80
            frame[:, :, 1] = 90
            frame[:, :, 2] = 100
            cv2.putText(frame, "ROUTINE PATROL / NORMAL BASELINE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # OSD Header
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"DAHUA CAM-01 | {ts_str} | REC", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        writer.write(frame)

    writer.release()
    return output_path


def discover_test_videos(target_path: Path) -> List[Path]:
    """Finds supported video files in the specified path or directory."""
    supported_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if target_path.is_file():
        if target_path.suffix.lower() in supported_exts:
            return [target_path]
        return []

    if target_path.is_dir():
        videos = [
            p for p in target_path.iterdir()
            if p.is_file() and p.suffix.lower() in supported_exts
        ]
        videos.sort(key=lambda p: p.name)
        return videos

    return []


# ---------------------------------------------------------------------------
# Indexing & VLM Mocking Pipeline
# ---------------------------------------------------------------------------

def mock_vlm_for_clip(video_path: Path, channel: int, camera_name: str) -> VLMExplanation:
    """Generates rich simulated VLM analysis for test indexing."""
    fname = video_path.name.lower()
    if "intrusion" in fname or "intruder" in fname or "window" in fname:
        return VLMExplanation(
            summary=f"A person in dark clothing uses a long tool to strike a window near {camera_name}.",
            actors=["intruder", "person"],
            action="struck window with long tool and attempted forced entry",
            objects=["window", "long tool", "brick house", "dark clothing"],
            risk_assessment="breach",
            recommended_action="Contact law enforcement immediately and dispatch security.",
            model="pegasus1.5 [OFFLINE MOCK]",
            latency_ms=850.0,
        )
    elif "delivery" in fname or "package" in fname or "courier" in fname:
        return VLMExplanation(
            summary=f"Delivery driver in uniform carries cardboard parcel to {camera_name}.",
            actors=["delivery driver", "courier"],
            action="deposited parcel near door and photographed delivery",
            objects=["cardboard parcel", "box", "delivery vest", "handheld scanner"],
            risk_assessment="routine",
            recommended_action="No security response required. Parcel logged.",
            model="pegasus1.5 [OFFLINE MOCK]",
            latency_ms=620.0,
        )
    elif "vehicle" in fname or "car" in fname or "driveway" in fname:
        return VLMExplanation(
            summary=f"Vehicle pulled into {camera_name} and turned headlights off.",
            actors=["driver", "vehicle", "car"],
            action="parking in driveway during nighttime",
            objects=["car", "headlights", "driveway"],
            risk_assessment="suspicious",
            recommended_action="Verify authorized resident vehicle.",
            model="pegasus1.5 [OFFLINE MOCK]",
            latency_ms=710.0,
        )
    else:
        return VLMExplanation(
            summary=f"Activity observed in {camera_name} active detection zone.",
            actors=["unidentified subject"],
            action="moving across field of view",
            objects=["walkway", "outer gate"],
            risk_assessment="routine",
            recommended_action="Review clip in daily digest.",
            model="pegasus1.5 [OFFLINE MOCK]",
            latency_ms=510.0,
        )


async def index_test_videos(
    videos: List[Path],
    mock_mode: bool = False,
) -> List[IncidentRecord]:
    """Indexes discovered video clips into IncidentFormationEngine and Twelve Labs."""
    indexed_records: List[IncidentRecord] = []
    cameras = config.parse_camera_names()

    for idx, vid_path in enumerate(videos, 1):
        ch = ((idx - 1) % config.num_cameras) + 1
        cam_name = cameras.get(ch, f"Camera {ch}")
        meta = extract_video_metadata(vid_path)

        # Form an incident record
        fname = vid_path.name.lower()
        if "intrusion" in fname or "intruder" in fname or "window" in fname:
            edge_score = 0.82
            cloud_score = 0.88
            ens_score = 0.85
        elif "delivery" in fname or "package" in fname:
            edge_score = 0.28
            cloud_score = 0.32
            ens_score = 0.30
        else:
            edge_score = 0.40
            cloud_score = 0.45
            ens_score = 0.42

        inc = incident_engine.process_escalation(
            channel=ch,
            camera_id=f"cam-{ch}",
            camera_name=cam_name,
            edge_score=edge_score,
            cloud_score=cloud_score,
            ensemble_score=ens_score,
            timestamp=time.time() - (idx * 600),
            clip_url=str(vid_path.resolve()),
            scene_id=f"{cam_name.lower().replace(' ', '_')}_zone",
        )

        if inc is None:
            continue

        # Attach VLM explanation and index video into Marengo
        if mock_mode or not twelvelabs_client.is_enabled():
            vlm = mock_vlm_for_clip(vid_path, ch, cam_name)
        else:
            try:
                # Real Twelve Labs upload & Marengo indexing
                upload_res = twelvelabs_client.upload_video(vid_path, index_type="both")
                m_vid = upload_res.get("marengo_video_id")
                p_vid = upload_res.get("pegasus_video_id")
                if m_vid:
                    inc.notes = f"marengo_video_id={m_vid}; pegasus_video_id={p_vid}"
                    inc.clip_url = f"{vid_path.resolve()}?video_id={m_vid}"
                vlm = await explain_incident(incident=inc, video_id=p_vid, clip_path=str(vid_path))
            except Exception as exc:
                logger.warning("Pegasus VLM / Marengo live call failed for '%s', using mock explanation: %s", vid_path.name, exc)
                vlm = mock_vlm_for_clip(vid_path, ch, cam_name)

        incident_engine.attach_vlm_explanation(inc.incident_id, vlm)
        indexed_records.append(inc)

    return indexed_records


# ---------------------------------------------------------------------------
# CLI Display Formatters
# ---------------------------------------------------------------------------

def print_banner() -> None:
    """Prints the application banner."""
    print(f"{CYAN}{BOLD}")
    print("=" * 80)
    print("   TWELVE LABS MARENGO & MULTI-MODAL SEMANTIC VIDEO SEARCH - EVALUATION CLI")
    print("=" * 80)
    print(f"{RESET}")


def display_search_results(res: SemanticSearchResponse) -> None:
    """Renders formatted search results table and cards in the terminal."""
    sep = f"{CYAN}{'─' * 80}{RESET}"
    print(sep)
    print(f"{BOLD}{WHITE} 🔍 SEARCH QUERY: \"{CYAN}{res.query}{WHITE}\"{RESET}")
    print(f"    • Total Matches: {BOLD}{GREEN if res.total_matches > 0 else YELLOW}{res.total_matches}{RESET}")
    print(f"    • Search Latency: {YELLOW}{res.latency_ms:.1f} ms{RESET}")
    if res.filters_applied:
        filters_str = ", ".join(f"{k}={v}" for k, v in res.filters_applied.items() if k != "query")
        if filters_str:
            print(f"    • Filters Applied: {DIM}{filters_str}{RESET}")
    print(sep)

    if not res.results:
        print(f"   {YELLOW}No matching surveillance clips found for this query.{RESET}")
        print(f"   {DIM}Try broadening your query keywords or adjusting filter parameters.{RESET}")
        print(sep)
        return

    for idx, match in enumerate(res.results, 1):
        dt_str = datetime.fromtimestamp(match.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        print()
        print(f" {BOLD}#{idx}. {color_score(match.score, match.confidence)}  {WHITE}{match.camera_name} (CH {match.channel}){RESET}  {DIM}[{dt_str}]{RESET}")
        print(f"     • Severity:     {color_severity(match.severity_badge, match.severity)}")
        print(f"     • Match Reason: {DIM}{match.match_reason}{RESET}")
        print(f"     • VLM Summary:  {WHITE}{match.summary}{RESET}")
        if match.actors:
            print(f"     • Actors:       {CYAN}{', '.join(match.actors)}{RESET}")
        if match.action:
            print(f"     • Action:       {WHITE}{match.action}{RESET}")
        if match.objects:
            print(f"     • Objects:      {DIM}{', '.join(match.objects)}{RESET}")
        if match.clip_url:
            print(f"     • Video Clip:   {BLUE}{match.clip_url}{RESET}")
        print(f"     {DIM}{'┄' * 76}{RESET}")

    print(sep)


# ---------------------------------------------------------------------------
# Interactive Query Shell
# ---------------------------------------------------------------------------

def interactive_shell(args: argparse.Namespace) -> None:
    """Runs interactive search prompt."""
    print(f"{GREEN}{BOLD}Interactive Semantic Search Shell Ready.{RESET}")
    print(f"{DIM}Enter queries (e.g. 'intruder with tool', 'delivery package', 'car in driveway').")
    print(f"Commands: 'list' (show all indexed clips), 'exit' or 'q' to quit.{RESET}\n")

    while True:
        try:
            prompt = input(f"{CYAN}{BOLD}search> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting search shell.")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            print("Exiting search shell.")
            break
        if prompt.lower() == "list":
            all_incs = incident_engine.list_incidents(limit=50)
            print(f"\n{BOLD}Total Indexed Surveillance Clips: {len(all_incs)}{RESET}")
            for inc in all_incs:
                sum_text = inc.vlm_explanation.summary if inc.vlm_explanation else "No VLM summary"
                print(f" • [CH {inc.channel}] {inc.camera_name} ({inc.severity_badge}): {sum_text}")
            print()
            continue

        res = search_engine.search(
            query=prompt,
            channel=args.channel,
            camera_id=args.camera_id,
            min_severity=args.min_severity,
            severity_badge=args.severity_badge,
            limit=args.limit,
            threshold=args.threshold,
        )
        display_search_results(res)


# ---------------------------------------------------------------------------
# CLI Argument Parser & Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI Evaluation & Search Tool for Semantic Video Search Engine (backend/search.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query", "-q", type=str, default="", help="Natural language query to search for.")
    parser.add_argument("--dir", "-d", type=str, default="data/tests", help="Directory containing test video clips.")
    parser.add_argument("--channel", "-c", type=int, default=None, help="Filter search results by camera channel (1-9).")
    parser.add_argument("--camera-id", type=str, default=None, help="Filter by camera ID (e.g. cam-01).")
    parser.add_argument("--min-severity", type=int, default=None, help="Filter by minimum severity score (0-100).")
    parser.add_argument("--severity-badge", type=str, default=None, help="Filter by severity badge (LOW, MODERATE, HIGH, CRITICAL).")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Maximum number of search results to return.")
    parser.add_argument("--threshold", type=str, default="medium", choices=["low", "medium", "high"], help="Twelve Labs search threshold.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive search prompt shell.")
    parser.add_argument("--generate-sample", action="store_true", help="Generate synthetic test clips if data/tests/ is empty.")
    parser.add_argument("--mock", action="store_true", help="Run in offline mock mode without Twelve Labs API calls.")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Save search results to JSON file.")
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()

    if not args.json:
        print_banner()

    test_dir = Path(args.dir)
    if not test_dir.is_absolute():
        test_dir = PROJECT_ROOT / test_dir

    # Discover videos
    videos = discover_test_videos(test_dir)

    # Generate samples if requested or if directory is empty
    if not videos and args.generate_sample:
        if not args.json:
            print(f"{YELLOW}No videos found in '{test_dir}'. Generating synthetic surveillance samples...{RESET}")
        s1 = generate_synthetic_video(test_dir / "sample_intrusion.mp4", duration_s=4.0, scene_type="intrusion")
        s2 = generate_synthetic_video(test_dir / "sample_delivery.mp4", duration_s=4.0, scene_type="delivery")
        videos = [s1, s2]
    elif not videos and not args.interactive:
        if not args.json:
            print(f"{YELLOW}No video clips found in '{test_dir}'.{RESET}")
            print(f"Use '--generate-sample' to create synthetic test videos or place .mp4 clips in data/tests/.")
        return 1

    # Index discovered videos
    if not args.json:
        print(f"Indexing {len(videos)} video clip(s) from '{test_dir}'...")

    indexed = await index_test_videos(videos, mock_mode=args.mock)
    if not args.json:
        print(f"{GREEN}✓ Successfully indexed {len(indexed)} surveillance event(s) into search engine.{RESET}\n")

    # Interactive mode
    if args.interactive or not args.query:
        if args.interactive or sys.stdin.isatty():
            interactive_shell(args)
            return 0

    # Direct query mode
    query = args.query or "intrusion"
    res = search_engine.search(
        query=query,
        channel=args.channel,
        camera_id=args.camera_id,
        min_severity=args.min_severity,
        severity_badge=args.severity_badge,
        limit=args.limit,
        threshold=args.threshold,
    )

    if args.json:
        print(json.dumps(res.model_dump(), indent=2))
    else:
        display_search_results(res)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res.model_dump(), indent=2), encoding="utf-8")
        if not args.json:
            print(f"\n{GREEN}Search results saved to '{out_path}'{RESET}")

    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
