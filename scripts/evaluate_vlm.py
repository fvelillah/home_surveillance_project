#!/usr/bin/env python3
"""
CLI Evaluation Tool for Twelve Labs Pegasus VLM Explainer (backend/vlm_explainer.py).

Evaluates surveillance video clips by uploading and generating structured AI incident
explanations via Twelve Labs Pegasus 1.2.

Features:
  - Scans data/tests/ for video files (.mp4, .avi, .mov, .mkv, .webm) by default.
  - Accepts specific video clip paths or Twelve Labs video IDs.
  - Generates synthetic test clips on demand (--generate-sample).
  - Rich ANSI colorized CLI terminal summary with risk assessment badges.
  - Structured JSON output mode (--json) and file export (--output).
  - Offline / CI mock mode (--mock) to test pipeline integration without API keys.

Usage:
  # Analyze a video in data/tests/
  python scripts/evaluate_vlm.py --video data/tests/intrusion_event.mp4

  # Automatically find and analyze videos in data/tests/
  python scripts/evaluate_vlm.py

  # Batch evaluate all test videos in data/tests/
  python scripts/evaluate_vlm.py --all

  # Generate a sample incident clip and evaluate it immediately
  python scripts/evaluate_vlm.py --generate-sample

  # Query Pegasus directly with an existing Twelve Labs video ID
  python scripts/evaluate_vlm.py --video-id <twelve_labs_video_id>

  # Output as raw JSON (for piping / automation)
  python scripts/evaluate_vlm.py --video data/tests/sample.mp4 --json
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
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import config
from backend.models import IncidentRecord, VLMExplanation
from backend.vlm_explainer import explain_incident
from backend import twelvelabs_client

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_vlm")

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
BG_MAGENTA = "\033[45m\033[97m"
BG_BLUE = "\033[44m\033[97m"


def color_risk(risk: str) -> str:
    """Format risk assessment with color badge."""
    r = risk.lower().strip()
    if r == "routine":
        return f"{BG_GREEN} ROUTINE {RESET}"
    if r == "suspicious":
        return f"{BG_YELLOW} SUSPICIOUS {RESET}"
    if r == "hazard":
        return f"{BG_MAGENTA} HAZARD {RESET}"
    if r == "breach":
        return f"{BG_RED} BREACH {RESET}"
    return f"{BOLD}{WHITE}{risk.upper()}{RESET}"


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
# Video Metadata & Synthetic Generator Helpers
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
    scene_label: str = "Front Door - Motion Triggered",
) -> Path:
    """Generates a synthetic surveillance test MP4 video with animated movement."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = int(duration_s * fps)
    for f_idx in range(total_frames):
        t = f_idx / fps

        # Background gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = 30 + int(15 * math.sin(t * 0.5))  # Blue
        frame[:, :, 1] = 35 + int(10 * math.cos(t * 0.5))  # Green
        frame[:, :, 2] = 40  # Red

        # Draw grid lines / doorway
        cv2.rectangle(frame, (100, 80), (width - 100, height - 60), (70, 70, 70), 2)
        cv2.line(frame, (width // 2, 80), (width // 2, height - 60), (60, 60, 60), 1)

        # Draw simulated intruder / moving subject
        actor_x = int(180 + (width - 360) * (0.5 + 0.5 * math.sin(t * 1.5)))
        actor_y = int(140 + 60 * math.sin(t * 3.0))
        cv2.rectangle(frame, (actor_x, actor_y), (actor_x + 60, actor_y + 140), (0, 0, 220), -1)
        cv2.circle(frame, (actor_x + 30, actor_y - 20), 25, (200, 200, 200), -1)

        # Draw bounding box & label
        cv2.rectangle(frame, (actor_x - 10, actor_y - 55), (actor_x + 70, actor_y + 145), (0, 255, 255), 2)
        cv2.putText(
            frame,
            "PERSON (0.92)",
            (actor_x - 10, actor_y - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Camera OSD overlay
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame,
            f"CAM-01 [TEST FEED] | {ts_str} | REC",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            scene_label,
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

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
# Incident Record Factory & Mock Runner
# ---------------------------------------------------------------------------

def create_incident_record(
    video_path: Optional[Path],
    meta: Optional[Dict[str, Any]],
    camera_name: str = "Front Door",
    channel: int = 1,
    severity: int = 75,
    incident_id: Optional[str] = None,
) -> IncidentRecord:
    """Builds a standard IncidentRecord instance for VLM evaluation."""
    now = time.time()
    dur = (meta.get("duration_s") or 10.0) if meta else 10.0
    inc_id = incident_id or f"inc-eval-{int(now)}-{channel}"

    if severity >= 75:
        badge = "CRITICAL"
    elif severity >= 50:
        badge = "HIGH"
    elif severity >= 25:
        badge = "MODERATE"
    else:
        badge = "LOW"

    return IncidentRecord(
        incident_id=inc_id,
        channel=channel,
        camera_id=f"cam-{channel:02d}",
        camera_name=camera_name,
        start_time=now - dur,
        end_time=now,
        duration_s=dur,
        peak_score=min(1.0, severity / 100.0 + 0.05),
        mean_score=min(1.0, severity / 100.0),
        smoothed_score=min(1.0, severity / 100.0),
        severity=severity,
        severity_badge=badge,
        status="OPEN",
        event_count=1,
        clip_url=str(video_path) if video_path else None,
        created_at=now,
        updated_at=now,
    )


def mock_vlm_explanation(incident: IncidentRecord, video_path: Optional[Path]) -> VLMExplanation:
    """Generates a simulated VLMExplanation response for offline testing."""
    fname = video_path.name.lower() if video_path else "video"
    if "intrusion" in fname or "intruder" in fname or incident.severity >= 70:
        return VLMExplanation(
            summary=f"Unidentified subject approaching entry zone in {incident.camera_name}.",
            actors=["unidentified person", "intruder"],
            action="loitering near entrance and attempting door handle",
            objects=["door handle", "backpack", "perimeter fence"],
            risk_assessment="breach",
            recommended_action=f"Activate perimeter deterrents and notify homeowner immediately.",
            model="pegasus1.5 [MOCK]",
            latency_ms=842.5,
        )
    elif "delivery" in fname or "package" in fname:
        return VLMExplanation(
            summary=f"Delivery driver dropping package near {incident.camera_name}.",
            actors=["delivery driver"],
            action="placed parcel on porch and walked away",
            objects=["parcel", "cardboard box", "uniform"],
            risk_assessment="routine",
            recommended_action="No security action required. Parcel delivered.",
            model="pegasus1.5 [MOCK]",
            latency_ms=615.2,
        )
    else:
        return VLMExplanation(
            summary=f"Movement detected in {incident.camera_name} active monitoring area.",
            actors=["visitor", "individual"],
            action="walking across camera field of view",
            objects=["paved walkway", "outer gate"],
            risk_assessment="suspicious",
            recommended_action=f"Review surveillance recording for {incident.camera_name}.",
            model="pegasus1.5 [MOCK]",
            latency_ms=710.0,
        )


# ---------------------------------------------------------------------------
# CLI Presentation Formatter
# ---------------------------------------------------------------------------

def print_banner() -> None:
    """Prints the application banner."""
    print(f"{CYAN}{BOLD}")
    print("=" * 76)
    print("   TWELVE LABS PEGASUS VLM INCIDENT EXPLAINER - EVALUATION CLI")
    print("=" * 76)
    print(f"{RESET}")


def display_analysis_card(
    video_path: Optional[Path],
    meta: Optional[Dict[str, Any]],
    incident: IncidentRecord,
    explanation: VLMExplanation,
) -> None:
    """Renders a rich visual report in terminal."""
    sep = f"{CYAN}{'─' * 76}{RESET}"

    print(sep)
    print(f"{BOLD}{WHITE} 📹 VIDEO CLIP DETAILS{RESET}")
    if video_path and meta:
        print(f"   • Path:        {CYAN}{meta['file_path']}{RESET}")
        print(f"   • Resolution:  {WHITE}{meta['width']}x{meta['height']}{RESET} @ {meta['fps']} fps")
        print(f"   • Duration:    {WHITE}{meta['duration_s']}s{RESET} ({meta['frame_count']} frames)")
        print(f"   • File Size:   {WHITE}{meta['size_mb']} MB{RESET}")
    else:
        print(f"   • Twelve Labs Video ID: {CYAN}{incident.incident_id}{RESET}")

    print()
    print(f"{BOLD}{WHITE} 🛡️ INCIDENT METADATA{RESET}")
    print(f"   • Incident ID: {WHITE}{incident.incident_id}{RESET}")
    print(f"   • Camera:      {BOLD}{incident.camera_name}{RESET} (CH {incident.channel})")
    print(f"   • Severity:    {color_severity(incident.severity_badge, incident.severity)}")
    print(f"   • Status:      {BOLD}{incident.status}{RESET}")

    print()
    print(f"{BOLD}{WHITE} 🤖 PEGASUS VLM ANALYSIS RESULTS{RESET}")
    print(f"   • Model:       {WHITE}{explanation.model}{RESET}")
    print(f"   • Risk Level:  {color_risk(explanation.risk_assessment)}")
    print(f"   • Latency:     {YELLOW}{explanation.latency_ms:.1f} ms{RESET} ({explanation.latency_ms / 1000.0:.2f}s)")
    print()
    print(f"   {BOLD}📝 Summary:{RESET}")
    print(f"      {WHITE}{explanation.summary}{RESET}")
    print()
    print(f"   {BOLD}👥 Actors Detected:{RESET}")
    if explanation.actors:
        for actor in explanation.actors:
            print(f"      - {CYAN}{actor}{RESET}")
    else:
        print(f"      {DIM}(None identified){RESET}")
    print()
    print(f"   {BOLD}⚡ Action Observed:{RESET}")
    print(f"      {WHITE}{explanation.action or 'N/A'}{RESET}")
    print()
    print(f"   {BOLD}🔍 Objects Visible:{RESET}")
    if explanation.objects:
        for obj in explanation.objects:
            print(f"      - {CYAN}{obj}{RESET}")
    else:
        print(f"      {DIM}(None listed){RESET}")
    print()
    print(f"   {BOLD}💡 Recommended Action:{RESET}")
    print(f"      {BOLD}{GREEN if explanation.risk_assessment == 'routine' else RED}{explanation.recommended_action}{RESET}")
    print(sep)


# ---------------------------------------------------------------------------
# Core Evaluation Async Logic
# ---------------------------------------------------------------------------

async def run_evaluation(
    video_path: Optional[Path],
    video_id: Optional[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Runs VLM explanation for a single video or video_id."""
    meta = extract_video_metadata(video_path) if video_path else None
    incident = create_incident_record(
        video_path=video_path,
        meta=meta,
        camera_name=args.camera_name,
        channel=args.channel,
        severity=args.severity,
        incident_id=args.incident_id,
    )

    if args.mock:
        explanation = mock_vlm_explanation(incident, video_path)
    else:
        if not twelvelabs_client.is_enabled():
            raise RuntimeError(
                "TWELVE_LABS_API_KEY is not set in environment or .env file.\n"
                "Please configure your Twelve Labs API key or use '--mock' to test offline."
            )
        clip_str = str(video_path) if video_path else None
        explanation = await explain_incident(incident=incident, video_id=video_id, clip_path=clip_str)

    incident.vlm_explanation = explanation

    result_payload = {
        "video_metadata": meta,
        "incident": incident.model_dump(),
        "vlm_explanation": explanation.model_dump(),
    }

    if not args.json:
        display_analysis_card(video_path, meta, incident, explanation)

    return result_payload


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Twelve Labs Pegasus VLM Explainer functionality on surveillance videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan data/tests/ and evaluate found videos:
  python scripts/evaluate_vlm.py

  # Evaluate a specific video file:
  python scripts/evaluate_vlm.py --video data/tests/intrusion_clip.mp4

  # Evaluate all videos in a directory:
  python scripts/evaluate_vlm.py --dir data/tests --all

  # Generate a synthetic sample incident video and test it:
  python scripts/evaluate_vlm.py --generate-sample

  # Test offline / without Twelve Labs API key:
  python scripts/evaluate_vlm.py --generate-sample --mock

  # Output in JSON format:
  python scripts/evaluate_vlm.py --video data/tests/sample.mp4 --json
        """,
    )

    parser.add_argument(
        "-v", "--video",
        type=str,
        default=None,
        help="Path to a video file to evaluate.",
    )
    parser.add_argument(
        "-d", "--dir",
        type=str,
        default="data/tests",
        help="Directory to scan for test videos (default: data/tests).",
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Evaluate all video clips found in the target directory.",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Twelve Labs video ID to query Pegasus directly without uploading a file.",
    )
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a synthetic test surveillance video in data/tests/ and evaluate it.",
    )
    parser.add_argument(
        "-c", "--camera-name",
        type=str,
        default="Front Door",
        help="Camera label to attach to incident metadata (default: 'Front Door').",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=1,
        help="Camera channel number (default: 1).",
    )
    parser.add_argument(
        "-s", "--severity",
        type=int,
        default=75,
        help="Incident severity score 0-100 (default: 75).",
    )
    parser.add_argument(
        "--incident-id",
        type=str,
        default=None,
        help="Custom incident ID (default: auto-generated timestamp).",
    )
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output raw JSON results instead of colored terminal UI.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="File path to save the JSON analysis results.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in offline mock mode without making live API requests to Twelve Labs.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.json:
        print_banner()

    # Step 1: Handle sample generation if requested
    target_dir = Path(args.dir)
    if not target_dir.is_absolute():
        target_dir = PROJECT_ROOT / target_dir

    if args.generate_sample:
        sample_path = target_dir / "sample_incident.mp4"
        if not args.json:
            print(f"{CYAN}Generating synthetic test surveillance video: {sample_path}{RESET}")
        generate_synthetic_video(sample_path, duration_s=5.0)
        args.video = str(sample_path)

    # Step 2: Resolve video targets
    videos_to_evaluate: List[Optional[Path]] = []
    video_id_target: Optional[str] = args.video_id

    if video_id_target:
        videos_to_evaluate = [None]
    elif args.video:
        vpath = Path(args.video)
        if not vpath.is_absolute():
            vpath = PROJECT_ROOT / vpath
        if not vpath.exists():
            print(f"{RED}Error: Video file not found at '{vpath}'{RESET}", file=sys.stderr)
            return 1
        videos_to_evaluate = [vpath]
    else:
        # Scan directory
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)

        discovered = discover_test_videos(target_dir)
        if not discovered:
            if not args.json:
                print(f"{YELLOW}No video files (.mp4, .avi, .mov, .mkv) found in '{target_dir}'.{RESET}")
                print()
                print(f"{WHITE}Quick options to get started:{RESET}")
                print(f"  1. Place an MP4 surveillance clip inside {BOLD}{target_dir}{RESET}")
                print(f"  2. Run with {BOLD}--generate-sample{RESET} to generate a test video automatically:")
                print(f"     {CYAN}python scripts/evaluate_vlm.py --generate-sample{RESET}")
                print(f"  3. Specify a specific video file with {BOLD}--video /path/to/clip.mp4{RESET}")
            return 0

        if args.all:
            videos_to_evaluate = discovered
            if not args.json:
                print(f"{CYAN}Found {len(discovered)} test video(s) in '{target_dir}'. Evaluating all...{RESET}\n")
        else:
            # Pick first/latest video
            selected = discovered[0]
            videos_to_evaluate = [selected]
            if not args.json:
                print(f"{CYAN}Found {len(discovered)} video(s) in '{target_dir}'. Evaluating '{selected.name}'...{RESET}")
                if len(discovered) > 1:
                    print(f"{DIM}(Use --all to evaluate all {len(discovered)} videos or --video <path> for a specific one){RESET}\n")

    # Step 3: Run evaluations
    all_results = []
    for idx, vid in enumerate(videos_to_evaluate):
        if not args.json and len(videos_to_evaluate) > 1:
            print(f"\n{BOLD}{CYAN}=== Video {idx + 1} of {len(videos_to_evaluate)} ==={RESET}")
        try:
            res = asyncio.run(run_evaluation(vid, video_id_target, args))
            all_results.append(res)
        except Exception as exc:
            if args.json:
                print(json.dumps({"error": str(exc), "video": str(vid) if vid else None}))
            else:
                print(f"\n{RED}{BOLD}Evaluation Failed:{RESET} {RED}{exc}{RESET}\n", file=sys.stderr)
            return 1

    # Step 4: JSON / Export output
    if args.json:
        output_data = all_results[0] if len(all_results) == 1 else all_results
        print(json.dumps(output_data, indent=2))

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            output_data = all_results[0] if len(all_results) == 1 else all_results
            json.dump(output_data, f, indent=2)
        if not args.json:
            print(f"\n{GREEN}Results saved to: {out_path}{RESET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
