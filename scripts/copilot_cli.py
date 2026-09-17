#!/usr/bin/env python3
"""
Conversational AI Security Copilot CLI Tool (backend/copilot.py).

Provides an interactive natural-language chat interface for operators to query
surveillance footage, investigate security incidents, inspect camera feeds, and
generate automated 24-hour daily surveillance digest reports.

Features:
  - Interactive multi-turn conversation shell with session memory.
  - Natural-language intent extraction across all 9 Dahua camera channels.
  - Video-grounded security Q&A with explicit evidence and clip citations.
  - Built-in commands: /digest, /history, /clear, /list, /help, /exit.
  - 24-hour daily surveillance summary generator (--digest).
  - Single-query CLI execution mode (--query).
  - Structured JSON export mode (--json) and file saving (--output).
  - Auto-indexes test footage in data/tests/ for instant reasoning.
  - Offline / CI mock mode (--mock).

Usage:
  # Launch the interactive Security Copilot CLI
  python scripts/copilot_cli.py

  # Ask a single question directly from CLI
  python scripts/copilot_cli.py --query "Was there any suspicious motion near the window?"

  # Generate 24-hour daily surveillance report
  python scripts/copilot_cli.py --digest

  # Ask question and output response as JSON
  python scripts/copilot_cli.py --query "Show critical incidents" --json

  # Run in offline mock mode
  python scripts/copilot_cli.py --mock
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
from backend.copilot import copilot_engine
from backend.incidents import incident_engine
from backend.models import CopilotChatResponse, DailyDigestResponse, IncidentRecord, VLMExplanation
from backend.vlm_explainer import explain_incident
from backend import twelvelabs_client

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("copilot_cli")

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


def color_threat_level(threat: str) -> str:
    """Format threat level with color badge."""
    t = threat.upper().strip()
    if t == "SEVERE":
        return f"{BG_RED} SEVERE THREAT {RESET}"
    if t == "ELEVATED":
        return f"{BG_MAGENTA} ELEVATED THREAT {RESET}"
    if t == "MODERATE":
        return f"{BG_YELLOW} MODERATE THREAT {RESET}"
    return f"{BG_GREEN} LOW THREAT (OPTIMAL) {RESET}"


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
# Sample Test Video Indexer
# ---------------------------------------------------------------------------

def discover_test_videos(target_path: Path) -> List[Path]:
    """Finds supported video files in directory."""
    supported_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if target_path.is_file():
        return [target_path] if target_path.suffix.lower() in supported_exts else []
    if target_path.is_dir():
        videos = [p for p in target_path.iterdir() if p.is_file() and p.suffix.lower() in supported_exts]
        videos.sort(key=lambda p: p.name)
        return videos
    return []


async def auto_seed_surveillance_data(test_dir: Path, mock_mode: bool = False) -> int:
    """Ensures incident engine has surveillance events for reasoning."""
    if len(incident_engine.list_incidents(limit=10)) > 0:
        return len(incident_engine.list_incidents(limit=10))

    videos = discover_test_videos(test_dir)
    cameras = config.parse_camera_names()

    if not videos:
        # Seed realistic test incidents directly
        t_now = time.time()
        # Incident 1: Front door delivery
        i1 = incident_engine.process_escalation(
            channel=1,
            camera_id="cam-1",
            camera_name=cameras.get(1, "Front Door"),
            edge_score=0.35,
            cloud_score=0.40,
            ensemble_score=0.38,
            timestamp=t_now - 7200,
            scene_id="front_door_porch",
        )
        if i1:
            incident_engine.attach_vlm_explanation(
                i1.incident_id,
                VLMExplanation(
                    summary="Delivery driver in uniform leaves cardboard package at front door.",
                    actors=["delivery driver", "courier"],
                    action="depositing parcel on porch and taking confirmation photo",
                    objects=["cardboard parcel", "delivery vest", "phone"],
                    risk_assessment="routine",
                    recommended_action="No security intervention needed. Parcel logged.",
                ),
            )

        # Incident 2: Back Patio Intrusion
        i2 = incident_engine.process_escalation(
            channel=5,
            camera_id="cam-5",
            camera_name=cameras.get(5, "Back Patio"),
            edge_score=0.88,
            cloud_score=0.92,
            ensemble_score=0.90,
            timestamp=t_now - 3600,
            scene_id="back_patio_window",
        )
        if i2:
            incident_engine.attach_vlm_explanation(
                i2.incident_id,
                VLMExplanation(
                    summary="Person in dark clothing striking sliding patio window with heavy crowbar.",
                    actors=["masked intruder", "subject"],
                    action="striking window glass with crowbar",
                    objects=["crowbar", "window", "dark hoodie"],
                    risk_assessment="breach",
                    recommended_action="Contact law enforcement immediately and dispatch security.",
                ),
            )
        return 2

    # If videos exist, index them
    for idx, vid_path in enumerate(videos, 1):
        ch = ((idx - 1) % config.num_cameras) + 1
        cam_name = cameras.get(ch, f"Camera {ch}")
        fname = vid_path.name.lower()

        if "intrusion" in fname or "intruder" in fname or "window" in fname:
            edge_s, cloud_s = 0.85, 0.90
            risk = "breach"
            summary = f"Person in dark clothing tampering with entry near {cam_name}."
            action = "striking window with tool"
            actors = ["intruder"]
            objects = ["tool", "window"]
            rec = "Dispatch security response immediately."
        elif "delivery" in fname or "package" in fname:
            edge_s, cloud_s = 0.28, 0.32
            risk = "routine"
            summary = f"Courier delivering package to {cam_name}."
            action = "placing parcel by entrance"
            actors = ["delivery driver"]
            objects = ["package", "uniform"]
            rec = "No action required."
        else:
            edge_s, cloud_s = 0.42, 0.45
            risk = "suspicious"
            summary = f"Movement detected in {cam_name} active monitoring zone."
            action = "walking across monitoring area"
            actors = ["individual"]
            objects = ["walkway"]
            rec = "Review footage."

        inc = incident_engine.process_escalation(
            channel=ch,
            camera_id=f"cam-{ch}",
            camera_name=cam_name,
            edge_score=edge_s,
            cloud_score=cloud_s,
            ensemble_score=(edge_s + cloud_s) / 2.0,
            timestamp=time.time() - (idx * 1800),
            clip_url=str(vid_path.resolve()),
        )
        if inc:
            if mock_mode or not twelvelabs_client.is_enabled():
                vlm = VLMExplanation(
                    summary=summary,
                    actors=actors,
                    action=action,
                    objects=objects,
                    risk_assessment=risk,
                    recommended_action=rec,
                    model="pegasus1.5 [OFFLINE MOCK]",
                )
            else:
                try:
                    vlm = await explain_incident(incident=inc, clip_path=str(vid_path))
                except Exception:
                    vlm = VLMExplanation(
                        summary=summary,
                        actors=actors,
                        action=action,
                        objects=objects,
                        risk_assessment=risk,
                        recommended_action=rec,
                        model="pegasus1.5 [OFFLINE MOCK]",
                    )
            incident_engine.attach_vlm_explanation(inc.incident_id, vlm)

    return len(videos)


# ---------------------------------------------------------------------------
# CLI Display Formatters
# ---------------------------------------------------------------------------

def print_banner() -> None:
    """Prints the application banner."""
    print(f"{CYAN}{BOLD}")
    print("=" * 80)
    print("   AI SECURITY COPILOT - CONVERSATIONAL SURVEILLANCE CLI CONSOLE")
    print("=" * 80)
    print(f"{RESET}")


def display_copilot_response(resp: CopilotChatResponse) -> None:
    """Renders formatted Copilot response card with cited evidence in the terminal."""
    sep = f"{CYAN}{'─' * 80}{RESET}"
    print(sep)
    print(f"{BOLD}{WHITE} 🛡️ AI SECURITY COPILOT RESPONSE{RESET}  {DIM}(Session: {resp.session_id} | Latency: {resp.latency_ms:.1f}ms){RESET}")
    print(sep)

    # Format the response text with nice indentation
    for line in resp.response.splitlines():
        print(f" {line}")

    if resp.evidence:
        print()
        print(f" {BOLD}{WHITE}📌 CITED EVIDENCE & SURVEILLANCE RECORDS ({len(resp.evidence)}):{RESET}")
        for idx, ev in enumerate(resp.evidence, 1):
            dt_str = datetime.fromtimestamp(ev.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            print(f"   [{idx}] {BOLD}{ev.camera_name} (CH {ev.channel}){RESET} - {color_severity(ev.severity_badge, ev.severity)} at {DIM}{dt_str}{RESET}")
            if ev.clip_url:
                print(f"       📹 Clip: {BLUE}{ev.clip_url}{RESET}")
    print(sep)


def display_daily_digest(digest: DailyDigestResponse) -> None:
    """Prints the rich 24-hour daily surveillance report."""
    sep = f"{CYAN}{'─' * 80}{RESET}"
    print(sep)
    print(f"{BOLD}{WHITE} 🛡️ 24-HOUR DAILY SURVEILLANCE SECURITY DIGEST{RESET}")
    print(f"    • Report ID:    {WHITE}{digest.digest_id}{RESET}")
    print(f"    • Threat Level: {color_threat_level(digest.threat_level)}")
    print(f"    • Incidents:    {BOLD}{digest.total_incidents}{RESET} ({digest.critical_incidents} Critical, {digest.high_incidents} High)")
    print(f"    • Total Events: {WHITE}{digest.total_events}{RESET}")
    print(sep)
    print()
    print(f"{BOLD}📊 Executive Summary:{RESET}")
    print(f"   {WHITE}{digest.executive_summary}{RESET}")
    print()
    print(f"{BOLD}📹 9-Channel Surveillance Matrix:{RESET}")
    for cs in digest.channel_summaries:
        status_color = RED if cs.critical_count > 0 else (MAGENTA if cs.high_count > 0 else (YELLOW if cs.moderate_count > 0 else GREEN))
        print(f"   • {BOLD}CH {cs.channel} ({cs.camera_name}):{RESET} {status_color}{cs.total_incidents} incidents (Peak: {cs.peak_severity}/100){RESET} — {cs.summary}")

    if digest.key_incidents:
        print()
        print(f"{BOLD}🚨 Key Incident Highlights:{RESET}")
        for idx, ki in enumerate(digest.key_incidents, 1):
            dt_str = datetime.fromtimestamp(ki.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            print(f"   [{idx}] {BOLD}{ki.camera_name} (CH {ki.channel}){RESET} — {color_severity(ki.severity_badge, ki.severity)} at {DIM}{dt_str}{RESET}")
            print(f"       • Summary: {WHITE}{ki.summary}{RESET}")
            if ki.actors:
                print(f"       • Actors:  {CYAN}{', '.join(ki.actors)}{RESET}")

    if digest.recommended_actions:
        print()
        print(f"{BOLD}💡 Recommended Security Actions:{RESET}")
        for act in digest.recommended_actions:
            print(f"   • {YELLOW}{act}{RESET}")

    print(sep)


# ---------------------------------------------------------------------------
# Interactive Copilot Shell
# ---------------------------------------------------------------------------

def interactive_copilot_shell(session_id: str, args: argparse.Namespace) -> None:
    """Runs interactive conversation shell loop."""
    print(f"{GREEN}{BOLD}AI Security Copilot Active.{RESET}")
    print(f"Session ID: {CYAN}{session_id}{RESET}")
    print(f"{DIM}Ask any questions about home surveillance activities (e.g. 'Any activity near the front door today?', 'Who was near the window?').")
    print(f"Commands: /digest (daily report), /history, /clear, /list, /help, /exit{RESET}\n")

    while True:
        try:
            prompt = input(f"{CYAN}{BOLD}copilot> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Copilot console.")
            break

        if not prompt:
            continue

        cmd = prompt.lower()
        if cmd in ("/exit", "exit", "quit", "/quit", "q"):
            print("Exiting Copilot console. Goodbye!")
            break

        if cmd in ("/help", "help"):
            print(f"\n{BOLD}Available Commands:{RESET}")
            print("  /digest    - Generate and view 24-hour daily surveillance report")
            print("  /history   - View conversational turn history in current session")
            print("  /clear     - Clear session conversation memory")
            print("  /list      - List all currently recorded surveillance incidents")
            print("  /exit      - Exit the Copilot console\n")
            continue

        if cmd in ("/digest", "digest"):
            digest = copilot_engine.generate_daily_digest()
            display_daily_digest(digest)
            continue

        if cmd in ("/history", "history"):
            history = copilot_engine.get_history(session_id)
            print(f"\n{BOLD}Session History ({len(history)} messages):{RESET}")
            for msg in history:
                role_color = CYAN if msg.role == "user" else GREEN
                print(f" [{role_color}{msg.role.upper()}{RESET}]: {msg.content}")
            print()
            continue

        if cmd in ("/clear", "clear"):
            copilot_engine.clear_history(session_id)
            print(f"\n{GREEN}✓ Session history cleared.{RESET}\n")
            continue

        if cmd in ("/list", "list"):
            all_incs = incident_engine.list_incidents(limit=20)
            print(f"\n{BOLD}Recorded Incidents ({len(all_incs)}):{RESET}")
            for inc in all_incs:
                dt_str = datetime.fromtimestamp(inc.start_time).strftime("%Y-%m-%d %H:%M:%S")
                print(f" • [CH {inc.channel}] {inc.camera_name} ({color_severity(inc.severity_badge, inc.severity)}) - {inc.vlm_explanation.summary if inc.vlm_explanation else 'No VLM'}")
            print()
            continue

        # Send question to Copilot
        resp = copilot_engine.chat(
            message=prompt,
            session_id=session_id,
        )
        display_copilot_response(resp)


# ---------------------------------------------------------------------------
# CLI Argument Parser & Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conversational AI Security Copilot CLI Tool (backend/copilot.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query", "-q", type=str, default="", help="Single question to ask Copilot.")
    parser.add_argument("--session-id", "-s", type=str, default=None, help="Specific session ID to attach conversation to.")
    parser.add_argument("--digest", action="store_true", help="Generate and print 24-hour daily surveillance report.")
    parser.add_argument("--channel", "-c", type=int, default=None, help="Constrain queries/digest to specific camera channel.")
    parser.add_argument("--dir", "-d", type=str, default="data/tests", help="Directory containing test video clips.")
    parser.add_argument("--mock", action="store_true", help="Run in offline mock mode without Twelve Labs API calls.")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Save output to a file.")
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()

    if not args.json:
        print_banner()

    test_dir = Path(args.dir)
    if not test_dir.is_absolute():
        test_dir = PROJECT_ROOT / test_dir

    # Auto-seed incident context from test clips or sample incidents
    indexed_count = await auto_seed_surveillance_data(test_dir, mock_mode=args.mock)
    if not args.json:
        print(f"{GREEN}✓ Surveillance Memory active: {indexed_count} security incident(s) loaded.{RESET}\n")

    # Generate daily digest mode
    if args.digest:
        digest = copilot_engine.generate_daily_digest(channel=args.channel)
        if args.json:
            print(json.dumps(digest.model_dump(), indent=2))
        else:
            display_daily_digest(digest)

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.suffix == ".md":
                out_p.write_text(digest.markdown_text, encoding="utf-8")
            else:
                out_p.write_text(json.dumps(digest.model_dump(), indent=2), encoding="utf-8")
            if not args.json:
                print(f"\n{GREEN}Daily digest saved to '{out_p}'{RESET}")
        return 0

    # Single-turn query mode
    if args.query:
        resp = copilot_engine.chat(
            message=args.query,
            session_id=args.session_id,
            channel=args.channel,
        )
        if args.json:
            print(json.dumps(resp.model_dump(), indent=2))
        else:
            display_copilot_response(resp)

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(resp.model_dump(), indent=2), encoding="utf-8")
            if not args.json:
                print(f"\n{GREEN}Response saved to '{out_p}'{RESET}")
        return 0

    # Interactive Shell mode
    active_session = args.session_id or f"session-cli-{int(time.time())}"
    interactive_copilot_shell(active_session, args)
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
