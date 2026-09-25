#!/usr/bin/env python3
"""
Dahua NVR Direct HTTP Video Stream Diagnostic & Validation Tool.

Probes all connected camera channels via Dahua's native HTTP video engine:
  - Sub-stream (subtype=1): Connection latency, frame decoding, actual FPS, resolution.
  - Main-stream (subtype=0): High-resolution frame decoding and resolution.
  - Snapshot API: Verifies instant single-frame capture.

Usage:
  python scripts/test_nvr_connection.py
  python scripts/test_nvr_connection.py --ip 192.168.1.100 --user admin --password your_password --channels 9
  python scripts/test_nvr_connection.py --channel 2  # test single channel
"""

import argparse
import os
import sys
import time
from typing import Dict, Any, Optional, List

import cv2
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

# Optional dotenv support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_default_config() -> Dict[str, Any]:
    return {
        "ip": os.getenv("DAHUA_NVR_IP", "192.168.1.100"),
        "port": int(os.getenv("DAHUA_NVR_HTTP_PORT", "80")),
        "user": os.getenv("DAHUA_NVR_USER", "admin"),
        "password": os.getenv("DAHUA_NVR_PASSWORD", "your_password"),
        "channels": int(os.getenv("DAHUA_NUM_CAMERAS", "9")),
    }


def build_stream_url(ip: str, port: int, user: str, password: str, channel: int, subtype: int) -> str:
    port_str = f":{port}" if port not in (80, 443) else ""
    return f"http://{user}:{password}@{ip}{port_str}/cgi-bin/mjpg/video.cgi?channel={channel}&subtype={subtype}"


def build_snapshot_url(ip: str, port: int, channel: int) -> str:
    port_str = f":{port}" if port not in (80, 443) else ""
    return f"http://{ip}{port_str}/cgi-bin/snapshot.cgi?channel={channel}"


def probe_snapshot(ip: str, port: int, user: str, password: str, channel: int, timeout: float = 3.0) -> Dict[str, Any]:
    url = build_snapshot_url(ip, port, channel)
    t0 = time.perf_counter()
    try:
        # Try digest first (common in Dahua), fallback to basic
        resp = requests.get(url, auth=HTTPDigestAuth(user, password), timeout=timeout)
        if resp.status_code == 401:
            resp = requests.get(url, auth=HTTPBasicAuth(user, password), timeout=timeout)
            
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200 and len(resp.content) > 1000:
            return {
                "ok": True,
                "size_kb": round(len(resp.content) / 1024, 1),
                "latency_ms": round(elapsed_ms, 1),
                "error": None
            }
        else:
            return {
                "ok": False,
                "size_kb": 0,
                "latency_ms": round(elapsed_ms, 1),
                "error": f"HTTP {resp.status_code}"
            }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "ok": False,
            "size_kb": 0,
            "latency_ms": round(elapsed_ms, 1),
            "error": str(e)
        }


# Suppress FFmpeg stderr boundary noise during HTTP MJPEG handshake
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")


def probe_stream(url: str, sample_frames: int = 15, max_warmup_attempts: int = 6) -> Dict[str, Any]:
    """Test video stream connection, initial frame latency, resolution, and FPS."""
    t_start = time.perf_counter()
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        return {
            "ok": False,
            "resolution": "N/A",
            "latency_ms": 0.0,
            "measured_fps": 0.0,
            "error": "Failed to open stream"
        }

    # Synchronize to clean JPEG boundary (discard initial in-flight fragment if any)
    frame = None
    t_first_frame = 0.0
    for _ in range(max_warmup_attempts):
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
            t_first_frame = time.perf_counter()
            break
        time.sleep(0.04)  # 40ms (1 frame interval at 25fps)

    connect_latency_ms = (t_first_frame - t_start) * 1000 if t_first_frame > 0 else (time.perf_counter() - t_start) * 1000

    if frame is None:
        cap.release()
        return {
            "ok": False,
            "resolution": "N/A",
            "latency_ms": round(connect_latency_ms, 1),
            "measured_fps": 0.0,
            "error": "Failed to read initial frame"
        }

    h, w = frame.shape[:2]
    resolution = f"{w}x{h}"

    # Sample successive frames to measure actual live FPS
    frame_times: List[float] = [t_first_frame]
    for _ in range(sample_frames - 1):
        ret, f = cap.read()
        if ret and f is not None:
            frame_times.append(time.perf_counter())
        else:
            time.sleep(0.04)

    cap.release()

    measured_fps = 0.0
    if len(frame_times) > 1:
        total_time = frame_times[-1] - frame_times[0]
        if total_time > 0:
            measured_fps = round((len(frame_times) - 1) / total_time, 1)

    return {
        "ok": True,
        "resolution": resolution,
        "latency_ms": round(connect_latency_ms, 1),
        "measured_fps": measured_fps,
        "error": None
    }


def run_diagnostics(ip: str, port: int, user: str, password: str, channels: List[int]):
    print("\n" + "=" * 80)
    print(" 🎥  DAHUA NVR DIRECT HTTP VIDEO ENGINE DIAGNOSTIC TOOL")
    print("=" * 80)
    print(f"Target NVR Host : http://{ip}:{port}")
    print(f"Auth User       : {user}")
    print(f"Channels Tested : {channels}")
    print("=" * 80 + "\n")

    results = []

    for ch in channels:
        print(f"▶ Testing Channel {ch}...", end="", flush=True)

        # 1. Test Sub-Stream (subtype=1)
        sub_url = build_stream_url(ip, port, user, password, channel=ch, subtype=1)
        sub_res = probe_stream(sub_url, sample_frames=10)

        # 2. Test Main-Stream (subtype=0)
        main_url = build_stream_url(ip, port, user, password, channel=ch, subtype=0)
        main_res = probe_stream(main_url, sample_frames=3)

        # 3. Test Snapshot
        snap_res = probe_snapshot(ip, port, user, password, channel=ch)

        channel_ok = sub_res["ok"] and main_res["ok"]
        status_icon = "✅ OK" if channel_ok else ("⚠️ PARTIAL" if (sub_res["ok"] or snap_res["ok"]) else "❌ FAIL")

        results.append({
            "channel": ch,
            "status": status_icon,
            "sub_res": sub_res,
            "main_res": main_res,
            "snap_res": snap_res,
        })

        print(f" {status_icon}")
        time.sleep(0.08)  # 80ms socket cooldown between cameras

    # Print Detailed Diagnostics Table
    print("\n" + "=" * 105)
    print(f"{'Ch':<4} | {'Status':<10} | {'Sub-Stream (Triage)':<28} | {'Main-Stream (VLM)':<24} | {'Snapshot':<18}")
    print(f"{'':<4} | {'':<10} | {'Res':<10} {'Latency':<8} {'FPS':<7} | {'Res':<11} {'Latency':<9} | {'Size':<9} {'Latency':<7}")
    print("-" * 105)

    total_online = 0
    for r in results:
        ch = r["channel"]
        status = r["status"]
        sub = r["sub_res"]
        main = r["main_res"]
        snap = r["snap_res"]

        if sub["ok"]:
            total_online += 1
            sub_info = f"{sub['resolution']:<10} {sub['latency_ms']:>4.0f}ms {sub['measured_fps']:>5.1f}fps"
        else:
            sub_info = f"{'ERROR: ' + (sub['error'] or 'Failed'):<28}"

        if main["ok"]:
            main_info = f"{main['resolution']:<11} {main['latency_ms']:>5.0f}ms"
        else:
            main_info = f"{'ERROR: ' + (main['error'] or 'Failed'):<24}"

        if snap["ok"]:
            snap_info = f"{str(snap['size_kb']) + 'KB':<9} {snap['latency_ms']:>4.0f}ms"
        else:
            snap_info = f"{'ERR: ' + str(snap['error']):<18}"

        print(f"{ch:<4} | {status:<10} | {sub_info:<28} | {main_info:<24} | {snap_info:<18}")

    print("=" * 105)
    print(f"\n📊 Summary: {total_online}/{len(channels)} cameras online and ready for direct HTTP streaming.\n")


def main():
    defaults = get_default_config()

    parser = argparse.ArgumentParser(description="Dahua NVR Direct HTTP Stream Diagnostic Tool")
    parser.add_argument("--ip", default=defaults["ip"], help=f"NVR IP Address (default: {defaults['ip']})")
    parser.add_argument("--port", type=int, default=defaults["port"], help=f"NVR HTTP Port (default: {defaults['port']})")
    parser.add_argument("--user", default=defaults["user"], help=f"NVR Username (default: {defaults['user']})")
    parser.add_argument("--password", default=defaults["password"], help="NVR Password")
    parser.add_argument("--channels", type=int, default=defaults["channels"], help=f"Total channels to probe (default: {defaults['channels']})")
    parser.add_argument("--channel", type=int, default=None, help="Probe a specific single channel (e.g., --channel 2)")

    args = parser.parse_args()

    if args.channel is not None:
        target_channels = [args.channel]
    else:
        target_channels = list(range(1, args.channels + 1))

    run_diagnostics(
        ip=args.ip,
        port=args.port,
        user=args.user,
        password=args.password,
        channels=target_channels,
    )


if __name__ == "__main__":
    main()
