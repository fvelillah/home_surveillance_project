#!/usr/bin/env python3
"""
Dahua NVR RTSP Historical Playback & Diagnostics Tool.

Streams and tests recorded video footage from the NVR hard drive over RTSP.
Automatically syncs with the NVR's internal clock and supports both GUI and headless environments.

Usage:
  # Play channel 1 for the past 2 hours (auto-synced to NVR clock):
  python scripts/test_rtsp_playback.py 1

  # Play channel 2 for the past 48 hours:
  python scripts/test_rtsp_playback.py 2 48

  # Play channel 1 with explicit start and end timestamps (YYYY_MM_DD_HH_MM_SS):
  python scripts/test_rtsp_playback.py 1 2026_09_18_00_00_00 2026_09_19_12_00_00

  # Advanced options:
  python scripts/test_rtsp_playback.py --channel 1 --hours 24 --subtype 0 --save-sample
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import cv2
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

# Optional dotenv support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_nvr_time(ip: str, port: int, user: str, password: str, timeout: float = 3.0) -> Optional[datetime]:
    """Queries Dahua NVR internal system clock via HTTP CGI."""
    url = f"http://{ip}:{port}/cgi-bin/global.cgi?action=getCurrentTime"
    try:
        resp = requests.get(url, auth=HTTPDigestAuth(user, password), timeout=timeout)
        if resp.status_code == 401:
            resp = requests.get(url, auth=HTTPBasicAuth(user, password), timeout=timeout)

        if resp.status_code == 200 and "result=" in resp.text:
            match = re.search(r"result\s*=\s*([0-9]{4}[-/][0-9]{2}[-/][0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})", resp.text)
            if match:
                raw_time = match.group(1).replace("/", "-")
                return datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S")
    except Exception as exc:
        print(f"⚠️  Could not fetch NVR system time via HTTP ({exc}). Using host machine clock.")
    return None


def format_dahua_time(dt: datetime) -> str:
    """Formats datetime to Dahua RTSP playback timestamp: YYYY_MM_DD_HH_MM_SS."""
    return dt.strftime("%Y_%m_%d_%H_%M_%S")


def test_rtsp_live_fallback(ip: str, port: int, user: str, password: str, channel: int, subtype: int = 0) -> bool:
    """Probes live RTSP stream to verify credentials and port 554 connectivity."""
    live_url = f"rtsp://{user}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"
    print(f"\n🔍 Probing Live RTSP Stream (Channel {channel}) on port {port}...")
    cap = cv2.VideoCapture(live_url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"   ✓ Live RTSP Connection SUCCESSFUL ({w}x{h} px). Credentials & Port {port} are valid.")
            return True
    print(f"   ✗ Live RTSP Connection FAILED. Check credentials, RTSP port {port}, or NVR network access.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Dahua NVR RTSP Playback & Diagnostic Tool")
    parser.add_argument("channel_pos", nargs="?", type=int, default=None, help="Camera Channel ID (1-9)")
    parser.add_argument("time_arg1", nargs="?", type=str, default=None, help="Hours back (e.g. 2, 48) OR start timestamp (YYYY_MM_DD_HH_MM_SS)")
    parser.add_argument("time_arg2", nargs="?", type=str, default=None, help="End timestamp (YYYY_MM_DD_HH_MM_SS)")

    parser.add_argument("--channel", "-c", type=int, default=None, help="Camera Channel ID (1-9)")
    parser.add_argument("--hours", type=float, default=None, help="Hours of historical footage to request")
    parser.add_argument("--subtype", "-s", type=int, default=0, help="Stream subtype (0=Main, 1=Sub)")
    parser.add_argument("--save-sample", action="store_true", default=True, help="Save sample snapshot image on successful read")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum frames to read in headless mode (0 for unlimited)")
    parser.add_argument("--ip", default=os.getenv("DAHUA_NVR_IP", "192.168.1.100"))
    parser.add_argument("--http-port", type=int, default=int(os.getenv("DAHUA_NVR_HTTP_PORT", "80")))
    parser.add_argument("--rtsp-port", type=int, default=int(os.getenv("DAHUA_NVR_RTSP_PORT", "554")))
    parser.add_argument("--user", default=os.getenv("DAHUA_NVR_USER", "admin"))
    parser.add_argument("--password", default=os.getenv("DAHUA_NVR_PASSWORD", "your_password"))

    args = parser.parse_args()

    channel = args.channel or args.channel_pos or 1
    ip = args.ip
    http_port = args.http_port
    rtsp_port = args.rtsp_port
    user = args.user
    password = args.password
    subtype = args.subtype

    # Determine reference clock (NVR system clock preferred)
    print("=" * 65)
    print(" 🎥 Dahua NVR RTSP Playback Diagnostic (Method 1)")
    print("=" * 65)
    
    nvr_now = get_nvr_time(ip, http_port, user, password)
    if nvr_now:
        ref_time = nvr_now
        print(f" 🕒 Synchronized with NVR Clock : {ref_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        ref_time = datetime.now()
        print(f" 🕒 Using Local Host Clock     : {ref_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Parse Start and End Times
    if args.time_arg1 and args.time_arg2:
        starttime = args.time_arg1
        endtime = args.time_arg2
    elif args.time_arg1 and not args.time_arg2:
        try:
            hours = float(args.time_arg1)
            starttime = format_dahua_time(ref_time - timedelta(hours=hours))
            endtime = format_dahua_time(ref_time)
        except ValueError:
            starttime = args.time_arg1
            endtime = format_dahua_time(ref_time)
    elif args.hours:
        starttime = format_dahua_time(ref_time - timedelta(hours=args.hours))
        endtime = format_dahua_time(ref_time)
    else:
        # Default: past 2 hours
        starttime = format_dahua_time(ref_time - timedelta(hours=2))
        endtime = format_dahua_time(ref_time)

    playback_url = (
        f"rtsp://{user}:{password}@{ip}:{rtsp_port}/cam/playback"
        f"?channel={channel}&subtype={subtype}&starttime={starttime}&endtime={endtime}"
    )
    masked_url = (
        f"rtsp://{user}:***@{ip}:{rtsp_port}/cam/playback"
        f"?channel={channel}&subtype={subtype}&starttime={starttime}&endtime={endtime}"
    )

    print(f" Target NVR Host : {ip}:{rtsp_port}")
    print(f" Camera Channel  : {channel} (Subtype {subtype})")
    print(f" Playback Window : {starttime}  --->  {endtime}")
    print(f" Request URL     : {masked_url}")
    print("=" * 65)

    # Force TCP for RTSP transport
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    print("Connecting to RTSP playback endpoint...")
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(playback_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"\n❌ Error: Unable to open RTSP playback stream.")
        test_rtsp_live_fallback(ip, rtsp_port, user, password, channel, subtype)
        return

    # Check GUI windowing support
    gui_available = False
    try:
        test_win = "__gui_test__"
        cv2.namedWindow(test_win)
        cv2.destroyWindow(test_win)
        gui_available = True
    except Exception:
        gui_available = False

    print(f"Mode: {'GUI Video Window' if gui_available else 'Headless Console Streaming'}")
    print("Streaming started...\n")

    frame_count = 0
    t_first_frame = None
    saved_sample = False

    storage_dir = Path("./data/storage/snapshots")
    storage_dir.mkdir(parents=True, exist_ok=True)
    sample_file = storage_dir / f"playback_ch{channel}_sample.jpg"

    window_name = f"Dahua Playback - Ch {channel} ({starttime} to {endtime})"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if t_first_frame is None:
                t_first_frame = time.perf_counter()
                elapsed_connect = (t_first_frame - t0) * 1000
                h, w = frame.shape[:2]
                print(f"✓ First frame received in {elapsed_connect:.1f} ms | Resolution: {w}x{h} px")

            frame_count += 1

            # Save sample frame if requested
            if args.save_sample and not saved_sample:
                cv2.imwrite(str(sample_file), frame)
                print(f"📸 Sample frame saved to: {sample_file}")
                saved_sample = True

            # GUI display if supported
            if gui_available:
                cv2.putText(
                    frame,
                    f"Ch {channel} REC PLAYBACK | Frame: {frame_count}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n🛑 Playback stopped by user keypress.")
                    break
            else:
                # Headless logging
                if frame_count % 30 == 0:
                    fps_calc = frame_count / (time.perf_counter() - t_first_frame)
                    print(f"  ▶ [Headless] Streamed {frame_count} frames (~{fps_calc:.1f} FPS)...")
                if args.max_frames and frame_count >= args.max_frames:
                    print(f"\n✓ Reached limit of {args.max_frames} frames in headless diagnostic mode.")
                    break

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user (Ctrl+C).")

    cap.release()

    if gui_available:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("\n" + "=" * 65)
    if frame_count > 0:
        total_time = time.perf_counter() - (t_first_frame or t0)
        avg_fps = frame_count / max(0.001, total_time)
        print(f"🎉 PLAYBACK TEST SUCCESSFUL!")
        print(f"   Total Frames Decoded : {frame_count}")
        print(f"   Average Playback FPS : {avg_fps:.1f} FPS")
        if saved_sample:
            print(f"   Snapshot Sample Path : {sample_file.resolve()}")
    else:
        print(f"⚠️  No playback frames received (0 frames).")
        print(f"   Possible causes:")
        print(f"   1. No recordings exist on NVR HDD for Channel {channel} between {starttime} and {endtime}.")
        print(f"   2. The NVR clock or time range needs adjustment.")
        test_rtsp_live_fallback(ip, rtsp_port, user, password, channel, subtype)
    print("=" * 65)


if __name__ == "__main__":
    main()
