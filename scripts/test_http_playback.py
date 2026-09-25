#!/usr/bin/env python3
"""
Dahua NVR Native HTTP CGI Recording Search & Download Tool (Method 2).

Searches and downloads recorded video clips from the NVR hard drive using Dahua's
native HTTP REST/CGI interface (mediaFileFind.cgi and loadfile.cgi).

Usage:
  # Search recorded files for Channel 1 over the last 2 hours:
  python scripts/test_http_playback.py 1

  # Search recorded files for Channel 2 over the last 48 hours:
  python scripts/test_http_playback.py 2 48

  # Search and download the most recent recorded clip to disk:
  python scripts/test_http_playback.py 1 2 --download

  # Download up to 30 seconds of footage for a specific channel:
  python scripts/test_http_playback.py --channel 1 --hours 1 --download --max-mb 50
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

# Optional dotenv support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class DahuaHTTPMediaClient:
    """Client for Dahua NVR media search (mediaFileFind.cgi) and download (loadfile.cgi)."""

    def __init__(self, ip: str, port: int, user: str, password: str, timeout: float = 10.0):
        self.ip = ip
        self.port = port
        self.user = user
        self.password = password
        self.timeout = timeout
        self.base_url = f"http://{ip}:{port}"
        self.session = requests.Session()

    def _request(self, path: str, params: Optional[Dict[str, Any]] = None, stream: bool = False) -> requests.Response:
        """Sends authenticated GET request with Digest/Basic auth fallback."""
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.get(
                url,
                params=params,
                auth=HTTPDigestAuth(self.user, self.password),
                timeout=self.timeout if not stream else (self.timeout, 60.0),
                stream=stream,
            )
            if resp.status_code == 401:
                resp = self.session.get(
                    url,
                    params=params,
                    auth=HTTPBasicAuth(self.user, self.password),
                    timeout=self.timeout if not stream else (self.timeout, 60.0),
                    stream=stream,
                )
            return resp
        except Exception as exc:
            raise ConnectionError(f"HTTP request to {url} failed: {exc}") from exc

    def get_nvr_time(self) -> Optional[datetime]:
        """Fetches NVR internal clock time."""
        try:
            resp = self._request("/cgi-bin/global.cgi", params={"action": "getCurrentTime"})
            if resp.status_code == 200 and "result=" in resp.text:
                match = re.search(r"result\s*=\s*([0-9]{4}[-/][0-9]{2}[-/][0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})", resp.text)
                if match:
                    raw_time = match.group(1).replace("/", "-")
                    return datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S")
        except Exception as exc:
            print(f"⚠️  Could not fetch NVR time ({exc}). Using host machine clock.")
        return None

    def create_finder_session(self) -> Optional[str]:
        """Creates a mediaFileFind search session and returns the objectId."""
        resp = self._request("/cgi-bin/mediaFileFind.cgi", params={"action": "factory.create"})
        if resp.status_code == 200:
            match = re.search(r"result\s*=\s*([0-9]+)", resp.text)
            if match:
                return match.group(1)
            # Alternative format: object=123
            match_obj = re.search(r"object\s*=\s*([0-9]+)", resp.text)
            if match_obj:
                return match_obj.group(1)
        return None

    def destroy_finder_session(self, object_id: str) -> None:
        """Closes and destroys the mediaFileFind session."""
        try:
            self._request("/cgi-bin/mediaFileFind.cgi", params={"action": "destroy", "object": object_id})
        except Exception:
            pass

    def search_recordings(
        self,
        channel: int,
        start_time: datetime,
        end_time: datetime,
        file_types: Optional[List[str]] = None,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Searches recorded video files on NVR HDD between start_time and end_time.
        Returns a list of parsed recording metadata dictionaries.
        """
        object_id = self.create_finder_session()
        if not object_id:
            print("❌ Failed to create mediaFileFind session on NVR.")
            return []

        types = file_types or ["dav", "mp4"]
        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

        # Try Dahua channel index (Dahua CGI condition.Channel is usually 0-indexed or 1-indexed)
        # We query with 0-indexed channel first (e.g. channel 1 -> condition.Channel=0)
        ch_idx = channel - 1 if channel >= 1 else 0

        params = {
            "action": "findFile",
            "object": object_id,
            "condition.Channel": ch_idx,
            "condition.StartTime": start_str,
            "condition.EndTime": end_str,
            "condition.VideoStream": "Main",  # Main or Extra1
        }
        for i, t in enumerate(types):
            params[f"condition.Types[{i}]"] = t

        try:
            resp = self._request("/cgi-bin/mediaFileFind.cgi", params=params)
            if resp.status_code != 200 or "true" not in resp.text.lower():
                # Retry with 1-indexed channel if 0-indexed was rejected
                params["condition.Channel"] = channel
                resp = self._request("/cgi-bin/mediaFileFind.cgi", params=params)

            # Retrieve found records
            fetch_params = {
                "action": "findNextFile",
                "object": object_id,
                "count": max_results,
            }
            fetch_resp = self._request("/cgi-bin/mediaFileFind.cgi", params=fetch_params)
            if fetch_resp.status_code != 200:
                return []

            files = self._parse_find_results(fetch_resp.text)
            return files

        finally:
            self.destroy_finder_session(object_id)

    @staticmethod
    def _parse_find_results(text: str) -> List[Dict[str, Any]]:
        """Parses the text output of findNextFile into structured records."""
        items: Dict[int, Dict[str, Any]] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()

            match = re.match(r"items\[(\d+)\]\.(.+)", key)
            if match:
                idx = int(match.group(1))
                attr = match.group(2)
                if idx not in items:
                    items[idx] = {}
                items[idx][attr] = val

        records = []
        for idx in sorted(items.keys()):
            item = items[idx]
            rec = {
                "index": idx,
                "file_path": item.get("FilePath", ""),
                "start_time": item.get("StartTime", ""),
                "end_time": item.get("EndTime", ""),
                "length_bytes": int(item.get("Length", "0")) if item.get("Length", "").isdigit() else 0,
                "type": item.get("Type", ""),
                "cluster": item.get("Cluster", ""),
                "partition": item.get("Partition", ""),
            }
            if rec["file_path"] or rec["start_time"]:
                records.append(rec)
        return records

    def download_file(
        self,
        file_record: Dict[str, Any],
        output_path: Path,
        channel: int = 1,
        max_bytes: Optional[int] = None,
        progress_callback: Optional[Any] = None,
    ) -> bool:
        """Downloads a recording file via Dahua HTTP/RPC download endpoints."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        path = file_record.get("file_path", "")
        start_time = file_record.get("start_time", "")
        end_time = file_record.get("end_time", "")
        ch_idx = channel - 1 if channel >= 1 else 0

        print(f"📥 Initiating HTTP download from NVR...")
        print(f"   Source Path : {path or (start_time + ' -> ' + end_time)}")
        print(f"   Dest Path   : {output_path}")

        # Multiple Dahua firmware download candidate strategies
        strategies = []
        if path:
            strategies.extend([
                ("Direct RPC endpoint (/cgi-bin/RPC_Loadfile)", f"/cgi-bin/RPC_Loadfile{path}", None),
                ("Direct Root RPC endpoint (/RPC_Loadfile)", f"/RPC_Loadfile{path}", None),
                ("loadfile.cgi (fileName)", "/cgi-bin/loadfile.cgi", {"action": "startLoad", "fileName": path}),
                ("loadfile.cgi (filePath)", "/cgi-bin/loadfile.cgi", {"action": "startLoad", "filePath": path}),
                ("loadfile.cgi (file)", "/cgi-bin/loadfile.cgi", {"action": "startLoad", "file": path}),
            ])

        if start_time and end_time:
            strategies.extend([
                ("loadfile.cgi (time range, 0-indexed channel)", "/cgi-bin/loadfile.cgi", {
                    "action": "startLoad",
                    "channel": ch_idx,
                    "startTime": start_time,
                    "endTime": end_time,
                    "types[0]": "dav",
                }),
                ("loadfile.cgi (time range, 1-indexed channel)", "/cgi-bin/loadfile.cgi", {
                    "action": "startLoad",
                    "channel": channel,
                    "startTime": start_time,
                    "endTime": end_time,
                    "types[0]": "dav",
                }),
            ])

        resp = None
        active_strategy = None

        for name, req_path, params in strategies:
            try:
                print(f"   Trying strategy: {name}...")
                test_resp = self._request(req_path, params=params, stream=True)
                if test_resp.status_code == 200:
                    # Check that content is actually incoming
                    peek = test_resp.raw.peek(1024) if hasattr(test_resp.raw, "peek") else None
                    resp = test_resp
                    active_strategy = name
                    print(f"   ✓ Strategy '{name}' connected successfully (HTTP 200)!")
                    break
                else:
                    print(f"     ✗ Returned HTTP {test_resp.status_code}")
            except Exception as e:
                print(f"     ✗ Failed: {e}")

        if resp is None:
            print("❌ All HTTP download strategies failed on Dahua NVR.")
            return False

        try:
            total_downloaded = 0
            t_start = time.perf_counter()

            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        total_downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(total_downloaded)
                        elif total_downloaded % (2 * 1024 * 1024) == 0:
                            elapsed = max(0.001, time.perf_counter() - t_start)
                            speed_mb = (total_downloaded / (1024 * 1024)) / elapsed
                            print(f"   ▶ Downloaded {total_downloaded / (1024 * 1024):.1f} MB ({speed_mb:.2f} MB/s)...")

                        if max_bytes and total_downloaded >= max_bytes:
                            print(f"   ✓ Reached byte limit ({max_bytes / (1024*1024):.1f} MB). Stopping download.")
                            break

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            elapsed_total = max(0.001, time.perf_counter() - t_start)
            avg_speed = file_size_mb / elapsed_total
            print(f"✓ Download complete! Size: {file_size_mb:.2f} MB in {elapsed_total:.1f}s ({avg_speed:.2f} MB/s)")
            return True
        except Exception as exc:
            print(f"❌ Error during file download: {exc}")
            return False


def verify_video_file(file_path: Path) -> None:
    """Verifies that the downloaded video clip can be decoded by OpenCV."""
    print(f"\n🔍 Validating downloaded video container: {file_path.name}")
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        print(f"⚠️  Note: OpenCV could not natively open '{file_path.name}'. (If .dav format, FFmpeg can transcode it to .mp4).")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = frame_count / fps if fps > 0 else 0

    ret, first_frame = cap.read()
    cap.release()

    if ret and first_frame is not None:
        print(f"✓ Video successfully verified!")
        print(f"   Resolution : {width}x{height} px")
        print(f"   Framerate  : {fps:.1f} FPS")
        print(f"   Frames     : {frame_count} frames (~{duration_s:.1f} seconds)")
    else:
        print("⚠️  Could not decode initial frame from video file.")


def main():
    parser = argparse.ArgumentParser(description="Dahua NVR HTTP CGI Recording Search & Download (Method 2)")
    parser.add_argument("channel_pos", nargs="?", type=int, default=None, help="Camera Channel ID (1-9)")
    parser.add_argument("hours_pos", nargs="?", type=float, default=None, help="Hours back to search (e.g. 2, 48)")

    parser.add_argument("--channel", "-c", type=int, default=None, help="Camera Channel ID (1-9)")
    parser.add_argument("--hours", type=float, default=None, help="Hours of historical footage to search")
    parser.add_argument("--download", "-d", action="store_true", help="Download the first matching recorded file to disk")
    parser.add_argument("--max-mb", type=float, default=50.0, help="Max MB to download for testing (default: 50MB)")
    parser.add_argument("--ip", default=os.getenv("DAHUA_NVR_IP", "192.168.1.100"))
    parser.add_argument("--port", type=int, default=int(os.getenv("DAHUA_NVR_HTTP_PORT", "80")))
    parser.add_argument("--user", default=os.getenv("DAHUA_NVR_USER", "admin"))
    parser.add_argument("--password", default=os.getenv("DAHUA_NVR_PASSWORD", "your_password"))

    args = parser.parse_args()

    channel = args.channel or args.channel_pos or 1
    hours = args.hours or args.hours_pos or 2.0
    ip = args.ip
    port = args.port
    user = args.user
    password = args.password

    client = DahuaHTTPMediaClient(ip=ip, port=port, user=user, password=password)

    print("=" * 65)
    print(" 🎥 Dahua NVR Native HTTP Media Search & Download (Method 2)")
    print("=" * 65)
    print(f" Target NVR Host : http://{ip}:{port}")
    print(f" Camera Channel  : {channel}")
    print(f" Search Window   : Past {hours:g} hours")

    # Sync with NVR clock
    nvr_now = client.get_nvr_time()
    if nvr_now:
        ref_time = nvr_now
        print(f" 🕒 Synchronized with NVR Clock : {ref_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        ref_time = datetime.now()
        print(f" 🕒 Using Host Clock            : {ref_time.strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = ref_time - timedelta(hours=hours)
    end_time = ref_time

    print(f" Range           : {start_time.strftime('%Y-%m-%d %H:%M:%S')}  --->  {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    print("Searching NVR hard drive via mediaFileFind.cgi...")
    t0 = time.perf_counter()
    records = client.search_recordings(channel=channel, start_time=start_time, end_time=end_time)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\nSearch completed in {elapsed:.1f} ms. Found {len(records)} recording(s) on HDD:\n")

    if not records:
        print(f"⚠️  No recordings found on NVR HDD for Channel {channel} in the past {hours:g} hours.")
        print("   Tip: Try a wider search window (e.g., 'python scripts/test_http_playback.py 1 48')")
        return

    # Print table of recordings found
    print(f"{'#':<3} | {'Start Time':<19} | {'End Time':<19} | {'Size (MB)':<10} | {'Path'}")
    print("-" * 80)
    for i, r in enumerate(records, 1):
        size_mb = r['length_bytes'] / (1024 * 1024) if r['length_bytes'] > 0 else 0.0
        print(f"{i:<3} | {r['start_time']:<19} | {r['end_time']:<19} | {size_mb:>8.2f} MB | {r['file_path']}")

    # If --download is specified, download the first recorded file
    if args.download:
        first_rec = records[0]
        out_dir = Path("./data/storage/recordings")
        filename = f"ch{channel}_{first_rec['start_time'].replace(':', '-').replace(' ', '_')}.dav"
        out_path = out_dir / filename

        max_bytes = int(args.max_mb * 1024 * 1024) if args.max_mb > 0 else None
        print("\n" + "=" * 65)
        success = client.download_file(first_rec, out_path, channel=channel, max_bytes=max_bytes)
        if success:
            verify_video_file(out_path)
    else:
        print("\n💡 To download a sample recording to disk, run with '--download':")
        print(f"   python scripts/test_http_playback.py {channel} {hours:g} --download")
    print("=" * 65)


if __name__ == "__main__":
    main()
