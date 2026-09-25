#!/usr/bin/env python3
"""
Synthetic 9-Channel Dahua NVR HTTP Video Engine Simulator.

Provides a standalone HTTP mock server simulating Dahua NVR endpoints:
  - GET /cgi-bin/mjpg/video.cgi?channel={ch}&subtype={subtype}
  - GET /cgi-bin/snapshot.cgi?channel={ch}

Generates dynamic animated video frames with camera labels, timestamps,
and simulated motion for headless testing and CI without physical hardware.

Usage:
  python scripts/simulate_dahua_feeds.py --port 8888 --channels 9 --fps 25
"""

import argparse
import io
import math
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


class DahuaFeedSimulator:
    """Renders dynamic synthetic surveillance frames for any camera channel."""

    def __init__(self, num_channels: int = 9) -> None:
        self.num_channels = num_channels
        self.channel_names = {
            1: "Front Door",
            2: "Driveway",
            3: "Front Yard",
            4: "Garage",
            5: "Back Patio",
            6: "Backyard",
            7: "Side Alley North",
            8: "Side Alley South",
            9: "Perimeter Gate",
        }

    def generate_frame(self, channel: int, subtype: int, t: float) -> np.ndarray:
        """
        Generate an animated surveillance frame for the given channel.
        Subtype 1: 704x480 (Sub-stream)
        Subtype 0: 2304x1296 (Main-stream 3K)
        """
        if subtype == 0:
            width, height = 2304, 1296
            font_scale = 1.6
            thickness = 3
        else:
            width, height = 704, 480
            font_scale = 0.6
            thickness = 2

        # Create dark atmospheric background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (24, 28, 32)  # Dark slate background

        # Subtle background grid lines
        grid_step = 60 if subtype != 0 else 180
        for x in range(0, width, grid_step):
            cv2.line(frame, (x, 0), (x, height), (35, 42, 48), 1)
        for y in range(0, height, grid_step):
            cv2.line(frame, (0, y), (width, y), (35, 42, 48), 1)

        # Simulated moving object (bouncing circle)
        speed = 0.8 + (channel * 0.15)
        obj_x = int((width / 2) + (width * 0.35) * math.sin(t * speed))
        obj_y = int((height / 2) + (height * 0.3) * math.cos(t * speed * 1.3))
        radius = int(20 * (width / 704))

        # Color-code moving target based on channel
        hue = int((channel * 28) % 180)
        color_bgr = cv2.cvtColor(np.uint8([[[hue, 200, 230]]]), cv2.COLOR_HSV2BGR)[0][0]
        color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

        cv2.circle(frame, (obj_x, obj_y), radius, color, -1)
        cv2.circle(frame, (obj_x, obj_y), radius + 2, (255, 255, 255), 1)

        # Draw Camera Header & Live Metadata
        cam_name = self.channel_names.get(channel, f"Camera {channel}")
        stream_type = "SUB-STREAM (D1)" if subtype == 1 else "MAIN-STREAM (3K)"
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"

        # Top Overlay: Camera Name & Stream Type
        header_text = f"CAM {channel:02d}: {cam_name.upper()} | {stream_type}"
        cv2.putText(
            frame, header_text, (20, int(35 * (height / 480))),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA
        )

        # Bottom Overlay: Timestamp & Live Status
        footer_text = f"LIVE | {time_str} | REC [DAHUA NVR]"
        cv2.putText(
            frame, footer_text, (20, height - int(25 * (height / 480))),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, (0, 255, 0), max(1, thickness - 1), cv2.LINE_AA
        )

        # Blinking Recording Indicator
        if int(t * 2) % 2 == 0:
            rec_x = width - int(40 * (width / 704))
            rec_y = int(30 * (height / 480))
            cv2.circle(frame, (rec_x, rec_y), int(8 * (width / 704)), (0, 0, 255), -1)

        return frame

    def encode_jpeg(self, frame: np.ndarray, quality: int = 80) -> bytes:
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ret else b""


class DahuaSimulatorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for mock Dahua video and snapshot routes."""

    simulator = DahuaFeedSimulator()
    fps = 25

    def log_message(self, format, *args):
        # Suppress noisy request logs for video stream loops
        if "video.cgi" not in self.path:
            super().log_message(format, *args)

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        channel = int(query.get("channel", [1])[0])
        subtype = int(query.get("subtype", [1])[0])

        if parsed.path.startswith("/cgi-bin/mjpg/video.cgi"):
            self._handle_mjpeg_stream(channel, subtype)
        elif parsed.path.startswith("/cgi-bin/snapshot.cgi"):
            self._handle_snapshot(channel, subtype)
        elif parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "mock_simulator": true}')
        else:
            self.send_error(404, "Endpoint Not Found")

    def _handle_mjpeg_stream(self, channel: int, subtype: int):
        """Stream continuous multipart/x-mixed-replace MJPEG."""
        boundary = "myboundary"
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace;boundary={boundary}")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        frame_interval = 1.0 / self.fps

        try:
            while True:
                t0 = time.time()
                frame = self.simulator.generate_frame(channel, subtype, t0)
                jpeg_bytes = self.simulator.encode_jpeg(frame, quality=75 if subtype == 1 else 85)

                header = (
                    f"--{boundary}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg_bytes)}\r\n\r\n"
                ).encode("ascii")

                self.wfile.write(header)
                self.wfile.write(jpeg_bytes)
                self.wfile.write(b"\r\n")

                elapsed = time.time() - t0
                sleep_time = max(frame_interval - elapsed, 0.001)
                time.sleep(sleep_time)

        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_snapshot(self, channel: int, subtype: int):
        """Serve single JPEG snapshot."""
        now = time.time()
        frame = self.simulator.generate_frame(channel, subtype=0, t=now)
        jpeg_bytes = self.simulator.encode_jpeg(frame, quality=90)

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg_bytes)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(jpeg_bytes)


def run_server(port: int = 8888, channels: int = 9, fps: int = 25):
    DahuaSimulatorHandler.simulator = DahuaFeedSimulator(num_channels=channels)
    DahuaSimulatorHandler.fps = fps

    server = ThreadingHTTPServer(("0.0.0.0", port), DahuaSimulatorHandler)
    print("=" * 75)
    print(f" 🎥  MOCK DAHUA NVR STREAM SIMULATOR RUNNING ON PORT {port}")
    print("=" * 75)
    print(f"Channels available : 1 to {channels}")
    print(f"Framerate          : {fps} FPS")
    print(f"Sub-stream URL     : http://localhost:{port}/cgi-bin/mjpg/video.cgi?channel=1&subtype=1")
    print(f"Main-stream URL    : http://localhost:{port}/cgi-bin/mjpg/video.cgi?channel=1&subtype=0")
    print(f"Snapshot URL       : http://localhost:{port}/cgi-bin/snapshot.cgi?channel=1")
    print("=" * 75)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping simulator...")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock Dahua NVR HTTP Video Engine Simulator")
    parser.add_argument("--port", type=int, default=8888, help="HTTP Port (default: 8888)")
    parser.add_argument("--channels", type=int, default=9, help="Number of channels (default: 9)")
    parser.add_argument("--fps", type=int, default=25, help="Simulation FPS (default: 25)")
    args = parser.parse_args()

    run_server(port=args.port, channels=args.channels, fps=args.fps)
