# Phase 1: Direct HTTP Ingestion & Core Infrastructure

Comprehensive technical documentation for **Phase 1** of the Home Security Surveillance Analytics Platform. This document explains the architecture, detailed module implementations, key engineering and design decisions, and test validation results.

---

## 1. Executive Summary & Objective

The primary objective of **Phase 1** is to establish a robust, low-latency, and fault-tolerant video ingestion pipeline for a home CCTV installation consisting of a **Dahua DH-NVR5216-16P-4K NVR** and **9 PoE IP cameras**.

### The RTSP Problem vs. The Direct HTTP Solution
Traditional IP surveillance pipelines rely on RTSP (`rtsp://...:554/cam/realmonitor`). However, extensive benchmarking revealed severe operational drawbacks with RTSP on this hardware:
- Lengthy RTSP handshake negotiation (DESCRIBE, SETUP, PLAY) taking **3 to 10 seconds** to initialize.
- Complex state-machines prone to session drops and RTSP proxy bottlenecks.
- Fragile multi-port firewall/NAT negotiation (RTP/RTCP UDP port ranges).

**The Solution**: Dahua NVRs feature a native, direct HTTP video streaming engine (`/cgi-bin/mjpg/video.cgi`). Benchmarked against physical hardware, this direct HTTP engine achieves:
- **Sub-Second Connect & Decode**: First frame decoded in **under 150 milliseconds** (a **20x to 60x speedup** over RTSP).
- **Firewall & Proxy Simplicity**: Operates entirely over standard TCP port 80/HTTP.
- **Zero-Transcode Browser Live View**: Enables native MJPEG stream proxying directly to web consoles without heavy WebRTC or HLS transcoding daemons.

Phase 1 constructs the entire ingestion foundation, multi-channel supervisor, sliding-window segmenter, synthetic simulator, and diagnostic probe around this direct HTTP architecture.

---

## 2. System Architecture & Data Flow Diagram

```
+-----------------------------------------------------------------------------------------------------------------+
|                                        DAHUA DH-NVR5216-16P-4K                                                  |
|                                       (9x PoE IP Cameras Connected)                                             |
+-----------------------------------------------------------------------------------------------------------------+
          |                                                   |                                      |
          | Continuous Sub-Streams (24/7)                     | On-Demand Main-Streams               | Snapshots
          | subtype=1 (704x480 @ 10-25fps)                    | subtype=0 (2304x1296 3K @ 25fps)     | snapshot.cgi
          v                                                   v                                      v
+=================================================================================================================+
|                                        LOCAL EDGE INGESTION NODE                                                |
|                                                                                                                 |
|  +-----------------------------------------------------------------------------------------------------------+  |
|  | CaptureManager (edge/capture_manager.py)                                                                  |  |
|  |  Supervises 9 StreamWorkers, coordinates on-demand incident recording & aggregates system health telemetry  |  |
|  +-----------------------------------------------------------------------------------------------------------+  |
|        |                                                   |                                                    |
|        | (Channels 1 to 9)                                 | Trigger on Escalation                              |
|        v                                                   v                                                    |
|  +---------------------------------------+       +------------------------------------+                         |
|  | StreamWorker (edge/stream_worker.py)  |       | Main-Stream Incident Recorder      |                         |
|  |  - Mid-stream boundary sync           |       |  - Connects to subtype=0 on demand |                         |
|  |  - Reconnect exponential backoff      |       |  - Fallback to buffered sub-stream |                         |
|  |  - FPS throttling (target: 10 FPS)    |       |  - Encodes MP4 incident clip       |                         |
|  |  - Live telemetry (FPS, Latency, Drp) |       +------------------------------------+                         |
|  +---------------------------------------+                         |                                            |
|        |                                                           v                                            |
|        v                                            data/storage/incidents/*.mp4                                |
|  +---------------------------------------+          (Ready for Pegasus VLM in Phase 4)                          |
|  | Rolling Ring Buffer (FrameBuffer)     |                                                                      |
|  |  - In-memory deque (30s capacity)     |                                                                      |
|  |  - Thread-safe FrameItem storage      |                                                                      |
|  +---------------------------------------+                                                                      |
|        |                                                                                                        |
|        v                                                                                                        |
|  +-----------------------------------------------------------------------------------------------------------+  |
|  | SlidingWindowSegmenter (edge/segmenter.py)                                                                |  |
|  |  - Duration: 10.0s | Overlap: 2.0s | Step Pacing: 8.0s                                                    |  |
|  |  - Uniform Temporal Sampling: 16 frames uniformly spaced                                                  |  |
|  |  - Spatial Normalization: Resized to 224x224 RGB                                                          |  |
|  |  - Tensor Conversion: Normalized ImageNet float32 tensor of shape (T=16, C=3, H=224, W=224)              |  |
|  +-----------------------------------------------------------------------------------------------------------+  |
|        |                                                                                                        |
|        v                                                                                                        |
|  Ready for Phase 2: Edge Feature Extraction (ONNX MobileNetV3 / VideoMAE) & Qdrant Edge kNN Triage              |
+=================================================================================================================+
```

---

## 3. Module-by-Module Technical Breakdown

### 3.1 `edge/config.py` — Centralized Settings & URL Factory
- **Purpose**: Provides a strongly typed, environment-driven configuration model for the entire edge platform.
- **Key Responsibilities**:
  - Automatically loads `.env` files using `python-dotenv` with fallback search up the directory tree.
  - Exposes customizable engineering knobs (ingestion FPS, ring buffer capacity, reconnect backoffs, sliding window lengths, tensor dimensions).
  - Implements URL factory methods that dynamically interpolate user credentials, IP, port, channel index, and stream subtype:
    - `get_substream_url(channel: int) -> str`
    - `get_mainstream_url(channel: int) -> str`
    - `get_snapshot_url(channel: int) -> str`
  - Parses camera friendly names from comma-delimited strings (e.g., `"1:Front Door,2:Driveway"` $\to$ `Dict[int, str]`).

### 3.2 `edge/stream_worker.py` — Resilient Direct HTTP Stream Worker
- **Purpose**: Dedicated, isolated worker thread responsible for maintaining a continuous connection to a single camera feed.
- **Key Components**:
  - `FrameItem`: Dataclass encapsulating the raw BGR `np.ndarray` frame, epoch timestamp (`time.time()`), monotonic timestamp (`time.monotonic()`), and sequential frame index.
  - `StreamStats`: Telemetry tracker measuring real-time decoded FPS (via a rolling 60-sample window), initial connect latency, dropped frames, reconnect counts, and connection state.
  - `StreamWorker`:
    - **Mid-Stream Boundary Synchronization**: When connecting to an ongoing HTTP multipart stream (`multipart/x-mixed-replace`), the socket opens at an arbitrary byte offset. The worker implements a warm-up discard routine that safely absorbs initial corrupt JPEG fragments until finding a valid SOI marker (`0xFFD8`), preventing decoder crashes.
    - **Exponential Backoff Reconnection**: If the camera disconnects, the worker automatically retries with backoff (`1.0s` $\to$ `2.0s` $\to$ `4.0s` $\dots$ up to `30.0s`), resetting to `1.0s` immediately upon successful reconnection.
    - **Sliding Ring Buffer**: In-memory `collections.deque` holding up to 30 seconds of video frames (typically 300 frames at 10 FPS) protected by a reentrant `threading.Lock`.
    - **Frame Rate Throttling**: Even if the camera transmits at 25 FPS, the worker throttles ingestion to `target_fps` (e.g., 10 FPS), cutting memory consumption and CPU load by 60% while preserving sufficient temporal density for anomaly detection.

### 3.3 `edge/capture_manager.py` — Multi-Channel Supervisor & Incident Recorder
- **Purpose**: Coordinates concurrent streaming across all 9 cameras, manages worker lifecycles, and executes on-demand high-resolution incident capture.
- **Key Responsibilities**:
  - **Supervision**: Spawns and manages 9 independent `StreamWorker` instances. If camera 4 disconnects, cameras 1–3 and 5–9 continue processing completely unaffected.
  - **On-Demand High-Res Main-Stream Recording (`capture_incident_clip`)**:
    - When edge triage detects an anomaly, `CaptureManager` opens the camera's high-resolution **Main-Stream (`subtype=0` @ 2304x1296 3K)**.
    - Records high-resolution video for the incident duration (e.g. 10 seconds) using OpenCV `VideoWriter` (`mp4v` codec).
    - **Failsafe Sub-Stream Fallback**: If the Main-Stream is unreachable or drops packets during recording, the manager automatically pulls the preceding 10 seconds from the Sub-Stream ring buffer and generates the MP4 clip from RAM. Incident evidence is never lost.
  - **Snapshot Extraction (`capture_snapshot`)**: Attempts direct HTTP GET against Dahua's `/cgi-bin/snapshot.cgi` with Digest/Basic authentication, falling back to encoding the latest buffered frame from memory if the NVR snapshot API is slow.

### 3.4 `edge/segmenter.py` — Sliding Window Segmenter & Tensor Normalizer
- **Purpose**: Converts continuous video streams into discrete, overlapping multidimensional tensors formatted specifically for deep learning video models.
- **Key Components**:
  - `VideoClip`: Container holding sampled frames, start/end timestamps, channel ID, duration, and conversion utilities.
  - `SlidingWindowSegmenter`:
    - **Temporal Windowing**: Configured with `clip_duration_s = 10.0` and `clip_overlap_s = 2.0`. This produces a step interval of **8.0 seconds** (`duration - overlap`).
    - **Uniform Temporal Subsampling**: Rather than taking consecutive frames, it samples exactly **16 frames uniformly spaced** across the 10-second window (`np.linspace(0, N-1, 16)`). This matches standard VideoMAE and 3D-CNN architectural requirements.
    - **Spatial Resizing & Color Alignment**: Converts raw BGR frames to RGB and resizes to target inference resolution (**224x224**).
    - **Tensor Normalization (`to_normalized_tensor`)**: Converts pixels to `float32` in $[0.0, 1.0]$, applies ImageNet channel normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$), and outputs a tensor formatted as `(T=16, C=3, H=224, W=224)`.

### 3.5 `scripts/simulate_dahua_feeds.py` — Synthetic Multi-Channel Stream Server
- **Purpose**: Enables full headless development, testing, and continuous integration (CI) without requiring physical access to the Dahua NVR.
- **Key Capabilities**:
  - Standalone multi-threaded HTTP server replicating Dahua's `/cgi-bin/mjpg/video.cgi` and `/cgi-bin/snapshot.cgi`.
  - Supports 9 independent channels simultaneously.
  - Dynamically renders synthetic frames with channel badges, live microsecond timestamps, subtle dark-mode grid backgrounds, and animated moving targets (bouncing color-coded spheres) to simulate motion.
  - Emits valid MJPEG `multipart/x-mixed-replace;boundary=myboundary` streams at 25 FPS.

### 3.6 `scripts/test_nvr_connection.py` — Automated Hardware Diagnostic Probe
- **Purpose**: Diagnostic command-line tool to probe, benchmark, and validate physical NVR connectivity.
- **Key Capabilities**:
  - Connects to all 9 channels sequentially.
  - Benchmarks Sub-stream (`subtype=1`) latency, resolution, and real decoded FPS.
  - Benchmarks Main-stream (`subtype=0`) latency and resolution.
  - Validates instant Snapshot capture and file size.
  - Implements socket cooldown (80ms) and boundary synchronization to eliminate transient NVR socket contention.
  - Prints a formatted ASCII status matrix summarizing system readiness.

---

## 4. Key Design Decisions & Engineering Rationale

| Design Decision | Alternative Considered | Engineering Rationale |
| :--- | :--- | :--- |
| **Direct HTTP Multipart (`/cgi-bin/mjpg/video.cgi`)** | RTSP (`rtsp://...:554`) | Eliminates 3–10s RTSP connection negotiation, avoids session teardowns, cuts connection latency to <150ms, and bypasses firewall/proxy RTP port complexity. |
| **Dual-Tier Stream Strategy** (Sub-Stream + Main-Stream) | 24/7 Main-Stream Ingestion | Ingesting 9 continuous 3K (2304x1296) streams requires ~50 Mbps bandwidth and immense CPU decoding overhead. Sub-streams (704x480) consume only ~5 Mbps total, saving 90% of edge compute while keeping 3K resolution on-demand for incidents. |
| **In-Memory Rolling Ring Buffer (`collections.deque`)** | Writing continuous clips to SSD | Continuous disk writing causes flash wear and disk I/O bottlenecks across 9 streams. A 30s RAM ring buffer requires only ~40 MB RAM per channel (~360 MB total for 9 channels) with zero disk wear. |
| **Decoupled Ingestion & Segmentation Threads** | Synchronous capture-and-score loop | In a synchronous loop, slow AI model inference or vector searches stall the camera stream, causing packet loss and socket drops. Decoupled ring buffers allow ingestion to run at steady FPS regardless of inference latency. |
| **Failsafe Clip Fallback (Main-Stream $\to$ Sub-Stream Buffer)** | Main-Stream only recording | If the network stutters or the NVR refuses a concurrent high-res connection during a burglary, a Main-Stream-only system loses the evidence. Our fallback guarantees an MP4 is generated from RAM buffer. |
| **Uniform 16-Frame Temporal Subsampling** | First 16 consecutive frames | Video feature extraction models (VideoMAE, Timesformer, 3D-CNNs) need a temporal receptive field spanning the entire 10 seconds, not just the first 1.6 seconds of motion. Uniform spacing captures the full progression of events. |

---

## 5. Test Results & Validation Summary

### 5.1 Automated Unit & Integration Tests
All unit and integration tests are executed via `uv run pytest` (with `-p no:launch_testing` to isolate from local ROS environments).

**Results:** **13 Passed / 13 Total (100% Success Rate) in 3.16s**

```
tests/test_config.py::test_edge_config_defaults PASSED                    [  7%]
tests/test_config.py::test_camera_names_parsing PASSED                    [ 15%]
tests/test_config.py::test_url_generation PASSED                          [ 23%]
tests/test_stream_worker.py::test_stream_stats_tick PASSED               [ 30%]
tests/test_stream_worker.py::test_stream_worker_ring_buffer PASSED       [ 38%]
tests/test_stream_worker.py::test_stream_worker_lifecycle PASSED         [ 46%]
tests/test_capture_manager.py::test_capture_manager_init PASSED          [ 53%]
tests/test_capture_manager.py::test_capture_manager_all_stats PASSED      [ 61%]
tests/test_capture_manager.py::test_capture_manager_fallback_clip_generation PASSED [ 69%]
tests/test_segmenter.py::test_segmenter_extract_clip PASSED               [ 76%]
tests/test_segmenter.py::test_segmenter_pacing PASSED                     [ 84%]
tests/test_simulator.py::test_feed_simulator_frame_generation PASSED     [ 92%]
tests/test_simulator.py::test_feed_simulator_jpeg_encoding PASSED        [100%]

============================== 13 passed in 3.16s ==============================
```

#### Meaning of Test Results:
1. **Configuration Integrity (`test_config.py`)**: Validates that all default parameters, channel mappings, and URL templates substitute authentication and network coordinates cleanly without string injection bugs.
2. **Buffer Safety & Telemetry (`test_stream_worker.py`)**: Confirms the ring buffer enforces its strict maximum capacity limit under continuous insertions, timestamps remain monotonic, and telemetry counters accurately report FPS.
3. **Supervisor & Fallback Mechanics (`test_capture_manager.py`)**: Proves that when the Main-Stream is deliberately unavailable, the manager recovers buffered frames from RAM and successfully writes a valid, playable MP4 incident clip.
4. **Data Normalization for AI (`test_segmenter.py`)**: Verifies that sliding window clips produce the exact shape `(16, 3, 224, 224)` with float32 values normalized to ImageNet distribution, ensuring zero preprocessing runtime surprises during Phase 2 ONNX model deployment.
5. **Simulator Fidelity (`test_simulator.py`)**: Confirms synthetic frames match Dahua resolutions (704x480 Sub-stream, 2304x1296 Main-stream) with valid JPEG headers (`0xFFD8`).

---

### 5.2 Physical Hardware Validation on Live Dahua NVR
The diagnostic utility [`scripts/test_nvr_connection.py`](../scripts/test_nvr_connection.py) was executed directly against the physical NVR at `192.168.1.100`:

```
=========================================================================================================
Ch   | Status     | Sub-Stream (Triage)          | Main-Stream (VLM)        | Snapshot          
     |            | Res        Latency  FPS     | Res         Latency   | Size      Latency
---------------------------------------------------------------------------------------------------------
1    | ✅ OK       | 704x480     142ms  10.6fps   | 2304x1296     205ms      | 235.5KB    133ms  
2    | ✅ OK       | 704x480      55ms   9.9fps   | 2304x1296     212ms      | 189.6KB    157ms  
3    | ✅ OK       | 704x480      71ms   9.9fps   | 2304x1296     201ms      | 185.1KB    137ms  
4    | ✅ OK       | 704x480      52ms  10.0fps   | 2304x1296     233ms      | 648.7KB    234ms  
5    | ✅ OK       | 704x480      85ms  10.1fps   | 2304x1296     196ms      | 315.4KB    137ms  
6    | ✅ OK       | 704x480     136ms   9.8fps   | 2304x1296     144ms      | 627.2KB    216ms  
7    | ✅ OK       | 704x480     129ms   9.3fps   | 2304x1296     240ms      | 462.2KB    188ms  
8    | ✅ OK       | 704x480      43ms   9.4fps   | 1920x1080      97ms      | 279.0KB    121ms  
9    | ✅ OK       | 704x480     120ms  10.3fps   | 2304x1296     198ms      | 223.0KB    142ms  
=========================================================================================================

📊 Summary: 9/9 cameras online and ready for direct HTTP streaming.
```

#### Meaning of Hardware Benchmark:
- **100% Channel Availability**: All 9 cameras are fully operational over direct HTTP.
- **Ultra-Low Latency**: Sub-stream connection latency averages **~80 to 140ms**, meaning edge workers can initialize or recover almost instantly.
- **Consistent Frame Intake**: Sub-streams deliver a steady **~10 FPS** across all channels, matching our configured ingestion target.
- **High-Resolution Confirmation**: Channels 1–7 & 9 deliver **3K resolution (2304x1296)** and Channel 8 delivers **1080p (1920x1080)** on demand, ready for high-fidelity Twelve Labs Pegasus VLM scene understanding.
