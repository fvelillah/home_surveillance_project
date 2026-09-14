# Implementation Plan: Home Security Surveillance Analytics Platform (Direct HTTP Engine)

An end-to-end, modular, and systematic implementation roadmap for building an AI-powered home video surveillance analytics system. This plan adapts the edge-to-cloud architecture from the `video-anomaly-edge` reference project and tailors it specifically to a home CCTV installation comprising a **Dahua DH-NVR5216-16P-4K NVR** and **9 PoE IP cameras**, utilizing Dahua's native, high-performance **Direct HTTP Video Streaming Engine** (`/cgi-bin/mjpg/video.cgi`) instead of RTSP.

---

## 1. System Architecture & Topology

```
                                  +-------------------------------------------------------------+
                                  |                 Dahua DH-NVR5216-16P-4K                     |
                                  |              (9x PoE IP Cameras Connected)                  |
                                  +-------------------------------------------------------------+
                                                                 |
                                       Direct HTTP Sub-streams (9x channels, 704x480 @ 25fps)
                                       Direct HTTP Main-streams (On-demand 3K/4K clips on incident)
                                                                 v
+=======================================================================================================================+
|                                              LOCAL EDGE NODE (Home Server / Jetson)                                   |
|                                                                                                                       |
|  +---------------------------+   +----------------------------+   +------------------------------------------------+  |
|  | Multi-HTTP Stream Worker  |-->| Sliding Window Segmenter   |-->| Lightweight Edge Embedder                      |  |
|  | (9 Dahua Direct Streams)  |   | (10s clips, 2s overlap)     |   | (PyTorch MobileNetV3 / EfficientNet-B0)        |  |
|  +---------------------------+   +----------------------------+   +------------------------------------------------+  |
|                                                                                          |                            |
|                                                                                          v                            |
|  +---------------------------+   +----------------------------+   +------------------------------------------------+  |
|  | SQLite Offline Queue      |<--| Triage Filter              |<--| Qdrant Edge (Two-Shard on Disk)                |  |
|  | (Survives Network Drops)  |   | (Escalate top ~10-15%)     |   | - Mutable: Live recent context                 |  |
|  +---------------------------+   +----------------------------+   | - Immutable: Pre-built normal HNSW baseline    |  |
|                |                                                  +------------------------------------------------+  |
+================|======================================================================================================+
                 |
                 | HTTP / Secure WebSocket (Escalated clips & Edge Telemetry)
                 v
+=======================================================================================================================+
|                                              CENTRAL / CLOUD ANALYTICS TIER                                           |
|                                                                                                                       |
|  +-----------------------------------------------------------------------------------------------------------------+  |
|  | FastAPI Central Backend (Port 9876)                                                                             |  |
|  |                                                                                                                 |  |
|  |  +-----------------------------+  +-------------------------------+  +---------------------------------------+  |  |
|  |  | Twelve Labs Marengo 3.0     |  | Central Qdrant Vector Cluster |  | Multi-Model Ensemble Scorer           |  |  |
|  |  | (High-dim Video Embeddings) |  | (Multi-camera baseline & logs)|  | (70% Cloud + 30% Edge + Temp Boost)   |  |  |
|  |  +-----------------------------+  +-------------------------------+  +---------------------------------------+  |  |
|  |                 |                                                                    |                          |  |
|  |                 v                                                                    v                          |  |
|  |  +-----------------------------+  +-------------------------------+  +---------------------------------------+  |  |
|  |  | Twelve Labs Pegasus / VLM   |  | Incident Formation Engine     |  | Memory Governor & Anti-Poisoning      |  |  |
|  |  | (Natural Language Scene)    |  | (EMA Smoothing + Hysteresis)  |  | (Quarantine, Scrubbing, Per-Zone Cap) |  |  |
|  |  +-----------------------------+  +-------------------------------+  +---------------------------------------+  |  |
|  +-----------------------------------------------------------------------------------------------------------------+  |
+=======================================================================================================================+
                                                                 |
                                        REST API / WebSockets / Direct HTTP MJPEG Proxy
                                                                 v
+=======================================================================================================================+
|                                           OPERATOR CONSOLE / FRONTEND DASHBOARD                                       |
|                                                                                                                       |
|  +-----------------------------------------------------------------------------------------------------------------+  |
|  | Next.js / React Console (Port 3000)                                                                             |  |
|  |  - 9-Channel Low-Latency Live Camera Grid (Direct HTTP Stream Proxy with sub-second latency)                       |  |
|  |  - Real-Time Alert Queue with Severity Badges (0-100) & VLM Explanations                                        |  |
|  |  - Continuous Multi-Feed Timeline & Anomaly Heatmaps                                                            |  |
|  |  - Natural Language Video Search ("Find delivery driver yesterday", "Unusual motion near gate")                 |  |
|  |  - AI Security Copilot Chat (Video-grounded Q&A via Pegasus)                                                    |  |
|  |  - Baseline & Camera Governance Settings (Threshold calibration, quarantine review, device health)              |  |
|  +-----------------------------------------------------------------------------------------------------------------+  |
+=======================================================================================================================+
```

---

## 2. Operator Guide: Dahua Direct HTTP Streaming Integration

Dahua NVRs provide a native, high-performance HTTP video streaming interface that directly streams video via standard HTTP multipart (`multipart/x-mixed-replace`) or snapshot streams. This completely bypasses RTSP handshake negotiation, eliminating port proxy bottlenecks, packet loss, and RTSP session timeout errors.

### Why Direct HTTP Streaming?
- **Sub-Second Connect Time**: Connects and decodes the initial frame in under **0.5 seconds** (compared to 3–10s over RTSP).
- **Firewall & Proxy Friendly**: Operates over standard HTTP (port 80 or custom NVR web port), avoiding RTSP RTP/RTCP multi-port UDP/TCP complexity.
- **Direct Web & OpenCV Consumption**: Reads directly using standard `cv2.VideoCapture` or HTTP streaming proxies for zero-latency dashboard live view.

### Step 2.1: Dahua HTTP Video URI Specifications

| Stream Tier | Purpose | Resolution & FPS | Dahua HTTP URL Format |
| :--- | :--- | :--- | :--- |
| **Sub-Stream** (`subtype=1`) | Continuous Edge Triage, 10s Windowing & kNN Scoring | **704x480 (D1) @ 25 FPS** | `http://<user>:<password>@<nvr_ip>/cgi-bin/mjpg/video.cgi?channel=<ch>&subtype=1` |
| **Main-Stream** (`subtype=0`) | On-Demand Incident Capture & Cloud VLM Archiving | **2304x1296 (3K) / 4K @ 25 FPS** | `http://<user>:<password>@<nvr_ip>/cgi-bin/mjpg/video.cgi?channel=<ch>&subtype=0` |
| **Still Snapshot** | Instant Frame Grab / Thumbnail Generation | Native Camera Resolution | `http://<user>:<password>@<nvr_ip>/cgi-bin/snapshot.cgi?channel=<ch>` |

> *Note*: Replace `<user>`, `<password>`, `<nvr_ip>`, and `<ch>` (1–9) with your deployment values.

### Step 2.2: Live Verification in VLC / Media Player
To immediately verify an HTTP camera stream on any machine on your LAN:
1. Open **VLC Media Player** > **Media** > **Open Network Stream...** (`Ctrl + N`).
2. Enter the Sub-Stream URL (e.g., Channel 2):
   ```
   http://admin:abc12345@192.168.1.126/cgi-bin/mjpg/video.cgi?channel=2&subtype=1
   ```
3. Click **Play**. The stream starts instantly without RTSP negotiation delay.

### Step 2.3: Verification via CLI & ffprobe
Run the following commands from your edge host terminal:

```bash
# 1. Test HTTP Stream Header and Content-Type negotiation
curl -I -u admin:abc12345 "http://192.168.1.126/cgi-bin/mjpg/video.cgi?channel=2&subtype=1"

# 2. Decode Sub-Stream with ffprobe (<0.5s response)
ffprobe "http://admin:abc12345@192.168.1.126/cgi-bin/mjpg/video.cgi?channel=2&subtype=1"

# 3. Decode Main-Stream (3K) with ffprobe
ffprobe "http://admin:abc12345@192.168.1.126/cgi-bin/mjpg/video.cgi?channel=2&subtype=0"

# 4. Fetch a single JPEG snapshot
curl -u admin:abc12345 "http://192.168.1.126/cgi-bin/snapshot.cgi?channel=2" -o /tmp/cam2_test.jpg
```

### Step 2.4: Configure App Environment (`.env`)
Create or edit `.env` in the root of the project:

```env
# --- Dahua NVR HTTP Streaming Configuration ---
DAHUA_NVR_IP=192.168.1.126
DAHUA_NVR_HTTP_PORT=80
DAHUA_NVR_USER=admin
DAHUA_NVR_PASSWORD=abc12345
DAHUA_NUM_CAMERAS=9

# URL Templates for Direct HTTP Video Engine
DAHUA_SUBSTREAM_URL_TEMPLATE="http://{user}:{password}@{ip}:{port}/cgi-bin/mjpg/video.cgi?channel={channel}&subtype=1"
DAHUA_MAINSTREAM_URL_TEMPLATE="http://{user}:{password}@{ip}:{port}/cgi-bin/mjpg/video.cgi?channel={channel}&subtype=0"
DAHUA_SNAPSHOT_URL_TEMPLATE="http://{user}:{password}@{ip}:{port}/cgi-bin/snapshot.cgi?channel={channel}"

# Friendly names for all 9 channels
CAMERA_NAMES="1:Front Door,2:Driveway,3:Front Yard,4:Garage,5:Back Patio,6:Backyard,7:Side Alley North,8:Side Alley South,9:Perimeter Gate"

# Ingestion Parameters
INGEST_FPS=10
CLIP_DURATION_S=10
CLIP_OVERLAP_S=2
```

### Step 2.5: Run the Multi-Channel HTTP Diagnostic Tool
Run the automated diagnostic probe across all 9 camera channels:
```bash
python scripts/test_nvr_connection.py
```
This utility tests HTTP connection latency, validates frame resolution and framerate for both Sub and Main streams, captures a snapshot per channel, and prints an ASCII status matrix.

---

## 3. Systematic Phase-by-Phase Roadmap

### Phase 1: Direct HTTP Ingestion & Core Infrastructure
- **1.1 Workspace & Dependencies**: Set up `uv`, `pyproject.toml`, `.env.example`, Docker Compose definitions with OpenCV and HTTP video support.
- **1.2 Dahua Direct HTTP Multi-Stream Worker (`edge/stream_worker.py`)**: Multi-threaded HTTP video client using direct OpenCV/multipart chunk parsing, automatic reconnect with exponential backoff, rolling ring buffer, and FPS/latency instrumentation.
- **1.3 Multi-Channel Ingestion Manager (`edge/capture_manager.py`)**: Orchestrates 9 continuous Sub-Stream workers and maintains an on-demand Main-Stream trigger for high-res incident clip extraction.
- **1.4 Sliding Window Segmenter (`edge/segmenter.py`) & Mock Feed Simulator (`scripts/simulate_dahua_feeds.py`)**: Generates 10s video clips (2s overlap, 16 frames downsampled) for anomaly scoring, with a synthetic HTTP MJPEG server for headless CI testing.
- **1.5 Dahua HTTP Diagnostics Utility (`scripts/test_nvr_connection.py`)**: Multi-channel HTTP stream probe validating latency, FPS, and frame integrity.

### Phase 2: Edge Triage & On-Device Vector Store (Qdrant Edge)
- **2.1 Mandatory PyTorch Edge Feature Extractor & ONNX Runtime Acceleration (`edge/model.py`, `scripts/export_edge_model.py`)**: Strict PyTorch MobileNetV3-Small / EfficientNet-B0 backbone with temporal average pooling, producing 576-dim/1280-dim (or linearly projected) L2-normalized embeddings in <40ms per clip. Includes standalone and in-engine ONNX model export (`export_edge_model_onnx` with opset 14 and dynamic batch axes) and native ONNX Runtime execution acceleration.
- **2.2 Qdrant Edge Two-Shard Engine (`edge/detector.py`)**: Dual-shard (mutable recent context + immutable baseline) kNN scorer per camera feed.
- **2.3 Triage Filter & SQLite Offline Queue (`edge/queue.py`)**: High-recall filter ($T_{\text{edge}} = 0.060$) with crash-resilient SQLite persistence (WAL mode) and background auto-drain synchronization for internet outage resilience.
- **2.4 Edge FastAPI & Live Stream Proxy (`edge/main.py`)**: FastAPI service on port 7777 streaming live anomaly telemetry and low-latency HTTP video proxy to the dashboard.

### Phase 3: Central Cloud Backend, High-Precision Embedding & Scoring
- **3.1 Central FastAPI Architecture (`backend/main.py`)**: Core API routing, health checks, incident models (`backend/models.py`), and multi-camera stream catalog.
- **3.2 Twelve Labs Marengo 3.0 Integration (`backend/twelvelabs_client.py`)**: High-fidelity video embeddings (512-dim/1024-dim) with local PyTorch VideoMAE fallback server (`model_server.py`).
- **3.3 Central Qdrant Vector Baseline (`backend/anomaly.py`)**: Multi-camera vector baseline with spatial/temporal payload indexing and cosine similarity search.
- **3.4 Escalation Handler & Multi-Model Ensemble (`backend/escalation.py`, `backend/ensemble.py`)**: 70/30 cloud-edge score fusion, temporal escalation boost, and per-device performance tracking.

### Phase 4: Incident Formation, VLM Scene Understanding & Governance
- **4.1 Incident Formation Engine (`backend/incidents.py`)**: EMA score smoothing, hysteresis bounds ($T_{\text{start}}=0.080, T_{\text{end}}=0.050$), cooldown merging, and 0–100 severity normalization.
- **4.2 Natural Language VLM Scene Explainer (`backend/vlm_explainer.py`)**: Twelve Labs Pegasus / Gemini VLM integration generating structured incident breakdowns from high-res Main-Stream clips.
- **4.3 Memory Governor & Anti-Poisoning (`backend/memory.py`)**: 1-hour quarantine buffer, 7-day retention scrub, and per-camera vector caps to prevent environmental drift.
- **4.4 Streaming Backpressure & Load Shedding (`backend/streaming.py`)**: Adaptive 4-level load shedding (`NORMAL`, `SCORE_ONLY`, `PASSTHROUGH`, `SHED_LOAD`).

### Phase 5: Semantic Video Search & AI Security Copilot
- **5.1 Semantic Video Search (`backend/search.py`)**: Text-to-video search queries across all 9 feeds filtered by camera, date, and severity badge.
- **5.2 Conversational Security Copilot (`backend/copilot.py`)**: Interactive video Q&A and daily surveillance digest generation grounded in recorded video events.
- **5.3 Daily Digest & Routine Generator**: Automated daily morning summary of all home activity and flagged incidents.

### Phase 6: Multi-Camera Web Console & Live UI
- **6.1 Next.js 14 App Shell (`frontend/`)**: Modern dark-mode layout, glassmorphic styling, and WebSocket state management.
- **6.2 9-Channel Live Grid (`CameraGrid.tsx`, `CameraPanel.tsx`)**: 3x3 responsive video grid using direct HTTP stream proxies with per-camera live FPS, health, and anomaly score meters.
- **6.3 Real-Time Alert Queue & Incident Modal (`AlertQueue.tsx`, `IncidentModal.tsx`)**: Live incident cards with VLM summaries, video playback, and normal nearest-neighbor comparison grid.
- **6.4 Multi-Feed Continuous Timeline (`ContinuousTimeline.tsx`, `AnomalyMeter.tsx`)**: 24-hour interactive scrub bar and anomaly score heatmap across all 9 feeds.
- **6.5 Semantic Search & Intelligence Console (`OpsCopilot.tsx`)**: Video query bar, latent space UMAP visualization, and AI copilot drawer.

### Phase 7: Baseline Calibration, Evaluation & Production Deployment
- **7.1 Automated Baseline Recording Script (`scripts/record_baseline.py`)**: Captures 48 hours of normal home routine via HTTP Sub-Streams across all 9 cameras.
- **7.2 Baseline Clustering & Indexing (`scripts/embed_and_index_baseline.py`)**: Batch embedding and MiniBatchKMeans clustering (500 centroids per camera) into Qdrant.
- **7.3 Evaluation & Threshold Calibration (`scripts/evaluate_thresholds.py`)**: ROC analysis on staged test scenarios to tune decision boundaries.
- **7.4 Production Packaging**: Docker Compose definitions (`docker-compose.prod.yml`), systemd service units, and automated log rotations.
- **7.5 End-to-End Integration Test**: 9-camera stress testing, network drop simulation, and end-to-end incident escalation verification.

---

## 4. Master Task Checklist & Progress Tracker

### Overall Progress: `13 / 30 Tasks Completed (43.3%)`

```
[x] Phase 1: Direct HTTP Ingestion & Core Infrastructure (5/5)
[x] Phase 2: Edge Triage & Qdrant Edge (4/4)
[x] Phase 3: Central Backend & High-Precision kNN (4/4)
[ ] Phase 4: Incident Formation & VLM Explainer (0/4)
[ ] Phase 5: Semantic Search & AI Copilot (0/3)
[ ] Phase 6: Multi-Camera Web Console (0/6)
[ ] Phase 7: Calibration & Deployment (0/5)
```

---

### Phase 1: Direct HTTP Ingestion & Core Infrastructure
- [x] **Task 1.1**: Initialize project configuration (`pyproject.toml`, `.env.example`, `docker-compose.yml`, `.gitignore`) with `uv` and OpenCV HTTP support.
- [x] **Task 1.2**: Implement Dahua Direct HTTP stream worker (`edge/stream_worker.py`) with auto-reconnect backoff, rolling ring buffer, and FPS/latency metrics.
- [x] **Task 1.3**: Implement multi-channel capture manager (`edge/capture_manager.py`) orchestrating all 9 camera Sub-Streams and on-demand Main-Stream capture.
- [x] **Task 1.4**: Implement sliding window segmenter (`edge/segmenter.py`) and synthetic 9-channel HTTP stream simulator (`scripts/simulate_dahua_feeds.py`).
- [x] **Task 1.5**: Implement automated Dahua NVR HTTP connection diagnostic tool (`scripts/test_nvr_connection.py`).

### Phase 2: Edge Triage & On-Device Vector Store (Qdrant Edge)
- [x] **Task 2.1**: Implement mandatory PyTorch edge feature extractor (`edge/model.py`), standalone/in-engine ONNX export utility (`scripts/export_edge_model.py`, `export_edge_model_onnx`), and ONNX Runtime inference acceleration (supporting MobileNetV3 / EfficientNet with dynamic batch axes).
- [x] **Task 2.2**: Implement Qdrant Edge two-shard anomaly scorer (`edge/detector.py`) with mutable and immutable HNSW shards.
- [x] **Task 2.3**: Implement crash-resilient SQLite offline queue (`edge/queue.py` with native WAL mode) and background auto-drain sync.
- [x] **Task 2.4**: Implement Edge FastAPI & HTTP stream proxy server (`edge/main.py`) on port 7777 with live scoring stream.

### Phase 3: Central Cloud Backend, High-Precision Embedding & Scoring
- [x] **Task 3.1**: Implement Central FastAPI backend (`backend/main.py`) and Pydantic data schemas (`backend/models.py`).
- [x] **Task 3.2**: Implement Twelve Labs Marengo client wrapper (`backend/twelvelabs_client.py`) and local model server fallback (`model_server.py`).
- [x] **Task 3.3**: Implement central Qdrant baseline indexing and kNN scorer (`backend/anomaly.py`).
- [x] **Task 3.4**: Implement edge escalation receiver and multi-model ensemble scorer (`backend/escalation.py`, `backend/ensemble.py`) with temporal boosting.

### Phase 4: Incident Formation, VLM Scene Understanding & Governance
- [ ] **Task 4.1**: Implement incident formation engine (`backend/incidents.py`) with EMA smoothing, hysteresis thresholds, and cooldown merging.
- [ ] **Task 4.2**: Implement Twelve Labs Pegasus / Gemini VLM natural-language scene explainer (`backend/vlm_explainer.py`).
- [ ] **Task 4.3**: Implement memory governor (`backend/memory.py`) with quarantine buffer, 1-hour aging, 7-day retention scrub, and per-camera caps.
- [ ] **Task 4.4**: Implement streaming backpressure and load-shedding manager (`backend/streaming.py`).

### Phase 5: Semantic Video Search & AI Security Copilot
- [ ] **Task 5.1**: Implement multi-modal semantic video search endpoint (`backend/search.py`) combining Marengo queries with Qdrant metadata filters.
- [ ] **Task 5.2**: Implement conversational surveillance copilot chat engine (`backend/copilot.py`) grounded in video events.
- [ ] **Task 5.3**: Implement daily surveillance summary digest generator (`backend/copilot.py`).

### Phase 6: Multi-Camera Web Console & Live UI
- [ ] **Task 6.1**: Setup Next.js 14 project shell, dark mode theme, and global state (`AppContext.tsx`, `CameraContext.tsx`).
- [ ] **Task 6.2**: Implement 3x3 responsive live camera grid (`CameraGrid.tsx`, `CameraPanel.tsx`) with direct HTTP stream proxies and per-channel anomaly badges.
- [ ] **Task 6.3**: Implement real-time alert queue (`AlertQueue.tsx`, `ActiveIncidentCard.tsx`) with severity gauges and audible notifications.
- [ ] **Task 6.4**: Implement synchronized incident video modal (`IncidentModal.tsx`) with baseline nearest-neighbor comparison grid.
- [ ] **Task 6.5**: Implement 24-hour continuous multi-camera timeline (`ContinuousTimeline.tsx`, `AnomalyMeter.tsx`).
- [ ] **Task 6.6**: Implement semantic search console and interactive AI security copilot drawer (`OpsCopilot.tsx`).

### Phase 7: Baseline Calibration, Evaluation & Production Deployment
- [ ] **Task 7.1**: Implement automated 48-hour baseline recording script (`scripts/record_baseline.py`) over HTTP Sub-Streams.
- [ ] **Task 7.2**: Implement batch embedding, MiniBatchKMeans clustering (500 centroids/camera), and baseline ingestion (`scripts/embed_and_index_baseline.py`).
- [ ] **Task 7.3**: Implement threshold calibration and ROC evaluation suite (`scripts/evaluate_thresholds.py`).
- [ ] **Task 7.4**: Build production Docker Compose configs (`docker-compose.prod.yml`) and systemd service units.
- [ ] **Task 7.5**: Execute end-to-end integration and penetration test across all 9 camera feeds.

---

## 5. Verification & Testing Strategy

### 5.1 Automated Component Tests
```bash
# 1. Test Dahua Direct HTTP stream acquisition and frame buffer
pytest tests/test_dahua_http_stream.py

# 2. Test Qdrant Edge two-shard dual query and scoring
pytest tests/test_qdrant_edge.py

# 3. Test multi-model ensemble and temporal boosting logic
pytest tests/test_ensemble.py

# 4. Test incident builder hysteresis and cooldown merging
pytest tests/test_incidents.py

# 5. Test memory governor quarantine and scrub functions
pytest tests/test_governor.py
```

### 5.2 End-to-End Manual Verification
1. **Multi-Camera HTTP Ingest**: Stream 9 Dahua HTTP feeds simultaneously; verify sub-second startup, steady 25 FPS frame intake, and zero packet loss in the Edge Worker.
2. **Anomaly Detection & Escalation**: Inject simulated intrusion; verify edge triage escalates clip, cloud confirms, VLM generates description, and UI triggers audible alert.
3. **Offline Resilience**: Disconnect internet during activity; verify clips queue locally on disk in SQLite and flush to cloud when reconnected.
4. **Natural Language Search**: Query historical footage with natural-language text ("package delivered to porch") and verify matched video segments.
