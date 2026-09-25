# Phase 7: Baseline Calibration, Evaluation & Production Deployment

Comprehensive technical documentation for **Phase 7** of the Home Security Surveillance Analytics Platform. This document details the automated offline baseline vector calibration pipeline, direct in-memory Dahua ASF sub-stream video decoding, micro-clustering via `MiniBatchKMeans`, dual Qdrant Cloud/Edge vector indexing, empirical threshold calibration and ROC/AUC evaluation suite, containerized and daemonized production deployment artifacts, and end-to-end validation results.

---

## 1. Executive Summary & Design Vision

Phase 7 brings the surveillance platform from development into enterprise production readiness. The edge nodes require a rich, channel-specific baseline of normal activity (lighting variations, swaying trees, routine vehicle traffic, shadows) against which real-time frames can be evaluated using cosine distance metrics.

```
+=======================================================================================================================+
|                               PHASE 7: CALIBRATION, EVALUATION & PRODUCTION ARCHITECTURE                              |
+=======================================================================================================================+
|                                                                                                                       |
|   1. OFFLINE USB DISCOVERY (ZERO DISK COPY)       2. FEATURE EXTRACTION & CLUSTERING       3. DUAL VECTOR UPSERT      |
|   +---------------------------------------+       +--------------------------------+       +-----------------------+  |
|   | Dahua NVR USB External Storage        | ----> | PyTorch MobileNetV3 (576-dim)  | ----> | Qdrant Cloud Index    |  |
|   | NVR_ch01_extra1_20260901_20260907.asf |       | Sampled at 2.0 FPS in RAM      |       | (home_surveillance_   |  |
|   | • Streamed directly into memory       |       | MiniBatchKMeans (500 clusters) |       |  baseline)            |  |
|   | • No local disk wear or copies        |       | Hypersphere L2 Unit Normalized |       |                       |  |
|   +---------------------------------------+       +--------------------------------+       | Qdrant Edge Index     |  |
|                                                                    |                       | (Local RAM/Disk)      |  |
|                                                                    v                       +-----------------------+  |
|   4. EMPIRICAL THRESHOLD EVALUATION               5. PRODUCTION SYSTEM ORCHESTRATION                                  |
|   +---------------------------------------+       +-----------------------------------------------------------------+ |
|   | Cosine Distance Distribution Engine   |       | Docker Compose / Systemd Services                               | |
|   | • 50th, 90th, 95th, 99th Percentiles  | ----> | • surveillance-edge (Port 7777, RAM Ring Buffer, MobileNetV3)  | |
|   | • ROC / AUC Analysis & F1 Optima      |       | • surveillance-backend (Port 9876, Twelve Labs, Incident Engine)| |
|   | • Recommends EDGE & CLOUD Thresholds  |       | • surveillance-frontend (Port 3000, Next.js 14 Matrix UI)       | |
|   | • Automated .env Synchronization      |       | • Automated logrotate & health audit scripts                     | |
|   +---------------------------------------+       +-----------------------------------------------------------------+ |
+=======================================================================================================================+
```

### Key Requirements Delivered
1. **Zero Local Storage Overhead**: Ingests multi-gigabyte Dahua `.asf` sub-stream files directly into memory from USB mount points (`/media/usb`, `/mnt/usb`, or custom paths) without copying raw video files onto local drives.
2. **Channel-Specific Feature Condensation**: Extracts 576-dimensional MobileNetV3 embeddings and condenses thousands of raw frames per channel into exactly 500 representative baseline cluster centroids using `MiniBatchKMeans`.
3. **Hypersphere L2 Projection**: Strictly normalizes all centroid vectors to unit Euclidean norm ($\|\mathbf{c}\|_2 = 1.0$) ensuring geometric validity in cosine vector space.
4. **Dual Vector Upsert**: Concurrently provisions both Qdrant Cloud (for distributed cloud validation) and the Edge Node local Qdrant instance with unified payload schemas.
5. **Empirical Threshold Optimization**: Evaluates distance distributions across normal activity and simulated/historical anomalies, computing ROC/AUC metrics, precision-recall curves, and F1-optimal operating thresholds.
6. **Production Packaging**: Multi-container Docker Compose architecture, standalone Systemd service units, log rotation policies, and an automated deployment audit tool.

---

## 2. In-Memory Baseline Calibration Pipeline (`scripts/embed_and_index_baseline.py`)

### 2.1 Dahua ASF Filename Parser & Multi-Directory Discovery
The discovery engine inspects target storage paths for Dahua NVR sub-stream video files (`extra1` or `sub`) adhering to standard Dahua naming conventions:
- `NVR_ch01_extra1_20260901080000_20260901090000.asf`
- `ch02_extra1_20260901100000.asf`
- `channel-3_sub1_001.asf`

The primary regular expression parses the camera channel identifier and stream profile:
```python
DAHUA_ASF_PATTERN = re.compile(
    r"(?:NVR_)?ch(?:annel)?[-_]?0*([1-9]\d*)[-_](extra1|main|sub\d*)[-_](\d+)[-_](\d+)",
    re.IGNORECASE,
)
```
- **Multi-Directory & Comma-Separated Paths**: Accepts single directories, comma-separated lists (e.g. `/mnt/e/NVR/2026-9-19,/mnt/e/NVR/2026-9-20`), or parent directories (`/mnt/e/NVR`), recursively traversing file trees with automatic canonical path deduplication.
- **Stream Tier Prioritization**: Sub-stream (`extra1`) video files are prioritized for edge calibration due to their lower decoding overhead, fast frame extraction, and exact temporal fidelity.

### 2.2 In-Memory OpenCV Frame Extraction & Selective Grab Optimization
To prevent wear on local SSDs and conserve local disk space, video files are opened directly from their USB mount points using OpenCV's `cv2.VideoCapture`:
1. **Selective Frame Grab (`cap.grab()`)**: Dahua 1-hour sub-stream files contain ~36,000 frames at 10.0 FPS. Instead of decoding, resizing, and converting all 36,000 frames to RGB, the engine pre-calculates the exact 16 uniform sample indices per 10-second evaluation window (`sample_indices = set(np.linspace(0, frames_per_window - 1, 16, dtype=int))`).
   - Sampled frames execute full `cap.read()` $\rightarrow$ `cv2.resize(224, 224)` $\rightarrow$ `cv2.cvtColor(BGR2RGB)`.
   - Non-sampled frames execute ultra-fast `cap.grab()` (1,230+ FPS), bypassing the software video decoding pixel pipeline entirely.
   - Eliminates over 60% of decoding latency, dropping 1-hour video processing time to ~39 seconds.
2. **Configurable Window Stride (`--window-stride-s`)**: Evaluates 10-second clips every $S$ seconds (default 20.0s). Excess stride frames are rapidly skipped via `cap.grab()`.
3. **Preprocessing**: Converts BGR frames to RGB, resizes to $224 \times 224 \times 3$, and applies standard ImageNet normalization:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
4. **PyTorch MobileNetV3 Backbone (CUDA-Accelerated)**: Passes tensor batches through the feature extractor backbone, producing 576-dimensional embedding vectors in ~33ms per window on GPU.
5. **Unit Normalization**: Normalizes each frame vector:
   $$\mathbf{v} \leftarrow \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$

### 2.3 MiniBatchKMeans Centroid Condensation
Raw video streams generate hundreds of thousands of frame vectors per channel per week. To create a compact, high-speed baseline index:
1. Embeddings for camera channel $c$ are collected into an $N \times 576$ matrix.
2. If $N \le K$ (where $K = 500$ clusters by default), all frame vectors are retained directly.
3. If $N > K$, `sklearn.cluster.MiniBatchKMeans(n_clusters=K, batch_size=100, max_iter=100)` clusters the feature space into $K$ dense centroids.
4. **Critical Hypersphere Projection**: Because geometric cluster centers of unit vectors fall inside the unit hypersphere ($\|\mathbf{c}\|_2 < 1.0$), each centroid is explicitly re-projected to unit length:
   $$\mathbf{c}_k \leftarrow \frac{\mathbf{c}_k}{\|\mathbf{c}_k\|_2}$$

### 2.4 Vector Database Pre-Wipe (`--clear-db` / `clear_vector_databases`)
To guarantee that newly calibrated baselines start from a clean slate without stale or conflicting vectors from earlier runs, the CLI supports an automated pre-ingestion database wipe:
- **Qdrant Cloud**: Calls `client.delete_collection("anomaly_baseline")` and recreates a fresh, empty 576-dimensional collection configured with cosine distance.
- **Qdrant Edge**: Recursively wipes `data/qdrant_edge/` and re-initializes clean `edge_baseline` and `edge_recent_context` collections.

### 2.5 Dual Qdrant Upsert Schema
Centroids are formatted with deterministic point UUIDs (`uuid.uuid5(NAMESPACE, f"ch_{channel}_centroid_{i}")`) and indexed into both Cloud and Edge Qdrant instances:
```json
{
  "id": "a1b2c3d4-e5f6-5a6b-7c8d-9e0f1a2b3c4d",
  "vector": [0.0341, -0.0128, ..., 0.0892],
  "payload": {
    "camera_id": "cam-1",
    "channel": 1,
    "cluster_id": 42,
    "vector_type": "baseline_centroid",
    "is_centroid": true,
    "source_video": "NVR_ch1_extra1_20260918110000_20260918120000.asf",
    "timestamp_ms": 1790008637000
  }
}
```

---

## 3. Empirical Threshold Calibration & ROC/AUC Suite (`scripts/evaluate_thresholds.py`)

### 3.1 Cosine Distance Formulations
The anomaly detection score is the minimum cosine distance between a query vector $\mathbf{q}$ and the set of baseline centroids $\mathcal{C}_k$ for channel $k$:
$$d(\mathbf{q}, \mathcal{C}_k) = \min_{\mathbf{c} \in \mathcal{C}_k} \left( 1 - \frac{\mathbf{q} \cdot \mathbf{c}}{\|\mathbf{q}\|_2 \|\mathbf{c}\|_2} \right)$$
Because both query and centroid vectors are unit normalized ($\|\mathbf{q}\|_2 = \|\mathbf{c}\|_2 = 1$), this simplifies to:
$$d(\mathbf{q}, \mathcal{C}_k) = 1 - \max_{\mathbf{c} \in \mathcal{C}_k} (\mathbf{q} \cdot \mathbf{c})$$

### 3.2 Percentile & Statistical Analysis
The evaluation suite samples baseline vectors from Qdrant and measures the empirical distribution of nearest-neighbor cosine distances:
- **50th Percentile (Median)**: Expected nominal background similarity ($\sim 0.015 - 0.025$).
- **90th Percentile**: Minor lighting shifts and wind-induced motion ($\sim 0.040 - 0.050$).
- **95th Percentile**: High natural variance ($\sim 0.055 - 0.065$).
- **99th Percentile**: Extreme natural anomalies (insects near lens, heavy rain flares).
- **Max Distance**: Upper bound of nominal variance.

### 3.3 ROC Curve, AUC & F1 Optimization
To establish robust thresholds without manual guesswork, the suite performs automated binary classification analysis:
1. **Sensitivity Sweeps**: Sweeps threshold $\tau$ across $[0.01, 0.40]$ at $0.005$ steps.
2. **Metrics Evaluated**:
   $$\text{TPR}(\tau) = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FPR}(\tau) = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
   $$\text{Precision}(\tau) = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall}(\tau) = \text{TPR}(\tau)$$
   $$F_1(\tau) = 2 \cdot \frac{\text{Precision}(\tau) \cdot \text{Recall}(\tau)}{\text{Precision}(\tau) + \text{Recall}(\tau)}$$
3. **Threshold Recommendation Rules**:
   - `EDGE_TRIAGE_THRESHOLD`: Selected at the 95th percentile of baseline distances (default $\approx 0.060$). Filters out 95% of routine ambient fluctuations while forwarding novel movements to the cloud.
   - `CLOUD_ANOMALY_THRESHOLD`: Selected at the optimal $F_1$ score cutoff on high-order anomalies (default $\approx 0.150$). Triggers Pegasus VLM visual reasoning and security notifications.
4. **Direct Environment Application**: When invoked with `--apply-env`, the script atomically updates `.env` with the calibrated thresholds.

---

## 4. Production Packaging & Deployment Stack

### 4.1 Container Topology (`docker-compose.prod.yml`)

The multi-container production stack consists of four coordinated services:

```mermaid
graph TD
    subgraph "Host System (Ubuntu 22.04 / 24.04)"
        USB["Dahua USB Storage / NVR Feeds"]
    end

    subgraph "Docker Bridge Network (surveillance-net)"
        EDGE["surveillance-edge (Port 7777)<br/>MobileNetV3 Edge Engine<br/>RAM Ring Buffer & MJPEG Proxy"]
        BACKEND["surveillance-backend (Port 9876)<br/>FastAPI Orchestration<br/>Twelve Labs Pegasus VLM<br/>Incident Engine"]
        FRONTEND["surveillance-frontend (Port 3000)<br/>Next.js 14 Matrix Console<br/>Direct MJPEG Multi-Feed Grid"]
        QDRANT["qdrant (Port 6333, Optional)<br/>Local High-Speed Vector DB"]
    end

    USB -->|RTSP / Video Streams| EDGE
    EDGE -->|Live Alert Dispatch| BACKEND
    EDGE -->|Vector Baseline Queries| QDRANT
    BACKEND -->|Indexing & Anomaly Search| QDRANT
    FRONTEND -->|Direct HTTP MJPEG Streams| EDGE
    FRONTEND -->|REST API & SSE Telemetry| BACKEND
```

| Service | Base Image | Ports | Healthcheck | Resource Limits |
| :--- | :--- | :--- | :--- | :--- |
| `surveillance-edge` | `python:3.10-slim` | `7777:7777` | `curl -f http://localhost:7777/health` | 2.0 CPUs, 2.0 GB RAM |
| `surveillance-backend` | `python:3.10-slim` | `9876:9876` | `curl -f http://localhost:9876/health` | 2.0 CPUs, 2.0 GB RAM |
| `surveillance-frontend` | Multi-stage `node:18-alpine` | `3000:3000` | `wget --spider http://localhost:3000/` | 1.0 CPUs, 1.0 GB RAM |
| `qdrant` | `qdrant/qdrant:v1.7.4` | `6333:6333` | `/qdrant/tools/healthcheck` | 1.0 CPUs, 1.5 GB RAM |

### 4.2 Systemd Unit Files (`scripts/systemd/`)
For non-containerized bare-metal deployments (e.g. edge micro-servers or Jetson devices), native Systemd units provide automatic restart, resource sandboxing, and log aggregation:
- `surveillance-edge.service`: Starts the Edge FastAPI daemon via `uvicorn edge.main:app --host 0.0.0.0 --port 7777`.
- `surveillance-backend.service`: Starts the Cloud Backend daemon via `uvicorn backend.main:app --host 0.0.0.0 --port 9876`.
- `surveillance-frontend.service`: Serves the optimized Next.js build via `npm run start` on port 3000.
- `surveillance-logrotate.conf`: Daily log rotation policy with 14-day retention and gzip compression for `/var/log/surveillance/*.log`.

### 4.3 Automated Deployment Audit (`scripts/deploy_production.sh`)
The automated deployment script performs end-to-end verification:
1. Validates storage directory permissions (`storage/clips`, `storage/thumbnails`, `storage/cache`).
2. Builds the optimized Next.js 14 static and server bundle (`next build`).
3. Audits `docker-compose.prod.yml` syntax and environment variables.
4. Checks port availability for ports 7777, 9876, and 3000.
5. Displays deployment launch instructions.

---

## 5. End-to-End Integration & Test Suite Results

The Phase 7 verification suite exercises the entire platform across unit, integration, and end-to-end test suites.

### 5.1 Test Breakdown

1. **`tests/test_baseline_indexer_script.py` (8 tests)**:
   - `test_parse_dahua_asf_filename_standard`: Validates canonical sub-stream regex parsing.
   - `test_parse_dahua_asf_filename_mainstream_and_variations`: Tests mainstream and fallback variations.
   - `test_parse_dahua_asf_filename_unsupported`: Verifies invalid filename rejection.
   - `test_discover_dahua_videos_filtering`: Validates filesystem tree traversal for ASF assets.
   - `test_cluster_channel_embeddings_condensation`: Verifies `MiniBatchKMeans` clustering and hypersphere normalization ($\|\mathbf{c}\|_2 \approx 1.0$).
   - `test_index_channel_centroids_dual_upsert`: Verifies dual Qdrant payload assembly.
   - `test_run_baseline_pipeline_dry_run`: Verifies `--dry-run` CLI execution.
   - `test_clear_vector_databases`: Validates pre-ingestion database wiping for both Edge and Cloud.

2. **`tests/test_threshold_evaluation_script.py` (4 tests)**:
   - `test_compute_distance_percentiles`: Verifies percentile calculations (50th, 90th, 95th, 99th, max).
   - `test_compute_roc_auc_and_f1`: Validates ROC sweep, AUC integration, and F1 maximization.
   - `test_recommend_thresholds`: Verifies recommended thresholds fall within operational bounds ($0.03 \le \tau_{\text{edge}} \le 0.12$).
   - `test_apply_thresholds_to_env`: Verifies atomic `.env` file updating.

3. **`tests/test_e2e_phase7_integration.py` (4 tests)**:
   - `test_e2e_baseline_calibration_to_qdrant`: Exercises end-to-end calibration flow from synthetic video to Qdrant vector points.
   - `test_e2e_threshold_calibration_and_recommendation`: Verifies distance evaluation and threshold recommendation against calibrated points.
   - `test_e2e_production_docker_compose_validity`: Audits Docker Compose configuration and required volume definitions.
   - `test_e2e_production_systemd_units_validity`: Validates Systemd unit configuration, service directives, and restart policies.

### 5.2 Test Execution Results
```bash
$ uv run pytest
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-8.4.1, pluggy-1.6.0
rootdir: /opt/home_surveillance_project
configfile: pyproject.toml
testpaths: tests
collected 148 items

tests/test_anomaly.py .........................                          [ 17%]
tests/test_baseline_indexer_script.py ........                           [ 22%]
tests/test_dahua_client.py ............                                  [ 30%]
tests/test_e2e_phase7_integration.py ....                               [ 33%]
tests/test_edge_engine.py ...............                                [ 43%]
tests/test_incident_engine.py .........                                  [ 49%]
tests/test_live_stream_proxy.py .......                                  [ 54%]
tests/test_main_integration.py ........                                  [ 60%]
tests/test_pegasus_integration.py ......                                 [ 64%]
tests/test_pipeline.py ............                                      [ 72%]
tests/test_security_chat.py ......                                       [ 76%]
tests/test_threshold_evaluation_script.py ....                           [ 79%]
tests/test_twelvelabs_client.py ............                             [ 87%]
tests/test_video_buffer.py .......                                       [ 91%]
tests/test_web_ui_api_contract.py .............                          [100%]

============================= 148 passed in 42.98s =============================
```
**Result**: **148 tests passed, 0 failures, 100% pass rate.**

### 5.3 Full Production Calibration Execution & Empirical Matrix Results

The calibration pipeline was executed in live mode across all 203 Dahua ASF sub-stream video files on external USB storage (`E:\NVR\2026-9-19` and `E:\NVR\2026-9-20`, mounted at `/mnt/e/NVR`) with prior database wiping:

```bash
uv run python scripts/embed_and_index_baseline.py \
    --video-dir "/mnt/e/NVR/2026-9-19,/mnt/e/NVR/2026-9-20" \
    --clusters-per-camera 500 \
    --window-stride-s 20.0 \
    --clear-db
```

#### Pre-Ingestion Database Wipe Log
```text
Clearing Pre-Existing Vectors from Vector Databases...
[INFO] embed_and_index_baseline: Deleting pre-existing Qdrant Cloud collection 'anomaly_baseline'...
[INFO] embed_and_index_baseline: Recreating empty Qdrant Cloud collection 'anomaly_baseline' (dim=576, distance=COSINE)...
  ✓ Qdrant Cloud collection 'anomaly_baseline' deleted and recreated cleanly.
[INFO] embed_and_index_baseline: Deleting pre-existing Qdrant Edge database at: data/qdrant_edge
[INFO] edge.detector: Connected to local embedded Qdrant at: data/qdrant_edge
[INFO] edge.detector: Created Qdrant collection: edge_baseline (dim=576)
[INFO] edge.detector: Created Qdrant collection: edge_recent_context (dim=576)
  ✓ Local Qdrant Edge database at 'data/qdrant_edge' wiped and re-initialized cleanly.
```

#### Production Calibration Summary Matrix
```text
======================================================================
  BASELINE EXTRACTION SUMMARY MATRIX
======================================================================
  Channels Processed: 9
  Total Windows:      15,283
  Total Centroids:    4,500
  Total Elapsed Time: 3,500.63s (~58 mins)
----------------------------------------------------------------------
  CH 01 (Studio          ): 1,625 wins ->  500 centroids | Cloud:  500 | Edge: 500
  CH 02 (Dining Room     ): 1,625 wins ->  500 centroids | Cloud: 1000 | Edge: 500
  CH 03 (Kitchen         ): 1,626 wins ->  500 centroids | Cloud: 1500 | Edge: 500
  CH 04 (Backyard        ): 1,806 wins ->  500 centroids | Cloud: 2000 | Edge: 500
  CH 05 (Living Room     ): 1,801 wins ->  500 centroids | Cloud: 2500 | Edge: 500
  CH 06 (Side Alley West ): 1,803 wins ->  500 centroids | Cloud: 3000 | Edge: 500
  CH 07 (Courtyard East  ): 1,802 wins ->  500 centroids | Cloud: 3500 | Edge: 500
  CH 08 (Courtyard West  ): 1,576 wins ->  500 centroids | Cloud: 4000 | Edge: 500
  CH 09 (TV Room         ): 1,619 wins ->  500 centroids | Cloud: 4500 | Edge: 500
======================================================================
```

#### Live Vector Count & Query Verification
- **Qdrant Cloud (`anomaly_baseline`)**: **4,500 points** verified active.
- **Qdrant Edge (`edge_baseline`)**: **4,500 points** (500 per channel) verified active.
- **kNN Anomaly Queries**: Verified on both local edge engine (`d_baseline` computation with `engine='qdrant'`) and cloud backend (`score_clip` with cosine similarity retrieval).

---

## 6. CLI Quickstart & Operational Runbook

### 6.1 Baseline Calibration on USB Storage
To calibrate camera baselines directly from a connected USB drive without copying video files, with prior database wiping:
```bash
# Dry-run test to preview discovery and clustering without modifying databases:
uv run python scripts/embed_and_index_baseline.py \
    --video-dir "/mnt/e/NVR/2026-9-19,/mnt/e/NVR/2026-9-20" \
    --clusters-per-camera 500 \
    --window-stride-s 20.0 \
    --dry-run

# Full live calibration with pre-wipe and dual Qdrant Cloud / Edge indexing:
uv run python scripts/embed_and_index_baseline.py \
    --video-dir "/mnt/e/NVR/2026-9-19,/mnt/e/NVR/2026-9-20" \
    --clusters-per-camera 500 \
    --window-stride-s 20.0 \
    --clear-db
```

### 6.2 Empirical Threshold Calibration
To evaluate baseline distance distributions and update operating thresholds in `.env`:
```bash
# Evaluate distance percentiles, ROC/AUC metrics, and apply optimal thresholds:
uv run python scripts/evaluate_thresholds.py \
    --collection anomaly_baseline \
    --samples 200 \
    --apply-env \
    --json-output storage/threshold_evaluation_report.json
```

### 6.3 Deploying to Production

#### Option A: Docker Compose (Recommended)
```bash
# Audit pre-flight environment:
./scripts/deploy_production.sh

# Launch the production stack:
docker compose -f docker-compose.prod.yml up -d --build

# View operational logs:
docker compose -f docker-compose.prod.yml logs -f
```

#### Option B: Systemd Native Daemons
```bash
# Install systemd service units:
sudo cp scripts/systemd/surveillance-*.service /etc/systemd/system/
sudo cp scripts/systemd/surveillance-logrotate.conf /etc/logrotate.d/surveillance

# Reload systemd and start all services:
sudo systemctl daemon-reload
sudo systemctl enable --now surveillance-edge surveillance-backend surveillance-frontend

# Check service status:
sudo systemctl status surveillance-edge surveillance-backend surveillance-frontend
```

---

## 7. Operational Summary & Verification Sign-Off

| Milestone | Target Specification | Delivered Status |
| :--- | :--- | :--- |
| **Direct USB Sub-stream Ingestion** | In-memory OpenCV window decoding without disk copy | **Verified** (`embed_and_index_baseline.py`) |
| **Feature Extraction & Clustering** | 576-dim MobileNetV3 + `MiniBatchKMeans` (500 centroids/channel) | **Verified** (4,500 hypersphere L2 normalized centroids) |
| **Dual Vector Indexing** | Qdrant Cloud + Local Edge instance synchronization | **Verified** (4,500 points in Cloud & Edge) |
| **Vector DB Pre-Wipe** | Automated collection deletion and re-initialization | **Verified** (`--clear-db` / `clear_vector_databases`) |
| **Empirical Threshold Optimization** | Percentile profiling, ROC/AUC calculation, `.env` sync | **Verified** (`evaluate_thresholds.py`) |
| **Production Packaging** | Docker Compose, Multi-stage Dockerfiles, Systemd, Logrotate | **Verified** (`docker-compose.prod.yml`, `scripts/systemd/`) |
| **End-to-End Test Suite** | Full workspace test suite with 100% pass rate | **Verified** (148/148 tests passing) |
