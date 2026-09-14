# Phase 3: Central Cloud Backend, High-Precision Embedding & Scoring

Comprehensive technical documentation for **Phase 3** of the Home Security Surveillance Analytics Platform. This document explains the architecture, module implementations, key engineering decisions, multi-model ensemble scoring formulas, and validation results for the Central Cloud Analytics Tier.

---

## 1. Executive Summary & Objectives

While **Phase 2** established on-premises edge triage (filtering out ~85–90% of routine video locally on port 7777), **Phase 3** delivers the **Central Cloud Analytics Tier** operating on port 9876. This tier ingests escalated incident clips from edge devices, extracts high-precision deep video embeddings, queries a global multi-camera vector baseline in Qdrant, and executes multi-model ensemble fusion with temporal escalation boosting.

### Key Architectural Objectives Achieved:
1. **High-Precision Video Embeddings with Resilient Fallback**: Integrates Twelve Labs Marengo 3.0/2.7 cloud video embeddings with a standalone local PyTorch Model Server (`model_server.py` on port 9877). If Twelve Labs credentials are not configured or cloud connectivity drops, the system falls back to local PyTorch temporal feature extraction, or directly evaluates the edge embedding.
2. **Central Qdrant Baseline & Spatial Payload Indexing (`backend/anomaly.py`)**: Indexes pre-calibrated normal routine clusters across all 9 camera channels in a central Qdrant cluster. Supports spatial filtering (`camera_id`, `scene_id`) with automatic fallback to global search if a channel is unseeded.
3. **Multi-Model Ensemble Fusion ($70\%$ Cloud + $30\%$ Edge)**: Combines the semantic depth of cloud video models with the on-device edge score:
   $$S_{\text{raw}} = 0.70 \cdot S_{\text{cloud}} + 0.30 \cdot S_{\text{edge}}$$
4. **Sliding-Window Temporal Escalation Boosting**: Applies dynamic boosting ($+0.05$ to $+0.30$) for consecutive escalations originating from the same camera feed within a 5-minute sliding window, accelerating alerts for sustained intruder or vehicle activity.
5. **Agreement Confidence & Adaptive Per-Camera Governance**: Computes confidence based on model agreement ($1.0 - |S_{\text{edge}} - S_{\text{cloud}}|$) and allows per-camera decision threshold overrides (e.g. higher sensitivity for driveway vs lower for high-wind foliage).
6. **Central FastAPI Application (`backend/main.py`, Port 9876)**: Coordinates edge escalations (`POST /api/v1/escalate`), pre-computed embedding evaluation (`POST /api/v1/score`), baseline batch indexing, edge node lifecycle tracking (`backend/edge_registry.py`), and 9-channel Dahua stream endpoint catalogs.
7. **Twelve Labs Semantic Search & Pegasus VLM Q&A**: Exposes natural-language video search (`/api/v1/search`) and structured scene analysis (`/api/v1/analyze`).

---

## 2. System Architecture & Data Flow Diagram

```
+=======================================================================================================================+
|                                              LOCAL EDGE NODE (Port 7777)                                              |
|                                                                                                                       |
|  +---------------------------+   +----------------------------+   +------------------------------------------------+  |
|  | Multi-HTTP Stream Worker  |-->| Sliding Window Segmenter   |-->| Lightweight Edge Embedder                      |  |
|  | (9 Dahua Direct Streams)  |   | (10s clips, 2s overlap)     |   | (PyTorch MobileNetV3 / EfficientNet-B0)        |  |
|  +---------------------------+   +----------------------------+   +------------------------------------------------+  |
|                                                                                          |                            |
|                                                                                          v                            |
|  +---------------------------+   +----------------------------+   +------------------------------------------------+  |
|  | SQLite Offline Queue      |<--| Auto-Drain Worker          |<--| Qdrant Edge Two-Shard Anomaly Scorer           |  |
|  | (WAL Mode, Crash-Safe)    |   | (POST /api/v1/escalate)    |   | (T_edge = 0.060)                               |  |
|  +---------------------------+   +----------------------------+   +------------------------------------------------+  |
+================================================|======================================================================+
                                                 |
                                                 | HTTP POST multipart/form-data (MP4 Clip + Edge Telemetry)
                                                 | OR application/json (Edge Embedding + Channel ID)
                                                 v
+=======================================================================================================================+
|                                        CENTRAL CLOUD ANALYTICS TIER (Port 9876)                                       |
|                                                                                                                       |
|  +-----------------------------------------------------------------------------------------------------------------+  |
|  | FastAPI Central Application (backend/main.py)                                                                   |  |
|  |                                                                                                                 |  |
|  |  +-----------------------------------------------------------------------------------------------------------+  |  |
|  |  | Escalation Handler Pipeline (backend/escalation.py)                                                      |  |  |
|  |  |                                                                                                           |  |  |
|  |  |   [Step 1: Embedding Extraction Fallback Chain]                                                           |  |  |
|  |  |   1. Twelve Labs Marengo (twelvelabs_client.py) -> 2. Model Server (model_server.py) -> 3. Edge Embedding |  |  |
|  |  |                                                                                                           |  |  |
|  |  |   [Step 2: Central Vector Baseline kNN Scoring (backend/anomaly.py)]                                      |  |  |
|  |  |   Cosine distance against top-k normal vectors in Qdrant cluster (filtered by camera_id / scene_id)       |  |  |
|  |  |                                                                                                           |  |  |
|  |  |   [Step 3: Multi-Model Ensemble & Temporal Boost (backend/ensemble.py)]                                   |  |  |
|  |  |   S_ensemble = min(1.0, 0.70 * S_cloud + 0.30 * S_edge + Temporal_Boost)                                  |  |  |
|  |  |                                                                                                           |  |  |
|  |  |   [Step 4: Registry & Metric Tracking (backend/edge_registry.py, EscalationTracker)]                      |  |  |
|  |  |   Updates per-camera confirmation rate, false-positive counts, and assigns incident IDs                   |  |  |
|  |  +-----------------------------------------------------------------------------------------------------------+  |  |
|  |                                                                                                                 |  |
|  |  +-----------------------------+  +-------------------------------+  +---------------------------------------+  |  |
|  |  | Central Qdrant Baseline     |  | Twelve Labs Marengo / Pegasus |  | 9-Channel Camera Catalog              |  |  |
|  |  | (anomaly_baseline index)    |  | (Search & Scene VLM Q&A)      |  | (Direct HTTP Stream URLs)             |  |  |
|  |  +-----------------------------+  +-------------------------------+  +---------------------------------------+  |  |
|  +-----------------------------------------------------------------------------------------------------------------+  |
+=======================================================================================================================+
```

---

## 3. Module-by-Module Technical Breakdown

### 3.1 `backend/config.py` — Centralized Configuration Knobs
- **Purpose**: Defines strongly typed environment configuration knobs for the Central Cloud Analytics Tier using `pydantic-settings` and `.env` parsing.
- **Key Settings & Knobs**:
  - **Central Server**: `central_host` (`CENTRAL_HOST`, default: `"0.0.0.0"`), `central_port` (`CENTRAL_PORT`, default: `9876`).
  - **Local Model Server**: `model_server_url` (`MODEL_SERVER_URL`, default: `"http://localhost:9877"`).
  - **Qdrant Vector Cluster**: `qdrant_url` (`QDRANT_URL`, default: `"http://localhost:6333"`), `qdrant_api_key` (`QDRANT_API_KEY`, default: `None`), `collection_name` (`COLLECTION_NAME`, default: `"anomaly_baseline"`).
  - **Anomaly Scoring & Thresholds**: `cloud_anomaly_threshold` (`CLOUD_ANOMALY_THRESHOLD`, default: `0.15`), `escalation_threshold` (`ESCALATION_THRESHOLD`, default: `0.15`), `confirmation_k` (`CONFIRMATION_K`, default: `5`).
  - **Multi-Model Ensemble Knobs**: `ensemble_cloud_weight` (`ENSEMBLE_CLOUD_WEIGHT`, default: `0.7`), `ensemble_edge_weight` (`ENSEMBLE_EDGE_WEIGHT`, default: `0.3`), `ensemble_threshold` (`ENSEMBLE_THRESHOLD`, default: `0.15`), `temporal_boost_window_min` (`TEMPORAL_BOOST_WINDOW`, default: `5.0`), `temporal_boost_factor` (`TEMPORAL_BOOST_FACTOR`, default: `0.1`), `max_temporal_boost` (`MAX_TEMPORAL_BOOST`, default: `0.3`).
  - **Twelve Labs API Configuration**: `twelve_labs_api_key` (`TWELVE_LABS_API_KEY`), `twelve_labs_api_url` (`TWELVE_LABS_API_URL`), `marengo_index_name` (`TWELVE_LABS_MARENGO_INDEX_NAME`), `pegasus_index_name` (`TWELVE_LABS_PEGASUS_INDEX_NAME`), `marengo_model` (`marengo2.7`), `pegasus_model` (`pegasus1.2`), `twelve_labs_upload_timeout` (`600`), `twelve_labs_max_clips` (`10`).
  - **Multi-Camera Inventory**: `num_cameras` (`DAHUA_NUM_CAMERAS`, default: `9`), `camera_names_raw` (`CAMERA_NAMES`), and `parse_camera_names()` method converting CSV strings to `dict[int, str]`.

### 3.2 `backend/models.py` — Pydantic Data Schemas
- **Purpose**: Establishes strict data schemas for API requests, responses, telemetry, and integration models.
- **Key Schemas**:
  - `HealthResponse`: Cluster status, Qdrant connectivity, model server state, Twelve Labs status, edge devices count, uptime.
  - `NeighborResponse`: Nearest neighbor point metadata (`clip_id`, `similarity`, `source_video`, `scene_id`, `camera_id`).
  - `AnomalyResultResponse`: Anomaly score, binary flag (`is_anomaly`), neighbors list, and latency breakdown.
  - `EmbedRequest`: Direct embedding evaluation payload with optional collection name, $k$, and spatial filters.
  - `BaselineIndexRequest`: Batch baseline vector upsert payload with metadata.
  - `EscalationRequestModel` & `EscalationResponseModel`: Payloads exchanged between edge nodes and cloud during escalation.
  - `EnsembleResultModel`: Fused ensemble score, agreement confidence, and temporal boost details.
  - `EdgeDeviceModel` & `EdgeRegisterRequest`: Edge device registration and analytics status.
  - `CameraStreamInfo`: Channel metadata and direct HTTP streaming URLs.
  - `SearchResultModel` & `AnalysisResultModel`: Twelve Labs video search and Pegasus text generation outputs.

### 3.3 `backend/twelvelabs_client.py` — Twelve Labs Marengo & Pegasus Integration
- **Purpose**: Client wrapper around the Twelve Labs Python SDK providing video clip indexing, high-dimensional embedding extraction, semantic text-to-video search, and VLM incident analysis.
- **Key Components**:
  - `is_enabled() -> bool`: Evaluates if `TWELVE_LABS_API_KEY` is set.
  - `get_client()`: Cached singleton instantiating `TwelveLabs(api_key=...)`.
  - `_ensure_index(index_name, models)`: Finds or creates indexes for Marengo and Pegasus.
  - `upload_video(file_path, index_type)`: Submits MP4 video tasks and polls for completion (`ready` state).
  - `get_video_embedding(video_id)`: Retrieves dense feature vectors from Marengo.
  - `search_videos(query, max_clips, threshold, group_by)`: Performs natural-language semantic video queries across surveillance footage.
  - `analyze_video(video_id, prompt)`: Generates natural-language scene descriptions using Pegasus VLM.

### 3.4 `model_server.py` — Standalone Local Embedding Server
- **Purpose**: Standalone FastAPI service running on port 9877 (`MODEL_SERVER_URL`), providing local PyTorch feature extraction as a drop-in fallback for environments without Twelve Labs API keys.
- **Key Components**:
  - `load_local_model(device_str)`: Loads torchvision `mobilenet_v3_small` with decoupled classifier (`Identity()`) on CUDA/CPU.
  - `extract_frames_from_video(video_path, num_frames=16)`: Uses OpenCV `cv2.VideoCapture` to extract 16 evenly spaced RGB frames.
  - `extract_embedding_from_path(video_path)`: Preprocesses frames via PIL transforms, executes batched forward pass, computes temporal average pooling (`mean(dim=0)`), and normalizes embedding to unit L2 length ($\|\mathbf{v}\|_2 = 1.0$).
  - **Endpoints**:
    - `GET /health`: Model loading state and compute device.
    - `POST /embed`: Multipart MP4 upload returning 576-dim L2-normalized embedding and latency.
    - `POST /batch-embed`: Multi-file batch embedding extraction.

### 3.5 `backend/anomaly.py` — Central Qdrant Baseline & kNN Scorer
- **Purpose**: Evaluates candidate video embeddings against a pre-calibrated normal baseline using cosine kNN distance in Qdrant.
- **Key Components**:
  - `get_qdrant()`: Thread-safe client singleton connecting to remote Qdrant (`QDRANT_URL`).
  - `InMemoryCentralBaselineIndex`: Pure NumPy in-memory fallback implementing cosine kNN search and spatial filtering when Qdrant daemon is offline.
  - `index_baseline_vectors(vectors, metadata_list, collection_name)`: Creates collection with cosine distance configuration and batch upserts points with payload.
  - `score_clip(embedding, collection_name, k, threshold, scene_id, camera_id)`:
    - Queries Qdrant with `FieldCondition` filtering on `camera_id` and `scene_id`.
    - Automatically falls back to unfiltered search if the channel-specific filter yields 0 matches.
    - Formula:
      $$S_{\text{cloud}} = 1.0 - \frac{1}{k} \sum_{i=1}^k \text{sim}(\mathbf{q}, \mathbf{b}_i)$$
    - Returns `AnomalyResult` containing anomaly score, boolean verdict (`is_anomaly`), neighbors, and search latency.
  - `get_collection_stats()`: Reports total indexed baseline points and storage backend.

### 3.6 `backend/edge_registry.py` — Device & Camera Channel Registry
- **Purpose**: Tracks connected edge nodes and maintains the master 9-channel Dahua camera inventory.
- **Key Components**:
  - `EdgeDeviceRecord`: Dataclass storing `device_id`, `name`, `location`, `status`, `baseline_version`, `threshold`, `escalation_count`, `confirmed_count`, and `last_seen`.
  - `EdgeRegistry`:
    - Automatically initializes the 9 Dahua camera records (`cam-1` to `cam-9`) from configuration.
    - Methods: `register()`, `get()`, `list_all()`, `update_heartbeat()`, `set_baseline_version()`, `set_threshold()`.
    - Computes confirmation rates and false-positive metrics per channel.

### 3.7 `backend/ensemble.py` — Multi-Model Ensemble Scorer
- **Purpose**: Fuses cloud and edge scores with agreement confidence and temporal escalation boosting.
- **Key Components**:
  - `EnsembleResult`: Dataclass containing `edge_score`, `cloud_score`, `ensemble_score`, `confidence`, `is_anomaly`, `temporal_boost`, and `weights_used`.
  - `EnsembleScorer`:
    - **Weighted Score Fusion**:
      $$S_{\text{raw}} = w_{\text{cloud}} \cdot S_{\text{cloud}} + w_{\text{edge}} \cdot S_{\text{edge}} \quad (w_{\text{cloud}}=0.7, w_{\text{edge}}=0.3)$$
    - **Temporal Escalation Boost**:
      $$\text{Boost} = \min\left(\text{max\_boost}, (N_{\text{recent}} - 1) \cdot \text{factor}\right)$$
      Tracks escalations per device in a 5-minute sliding window. The first escalation receives $+0.00$ boost; subsequent escalations add $+0.05$ up to $+0.30$.
    - **Final Ensemble Score**:
      $$S_{\text{ensemble}} = \min(1.0, S_{\text{raw}} + \text{Boost})$$
    - **Agreement Confidence**:
      $$\text{Confidence} = \max(0.0, \min(1.0, 1.0 - |S_{\text{edge}} - S_{\text{cloud}}|))$$
    - **Adaptive Per-Device Thresholds**: Supports custom sensitivity thresholds per camera (`set_device_threshold()`).

### 3.8 `backend/escalation.py` — Cloud Escalation Pipeline & Tracker
- **Purpose**: Ingests escalated incidents from edge nodes, orchestrates cloud re-embedding, runs baseline scoring, computes ensemble fusion, and records statistics.
- **Key Components**:
  - `EscalationRequest` & `EscalationResult`: Dataclasses encapsulating input video evidence and output evaluation.
  - `handle_escalation(request)`:
    1. Re-embeds video clip: attempts Twelve Labs Marengo $\to$ falls back to local Model Server $\to$ falls back to edge embedding.
    2. Queries Central Qdrant baseline via `score_clip()`.
    3. Evaluates ensemble score via `ensemble_scorer.score()`.
    4. Confirms anomaly and generates incident identifier (`inc-xxxx`).
    5. Updates edge device metrics in `edge_registry`.
  - `EscalationTracker`: Thread-safe telemetry tracker recording global escalations, confirmation rates, average cloud scores, edge accuracy, and per-device breakdown.

### 3.9 `backend/main.py` — Central FastAPI Service (Port 9876)
- **Purpose**: Main API server exposing REST endpoints for edge devices, admin tools, and the web console.
- **Endpoints**:
  - `GET /health`: Health matrix for Qdrant, Model Server, Twelve Labs, and edge nodes.
  - `POST /api/v1/escalate` / `POST /api/escalate`: Receives edge escalations (supports multipart form-data for MP4 video or JSON payloads).
  - `POST /api/v1/score` / `POST /api/score`: Scores pre-computed embedding vectors.
  - `POST /api/v1/baseline/index`: Batch indexes baseline calibration embeddings.
  - `GET /api/v1/qdrant/stats`: Returns baseline collection point counts and status.
  - `GET /api/v1/escalations/stats`: Global escalation telemetry and confirmation rate.
  - `GET /api/v1/edges`, `POST /api/v1/edges/register`, `GET /api/v1/edges/{id}/stats`: Edge node lifecycle and performance metrics.
  - `GET /api/v1/cameras`: 9-channel Dahua camera inventory with direct HTTP stream URLs.
  - `POST /api/v1/search` & `POST /api/v1/analyze`: Twelve Labs Marengo search and Pegasus video Q&A.
  - `GET /api/v1/twelvelabs/status`: Twelve Labs configuration status.

---

## 4. Key Design Decisions & Engineering Rationale

| Design Decision | Alternative Considered | Engineering Rationale |
| :--- | :--- | :--- |
| **70/30 Multi-Model Ensemble Fusion** | 100% Cloud-only decision | Edge models provide fast local context from continuous 10s windowing, while cloud models provide high-dimensional semantic depth. Fusing $70\%$ cloud and $30\%$ edge leverages the strengths of both tiers while preventing single-model bias. |
| **Sliding-Window Temporal Escalation Boosting** | Static fixed thresholds for all frames | Single isolated anomaly frames may result from transient lighting or wind. However, repeated escalations from the same camera within a 5-minute window strongly indicate persistent activity (e.g. intruder walking across the yard). Temporal boosting dynamically elevates score to trigger confirmed alerts faster. |
| **Three-Tier Embedding Fallback Chain** | Mandatory Twelve Labs API dependency | External cloud APIs may experience rate limits, latency spikes, or network outages. Fallback chain (Twelve Labs $\to$ Local PyTorch Model Server $\to$ Edge Embedding) guarantees zero service disruption and full offline operability. |
| **Direct HTTP Stream URLs in Camera Catalog** | Transcoding server (FFmpeg / HLS) | Central backend catalogs the direct HTTP MJPEG stream URLs (`/api/v1/stream/{ch}/live`) generated by the Edge Node, allowing web browsers to render 9 live streams via native `<img>` tags with $<500$ms latency and zero central CPU transcoding load. |
| **Dual Payload Support (Multipart vs JSON)** | Enforcing multipart MP4 uploads only | Low-bandwidth edge deployments or battery-powered auxiliary nodes can escalate by transmitting JSON-encoded feature embeddings without streaming large video files, preserving bandwidth during constrained conditions. |
| **Pure NumPy In-Memory Baseline Fallback** | Hard crash when Qdrant cluster is unreachable | Embedded in-memory index ensures unit tests, CI pipelines, and standalone edge-cloud nodes can execute kNN scoring without requiring an external Qdrant daemon. |

---

## 5. Test Results & Validation Summary

### 5.1 Full Test Suite Execution
Executed via `uv run pytest -v`:

**Results: 60 Passed / 60 Total (100% Success Rate) in 31.23s**

```
tests/test_backend_anomaly.py::test_in_memory_central_baseline_index PASSED      [  2%]
tests/test_backend_anomaly.py::test_score_clip_normal_and_anomalous PASSED      [  4%]
tests/test_backend_anomaly.py::test_collection_stats PASSED                     [  5%]
tests/test_backend_api.py::test_health_endpoint PASSED                          [  7%]
tests/test_backend_api.py::test_cameras_catalog PASSED                          [  8%]
tests/test_backend_api.py::test_baseline_index_and_score PASSED                 [ 10%]
tests/test_backend_api.py::test_qdrant_stats PASSED                             [ 12%]
tests/test_backend_api.py::test_escalate_json_payload PASSED                    [ 14%]
tests/test_backend_api.py::test_escalate_form_payload PASSED                    [ 15%]
tests/test_backend_api.py::test_escalations_stats_endpoint PASSED               [ 17%]
tests/test_backend_api.py::test_edges_endpoints PASSED                          [ 19%]
tests/test_backend_api.py::test_twelvelabs_status_endpoint PASSED               [ 20%]
tests/test_backend_ensemble.py::test_ensemble_raw_fusion PASSED                 [ 22%]
tests/test_backend_ensemble.py::test_ensemble_temporal_boosting PASSED          [ 23%]
tests/test_backend_ensemble.py::test_ensemble_adaptive_device_threshold PASSED  [ 25%]
tests/test_backend_escalation.py::test_handle_escalation_confirmed_and_rejected PASSED [ 27%]
tests/test_backend_escalation.py::test_escalation_tracker_metrics PASSED        [ 28%]
tests/test_capture_manager.py::test_capture_manager_init PASSED                 [ 30%]
tests/test_capture_manager.py::test_capture_manager_all_stats PASSED             [ 32%]
tests/test_capture_manager.py::test_capture_manager_fallback_clip_generation PASSED [ 33%]
tests/test_config.py::test_edge_config_defaults PASSED                           [ 35%]
tests/test_config.py::test_camera_names_parsing PASSED                           [ 37%]
tests/test_config.py::test_url_generation PASSED                                 [ 38%]
tests/test_detector.py::test_in_memory_two_shard_index PASSED                    [ 40%]
tests/test_detector.py::test_detector_scoring_and_triage PASSED                  [ 42%]
tests/test_detector.py::test_detector_recent_shard_fifo_pruning PASSED           [ 43%]
tests/test_detector.py::test_detector_channel_isolation PASSED                   [ 45%]
tests/test_edge_api.py::test_health_endpoint PASSED                             [ 47%]
tests/test_edge_api.py::test_cameras_endpoints PASSED                           [ 48%]
tests/test_edge_api.py::test_scores_endpoint PASSED                              [ 50%]
tests/test_edge_api.py::test_queue_endpoint PASSED                               [ 52%]
tests/test_edge_api.py::test_snapshot_endpoint PASSED                            [ 53%]
tests/test_edge_api.py::test_seed_baseline_endpoint PASSED                      [ 55%]
tests/test_edge_api.py::test_trigger_triage_endpoint PASSED                      [ 57%]
tests/test_model.py::test_feature_extractor_mobilenet_v3_default PASSED          [ 58%]
tests/test_model.py::test_feature_extractor_efficientnet_output PASSED           [ 60%]
tests/test_model.py::test_feature_extractor_custom_dimension_projection PASSED  [ 62%]
tests/test_model.py::test_feature_extractor_rejects_synthetic_and_unsupported_models PASSED [ 63%]
tests/test_model.py::test_feature_extractor_deterministic_consistency PASSED    [ 65%]
tests/test_model.py::test_feature_extractor_sensitivity_to_changes PASSED       [ 67%]
tests/test_model.py::test_feature_extractor_extract_from_clip PASSED             [ 68%]
tests/test_model.py::test_feature_extractor_batch PASSED                        [ 70%]
tests/test_model_server.py::test_extract_embedding_from_path PASSED             [ 72%]
tests/test_model_server.py::test_model_server_api_endpoints PASSED              [ 73%]
tests/test_queue.py::test_sqlite_queue_enqueue_and_peek PASSED                  [ 75%]
tests/test_queue.py::test_sqlite_queue_state_transitions PASSED                  [ 77%]
tests/test_queue.py::test_sqlite_queue_retry_and_fail PASSED                    [ 78%]
tests/test_queue.py::test_sqlite_queue_crash_resilience PASSED                  [ 80%]
tests/test_queue.py::test_auto_drain_worker_sync PASSED                          [ 82%]
tests/test_segmenter.py::test_segmenter_extract_clip PASSED                     [ 83%]
tests/test_segmenter.py::test_segmenter_pacing PASSED                           [ 85%]
tests/test_simulator.py::test_feed_simulator_frame_generation PASSED           [ 87%]
tests/test_simulator.py::test_feed_simulator_jpeg_encoding PASSED              [ 88%]
tests/test_stream_worker.py::test_stream_stats_tick PASSED                     [ 90%]
tests/test_stream_worker.py::test_stream_worker_ring_buffer PASSED             [ 92%]
tests/test_stream_worker.py::test_stream_worker_lifecycle PASSED               [ 93%]
tests/test_twelvelabs_client.py::test_twelvelabs_enabled_check PASSED           [ 95%]
tests/test_twelvelabs_client.py::test_get_client_raises_when_disabled PASSED    [ 97%]
tests/test_twelvelabs_client.py::test_search_videos_mocked PASSED               [ 98%]
tests/test_twelvelabs_client.py::test_analyze_video_mocked PASSED               [100%]

======================= 60 passed, 11 warnings in 31.23s =======================
```

### 5.2 Real-World Significance of Test Results

1. **Central Qdrant Baseline & Scoring (`test_backend_anomaly.py`)**:
   - Confirms that clips matching pre-indexed normal baseline routines receive an anomaly score of $0.0$, preventing false alarms.
   - Confirms that orthogonal anomalous clips score $1.0$, accurately crossing the $0.15$ anomaly threshold.
   - Validates that spatial filtering by `camera_id` and `scene_id` correctly isolates baseline distributions between different camera locations.
2. **Multi-Model Ensemble & Temporal Boost (`test_backend_ensemble.py`)**:
   - Verifies mathematical precision of the $70/30$ cloud/edge score fusion and agreement confidence calculation.
   - Confirms that repeated escalations from the same camera feed receive sequential temporal boosts ($+0.05 \to +0.10 \to \dots \to +0.30$), enabling faster alert confirmation for persistent anomalies.
   - Validates independent device history: an escalation on Camera 2 does not artificially boost an unrelated event on Camera 1.
   - Verifies per-device adaptive thresholding, allowing custom sensitivity tuning per zone.
3. **Escalation Receiver & Metric Tracking (`test_backend_escalation.py`)**:
   - Confirms that false escalations from the edge are rejected by the cloud baseline with `is_confirmed_anomaly=False` and no incident ID created.
   - Confirms that genuine anomalies are confirmed with `is_confirmed_anomaly=True` and a unique incident tracking ID (`inc-xxxx`).
   - Validates `EscalationTracker` metrics, including confirmation rate calculations and per-device false-positive tracking.
4. **Standalone Local Model Server (`test_model_server.py`)**:
   - Validates that MP4 video files uploaded to `/embed` return 576-dimensional L2-normalized feature vectors ($\|\mathbf{v}\|_2 = 1.0$) with latency measurements.
   - Validates `/batch-embed` handling of multiple video clips simultaneously.
5. **Twelve Labs Client Wrapper (`test_twelvelabs_client.py`)**:
   - Confirms proper enablement checks and error handling when API keys are absent.
   - Validates parsing of Twelve Labs Marengo search query responses and Pegasus text generation outputs.
6. **Central FastAPI REST API (`test_backend_api.py`)**:
   - Confirms that all 9 Dahua surveillance camera feeds are cataloged with appropriate names and direct HTTP stream URLs.
   - Validates end-to-end escalation processing via both JSON and multipart form-data requests.
