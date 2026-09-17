# Phase 4: Incident Formation, Natural Language VLM Scene Understanding & Governance

Comprehensive technical documentation for **Phase 4** of the Home Security Surveillance Analytics Platform. This document explains the architecture, algorithmic implementations, mathematical formulations, anti-poisoning baseline memory governance, adaptive backpressure tiers, and validation results for the Incident Formation & VLM Scene Understanding Tier.

---

## 1. Executive Summary & Objectives

While **Phase 3** established the Central Cloud Analytics Tier (coordinating cloud re-embedding, central Qdrant baseline scoring, and $70/30$ ensemble fusion), raw anomaly scores alone do not provide actionable security intelligence. A momentary compression glitch or sudden cloud shadow could create noisy false spikes, while multiple consecutive escalations within seconds could overwhelm operators with redundant alert fragments.

**Phase 4** transforms raw anomaly scores into **actionable, human-interpretable incident intelligence** while providing rigorous **anti-poisoning vector baseline governance** and **adaptive streaming backpressure**.

```
                                      +-------------------------------------------------------+
                                      |              Central FastAPI (Port 9876)              |
                                      +-------------------------------------------------------+
                                                                  |
                                          Escalated Clip / Anomaly Score / Telemetry
                                                                  v
  +=======================================================================================================================+
  |                                              PHASE 4 PROCESSING PIPELINE                                              |
  |                                                                                                                       |
  |  +-----------------------------------------------------------------------------------------------------------------+  |
  |  | 1. Adaptive Streaming Backpressure & Load Shedding (backend/streaming.py)                                       |  |
  |  |    - Level 0: NORMAL (Full cloud re-embedding + Pegasus VLM + Qdrant baseline)                                  |  |
  |  |    - Level 1: SCORE_ONLY (Bypasses heavy VLM analysis during moderate load)                                     |  |
  |  |    - Level 2: PASSTHROUGH (Bypasses cloud kNN search; uses edge score directly)                                 |  |
  |  |    - Level 3: SHED_LOAD (Drops non-critical alerts S_edge < 0.25 to protect system)                             |  |
  |  +-----------------------------------------------------------------------------------------------------------------+  |
  |                                                         |                                                             |
  |                                                         v                                                             |
  |  +-----------------------------------------------------------------------------------------------------------------+  |
  |  | 2. Incident Formation Engine (backend/incidents.py)                                                             |  |
  |  |    - EMA Score Smoothing: S_ema^(t) = α * S^(t) + (1 - α) * S_ema^(t-1)  (α = 0.3)                               |  |
  |  |    - Dual-Threshold Hysteresis: Open @ S_ema >= 0.080, Close @ S_ema < 0.050                                    |  |
  |  |    - Cooldown Merging: Merges recurring triggers within 30s into existing incident                              |  |
  |  |    - Severity Normalization (0-100): Combines peak score (60%), mean score (25%), & duration/count (15%)         |  |
  |  +-----------------------------------------------------------------------------------------------------------------+  |
  |                             |                                                                   |                     |
  |                             v                                                                   v                     |
  |  +-------------------------------------------------+   +-----------------------------------------------------------+  |
  |  | 3. VLM Scene Explainer (backend/vlm_explainer.py)   |   | 4. Memory Governor & Anti-Poisoning (backend/memory.py)   |  |
  |  |    - Twelve Labs Pegasus 1.2 / 1.5 Video Analysis   |   |    - 1-Hour Qdrant Staging Buffer (anomaly_quarantine)    |  |
  |  |    - Structured Breakdown: Actor, Action, Risk, Rec |   |    - Anti-Poisoning Filter: Rejects S_anom >= 0.080       |  |
  |  |    - Strict Pegasus VLM (No Heuristic Fallback)     |   |    - 7-Day Retention Scrubbing of historical logs         |  |
  |  |    - Asynchronous Execution & Explicit Errors       |   |    - Per-Camera Vector Hard Caps (500 vectors/cam)        |  |
  |  +-------------------------------------------------+   +-----------------------------------------------------------+  |
  +=======================================================================================================================+
```

### Key Architectural Objectives Achieved:
1. **Exponential Moving Average (EMA) Anomaly Smoothing**: Attenuates transient sensor glitches and momentary artifact spikes ($\alpha = 0.30$) while tracking sustained anomalous motion trends.
2. **Dual-Threshold Hysteresis State Machine ($T_{\text{start}}=0.080, T_{\text{end}}=0.050$)**: Prevents rapid chattering (oscillation between alert and normal states) during border-case surveillance events.
3. **Sliding Cooldown Window Event Consolidation ($30\text{s}$ window)**: Merges closely spaced escalations on the same camera channel into a single coherent incident record, aggregating duration, peak score, mean score, and discrete event milestones.
4. **Normalized 0–100 Severity Rating & Danger Badges**: Formulates a weighted composite severity rating mapping into four discrete operator tiers: `LOW` (0–24), `MODERATE` (25–49), `HIGH` (50–74), and `CRITICAL` (75–100).
5. **Exclusive Twelve Labs Pegasus Natural-Language Scene Explainer**: Extracts structured security intelligence (actors, specific actions, objects visible, risk assessment, and recommended operator actions) strictly powered by Twelve Labs Pegasus multimodal models (`pegasus-1.2` / `pegasus-1.5`). Handcrafted heuristic fallbacks have been securely removed in favor of explicit error handling and genuine visual evidence grounding.
6. **Anti-Poisoning Vector Memory Governor with Qdrant Staging**: Stages newly proposed normal baseline candidate vectors into a dedicated, persistent 1-hour quarantine collection in Qdrant (`anomaly_quarantine`), validates that anomaly distance remains below poisoning thresholds ($S_{\text{anom}} < 0.080$), enforces per-camera hard vector caps ($500\text{ vectors/camera}$), and executes 7-day retention scrubbing.
7. **Adaptive 4-Tier Streaming Backpressure (`NORMAL`, `SCORE_ONLY`, `PASSTHROUGH`, `SHED_LOAD`)**: Automatically sheds non-critical workload under latency surges ($>500\text{ms}, >1500\text{ms}$) or request queue depth backpressure.

---

## 2. Mathematical Formulations & Algorithms

### 2.1 Exponential Moving Average (EMA) Score Smoothing
For a sequence of incoming ensemble anomaly scores $S^{(t)}$ on camera channel $c$:
$$S_{\text{ema}}^{(t)} = \alpha \cdot S^{(t)} + (1 - \alpha) \cdot S_{\text{ema}}^{(t-1)}$$
Where:
- $\alpha = 0.30$ (configurable via `EMA_ALPHA`).
- Initial condition: $S_{\text{ema}}^{(0)} = S^{(0)}$.

### 2.2 Dual-Threshold Hysteresis State Machine
- **State Transition to `OPEN`**:
  $$\text{State}^{(t)} = \text{OPEN} \iff S_{\text{ema}}^{(t)} \ge T_{\text{start}} \quad (T_{\text{start}} = 0.080)$$
- **State Transition to `CLOSED`**:
  $$\text{State}^{(t)} = \text{CLOSED} \iff S_{\text{ema}}^{(t)} < T_{\text{end}} \quad (T_{\text{end}} = 0.050)$$
- **State Preservation (`Hysteresis Band`)**:
  $$\text{State}^{(t)} = \text{State}^{(t-1)} \iff T_{\text{end}} \le S_{\text{ema}}^{(t)} < T_{\text{start}}$$

### 2.3 Cooldown Merging Window
If a new escalation event $E^{(t)}$ arrives at timestamp $t_{\text{event}}$ on channel $c$:
$$\text{Merge into } \text{Incident } I \iff \left(t_{\text{event}} - t_{\text{last\_event}}\right) \le T_{\text{cooldown}} \quad (T_{\text{cooldown}} = 30.0\text{s})$$

### 2.4 Normalized Composite Severity Formula (0–100)
$$\text{Severity} = \text{clamp}\left(\left(0.60 \cdot S_{\text{peak}} + 0.25 \cdot S_{\text{mean}} + 0.15 \cdot \min\left(1.0, \frac{N_{\text{events}}}{5}\right)\right) \times 100, 0, 100\right)$$

| Severity Range | Danger Badge | Description | Trigger Example |
| :--- | :--- | :--- | :--- |
| **0 – 24** | `LOW` | Minor motion or faint baseline divergence | Rustling foliage, small animal, distant shadows |
| **25 – 49** | `MODERATE` | Noticeable human or vehicular activity in routine area | Delivery driver dropping parcel, car turning in driveway |
| **50 – 74** | `HIGH` | Sustained abnormal movement or unauthorized presence | Person lingering near front gate, off-hours motion in garage |
| **75 – 100** | `CRITICAL` | High-confidence intruder or perimeter breach | Masked subject near patio window, multi-camera burst |

---

## 3. Module-by-Module Technical Breakdown

### 3.1 `backend/incidents.py` — Incident Formation Engine
- **Purpose**: Consolidates discrete anomaly detections into persistent, compound incident objects with lifecycle state management.
- **Key Classes & Functions**:
  - `compute_severity(peak_score, mean_score, event_count) -> (int, str)`: Formulates 0–100 integer severity and badge.
  - `IncidentFormationEngine`:
    - `update_ema(camera_key, raw_score) -> float`: Thread-safe EMA smoothing.
    - `process_escalation(...) -> Optional[IncidentRecord]`: Evaluates hysteresis, manages cooldown merging, and instantiates or updates incident records.
    - `get_incident(incident_id) -> Optional[IncidentRecord]`: Fast dictionary lookup.
    - `list_incidents(channel, status, min_severity, limit, offset) -> List[IncidentRecord]`: Multi-criteria querying.
    - `update_status(incident_id, status, notes) -> Optional[IncidentRecord]`: Transitions lifecycle between `OPEN`, `ACKNOWLEDGED`, `CLOSED`, and `ARCHIVED`.
    - `attach_vlm_explanation(incident_id, explanation) -> Optional[IncidentRecord]`: Links structured Pegasus VLM breakdown.

### 3.2 `backend/vlm_explainer.py` — Natural Language VLM Scene Explainer
- **Purpose**: Generates human-interpretable natural language security breakdowns from incident video clips strictly via Twelve Labs Pegasus VLM.
- **Key Classes & Functions**:
  - `VLM_ANALYSIS_PROMPT`: Structured prompt directing Twelve Labs Pegasus to return strict JSON schema.
  - `_parse_vlm_text_response(raw_text) -> dict`: Robust JSON extractor supporting Markdown code fences, raw JSON regex matching, and fallback plain-text encapsulation.
  - `explain_incident(incident, video_id, clip_path) -> VLMExplanation`: Asynchronous pipeline interfacing exclusively with Pegasus video upload and analysis. If Twelve Labs is disabled or unconfigured, it raises `RuntimeError`. If neither `video_id` nor `clip_path` is provided, it raises `ValueError`. Synthetic heuristic fallbacks are completely eliminated.

### 3.3 `backend/memory.py` — Memory Governor & Anti-Poisoning Engine
- **Purpose**: Protects the central Qdrant baseline against concept drift and malicious or corrupted vector poisoning, persisting candidate quarantine points to a dedicated Qdrant staging collection (`anomaly_quarantine`).
- **Key Classes & Functions**:
  - `MemoryGovernor`:
    - `validate_candidate(vector, camera_id, anomaly_score) -> (bool, str)`: Validates vector dimensions, non-zero norm, NaN checks, and asserts $S_{\text{anom}} < \text{anti\_poisoning\_threshold}$ ($0.080$).
    - `stage_to_quarantine(vector, camera_id, channel, anomaly_score, metadata) -> (Optional[QuarantineItemModel], str)`: Persists candidate vectors and metadata payloads directly into Qdrant staging collection (`QUARANTINE_COLLECTION_NAME = "anomaly_quarantine"`), maintaining in-memory cache sync.
    - `list_quarantine(camera_id, status) -> List[QuarantineItemModel]`: Queries and scrolls quarantine items directly from Qdrant with optional filters.
    - `promote_quarantined_vectors(vector_ids, camera_id, force) -> int`: Promotes matured or manually approved vectors from the staging collection into the active baseline collection (`COLLECTION_NAME = "anomaly_baseline"`) and updates staging payload status to `PROMOTED`.
    - `scrub_retention(retention_days) -> dict`: Purges expired quarantine entries older than the retention window (`BASELINE_RETENTION_DAYS = 7`) from Qdrant staging.
    - `get_stats() -> MemoryStatsResponse`: Telemetry for quarantine point counts, per-camera point counts, and hard capacity utilization.

### 3.4 `backend/streaming.py` — Adaptive Streaming Backpressure & Load Shedding
- **Purpose**: Protects the central analytics backend during extreme traffic spikes (e.g. multi-camera storm triggers) through 4 graceful degradation tiers.
- **Tiers & Levels**:
  - `Level 0 (NORMAL)`: Full deep pipeline (Twelve Labs Marengo re-embedding, Pegasus VLM analysis, Qdrant kNN search).
  - `Level 1 (SCORE_ONLY)`: Latency $> 500\text{ms}$ or queue $\ge 10$. Skips VLM inference; executes Qdrant baseline scoring directly.
  - `Level 2 (PASSTHROUGH)`: Latency $> 1500\text{ms}$ or queue $\ge 25$. Skips cloud Qdrant kNN search; uses edge anomaly score directly ($S_{\text{cloud}} = S_{\text{edge}}$).
  - `Level 3 (SHED_LOAD)`: Queue $\ge 50$ or latency $\ge 3000\text{ms}$. Drops sub-critical escalations ($S_{\text{edge}} < 0.25$); only processes high-priority alarms.

### 3.5 `backend/models.py` & `backend/config.py` — Schemas and Knobs
- **New Pydantic Schemas**: `IncidentRecord`, `IncidentEvent`, `IncidentStatusUpdateRequest`, `VLMExplanation`, `QuarantineItemModel`, `QuarantinePromotionRequest`, `MemoryStatsResponse`, `LoadSheddingStatus`, `LoadSheddingLevelRequest`.
- **Configuration Knobs**:
  - `EMA_ALPHA` (`0.30`)
  - `INCIDENT_START_THRESHOLD` (`0.080`)
  - `INCIDENT_END_THRESHOLD` (`0.050`)
  - `COOLDOWN_WINDOW_S` (`30.0`)
  - `QUARANTINE_COLLECTION_NAME` (`anomaly_quarantine`)
  - `QUARANTINE_DURATION_S` (`3600`)
  - `MAX_VECTORS_PER_CAMERA` (`500`)
  - `BASELINE_RETENTION_DAYS` (`7`)
  - `ANTI_POISONING_THRESHOLD` (`0.080`)
  - `LOAD_SHED_AUTO` (`true`)
  - `LOAD_SHED_SCORE_ONLY_LATENCY_MS` (`500.0`)
  - `LOAD_SHED_PASSTHROUGH_LATENCY_MS` (`1500.0`)
  - `LOAD_SHED_CRITICAL_THRESHOLD` (`0.25`)

---

## 4. Key Design Decisions & Trade-offs

| Design Decision | Alternative Considered | Engineering Rationale |
| :--- | :--- | :--- |
| **EMA Score Smoothing ($\alpha=0.3$)** | Raw frame-by-frame scoring | Camera compression artifacts, bitrate drops, and brief lighting changes cause single-frame score spikes. EMA filters high-frequency noise while reacting to true anomalous motion within 2–3 frames. |
| **Dual-Threshold Hysteresis ($0.080 / 0.050$)** | Single fixed threshold ($0.080$) | Single thresholds cause "chattering" (rapidly opening and closing multiple 1-second incident alerts when motion hovers near the threshold). Hysteresis ensures smooth, clean incident boundaries. |
| **30s Cooldown Merging Window** | Independent discrete incidents | Intruders or delivery drivers frequently pause or step in/out of camera view. Merging events within 30s preserves event context and prevents spamming operators with 10 separate 5-second alarms. |
| **Exclusive Pegasus VLM (Zero Heuristic Fallback)** | Rule-based heuristic string templates | Heuristic text generation produces repetitive, non-grounded assumptions based solely on camera names. Enforcing exclusive Twelve Labs Pegasus multimodal video analysis ensures that operator security explanations are strictly grounded in actual visual evidence. Unconfigured or failing states raise explicit telemetry/HTTP errors rather than silent synthetic degradation. |
| **Persistent Qdrant Quarantine Staging (`anomaly_quarantine`)** | Pure in-memory Python dictionary | Pure RAM dictionaries lose all pending candidate vectors during process restarts or container upgrades. Leveraging a dedicated Qdrant staging collection reuses existing vector DB infrastructure without introducing new external services, ensures crash resilience across the 1-hour observation window, and bounds application RAM usage. |
| **4-Tier Adaptive Load Shedding** | Hard queue reject (HTTP 429) | Dropping requests indiscriminately under load causes blind spots. Tiered degradation (first dropping heavy VLM, then cloud kNN, and only dropping sub-critical low scores under severe load) guarantees zero missed high-priority alarms. |

---

## 5. Verification & Test Results

The Phase 4 test suite was executed using `pytest` across all new and existing modules. **100% of tests passed successfully.**

```bash
uv run pytest tests/test_backend_incidents.py tests/test_vlm_explainer.py tests/test_memory_governor.py tests/test_streaming_backpressure.py tests/test_backend_api.py -v
```

### Test Suite Execution Summary:
- `tests/test_backend_incidents.py`:
  - `test_compute_severity_tiers`: Validates exact mathematical bounds for `LOW`, `MODERATE`, `HIGH`, and `CRITICAL` badges.
  - `test_ema_smoothing`: Verifies recursive EMA smoothing calculation and per-channel reset.
  - `test_incident_creation_and_hysteresis`: Verifies that sub-threshold scores produce `None`, while scores $\ge 0.080$ transition state to `OPEN`.
  - `test_cooldown_window_merging`: Validates that events occurring within 30s merge into the existing incident, updating duration, peak score, and event count.
  - `test_incident_query_and_status_management`: Verifies status transitions (`ACKNOWLEDGED`, `CLOSED`) and VLM explanation attachment.
- `tests/test_vlm_explainer.py`:
  - `test_parse_vlm_text_clean_json`, `markdown_json_fences`, `raw_fallback`: Validates multi-pattern JSON parsing resilience.
  - `test_explain_incident_raises_when_disabled`: Verifies explicit `RuntimeError` raised when Twelve Labs Pegasus is disabled.
  - `test_explain_incident_raises_without_video_or_clip`: Verifies explicit `ValueError` raised when neither `video_id` nor `clip_path` is provided.
  - `test_explain_incident_pegasus_mocked`: Verifies Twelve Labs Pegasus LLM integration and schema mapping with indexed `video_id`.
  - `test_explain_incident_pegasus_with_clip_upload`: Verifies automated clip indexing to Pegasus index and subsequent prompt analysis.
  - `test_explain_incident_pegasus_api_failure`: Verifies exception propagation on Twelve Labs Pegasus API failure.
- `tests/test_memory_governor.py`:
  - `test_to_point_id`: Validates deterministic UUID mapping for Qdrant compatibility.
  - `test_validate_candidate_rejections`: Validates anti-poisoning rejection of anomalous scores ($S \ge 0.080$), empty vectors, and zero vectors.
  - `test_stage_to_quarantine_in_memory`: Verifies quarantine staging with in-memory fallback.
  - `test_stage_to_quarantine_qdrant_persisted`: Verifies candidate vector and metadata payload persistence in Qdrant quarantine collection.
  - `test_promote_quarantined_vectors`: Verifies promotion to active Qdrant baseline and status payload updates in Qdrant quarantine collection.
  - `test_scrub_retention`: Verifies automated cleanup of expired entries exceeding retention days from Qdrant.
  - `test_memory_stats`: Verifies telemetry output and per-camera accounting.
- `tests/test_streaming_backpressure.py`:
  - `test_streaming_manager_default_normal_level`: Verifies default `NORMAL` tier.
  - `test_manual_level_override`: Verifies manual tier override capabilities.
  - `test_adaptive_level_transitions`: Verifies automatic transitions from Level 0 $\to$ Level 1 $\to$ Level 2 $\to$ Level 3 under simulated latency spikes and queue backpressure.
  - `test_status_telemetry`: Verifies real-time metrics tracking.
- `tests/test_backend_api.py`:
  - `test_incidents_api_lifecycle`: Full end-to-end integration test of incident creation, listing, querying by ID, status updates, and VLM explanation regeneration via Pegasus (verifying 503 when unconfigured and 200 when active).
  - `test_memory_governor_api`: Integration test of quarantine staging, listing, promotion, and stats endpoints.
  - `test_streaming_backpressure_api`: Integration test of backpressure status and manual override endpoints.


