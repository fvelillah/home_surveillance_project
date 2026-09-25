# Phase 6: Multi-Camera Web Console & Live UI (Option 2: Cyber-Defense Matrix)

Comprehensive technical documentation for **Phase 6 (Option 2)** of the Home Security Surveillance Analytics Platform. This document details the modern Web Console architecture, Cyber-Obsidian & Neon Telemetry design system, 9-channel direct HTTP video streaming grid, real-time threat stream, synchronized dual-pane incident investigation modal with Qdrant baseline nearest-neighbor comparison, continuous 24-hour multi-channel anomaly heatmap scrubber, conversational AI Security Copilot drawer, and verification results.

---

## 1. Executive Summary & Design Vision (Option 2 vs Option 1)

While Option 1 followed a standard security console pattern, **Option 2** introduces a high-density, mission-critical **Tactical Cyber-Defense & AI Intelligence Matrix**:

```
+=======================================================================================================================+
|                            CYBER-DEFENSE SURVEILLANCE MATRIX // DIRECT HTTP ENGINE (PORT 3000)                        |
|  [RADAR] BACKEND 9876: OK | EDGE 7777: OK | QDRANT: OK | PEGASUS VLM: OK | 25 FPS | DEFCON: NORMAL | AUDIO: UNMUTED   |
+=======================================================================================================================+
|                                                           |                                                           |
|   9-CHANNEL LIVE TACTICAL GRID (3x3 / 1+8 FOCUS)         |   REAL-TIME THREAT STREAM (ALERT QUEUE)                   |
|                                                           |                                                           |
|   +-------------------+  +-------------------+            |   +---------------------------------------------------+   |
|   | CH-01 DRIVEWAY    |  | CH-02 ENTRY PORCH |            |   | CH-02 • 2m ago                 CRITICAL • 91/100  |   |
|   | [LIVE MJPEG STREAM|  | [LIVE MJPEG STREAM|            |   | [THUMBNAIL] Suspicious movement near porch        |   |
|   | 25 FPS • 18ms     |  | 25 FPS • 19ms     |            |   | [PEGASUS VLM REASONED] [ACK] [INSPECT]            |   |
|   | (O) 0.02 NOMINAL  |  | (O) 0.91 ESCALATED|            |   +---------------------------------------------------+   |
|   +-------------------+  +-------------------+            |   +---------------------------------------------------+   |
|   +-------------------+  +-------------------+            |   | CH-05 • 14m ago                   HIGH • 72/100   |   |
|   | CH-03 NORTH WALK  |  | CH-04 SOUTH PATH  |            |   | [THUMBNAIL] Unidentified shadow near fence        |   |
|   | [LIVE MJPEG STREAM|  | [LIVE MJPEG STREAM|            |   | [PEGASUS VLM REASONED] [ACK] [INSPECT]            |   |
|   | 25 FPS • 21ms     |  | 25 FPS • 17ms     |            |   +---------------------------------------------------+   |
|   | (O) 0.03 NOMINAL  |  | (O) 0.04 NOMINAL  |            |                                                           |
|   +-------------------+  +-------------------+            |   [FILTER: ALL / OPEN / ACK / CLOSED] [SECTOR: ALL]       |
|                                                           |                                                           |
+===========================================================+===========================================================+
|   24-HOUR CONTINUOUS MULTI-FEED TIMELINE & ANOMALY HEATMAP                                                            |
|   CH-01 [--------------------||||-----------------------------]                                                       |
|   CH-02 [----------------------||||||||||||-------------------] <-- Interactive 24h Time Scrub Cursor (T - 2h 14m)     |
|   ...                                                                                                                 |
|   CH-09 [-----------------------------------------------------]                                                       |
|   [RANGE: 15M / 1H / 6H / 24H] • GREEN: NOMINAL (<0.05) • AMBER: ELEVATED (0.05-0.08) • RED: ESCALATED (>=0.08)       |
+=======================================================================================================================+
```

---

## 2. Key Architecture & Modules Breakdown

### 2.1 Next.js 14 App Shell & Cyber-Obsidian HUD (`frontend/src/app/`)
- **Root Layout (`layout.tsx`)**: Integrates Google Fonts (`Outfit`, `Inter`, `JetBrains Mono`), dark theme viewport metadata, and portal mount points.
- **Design System (`globals.css`)**: Implements CSS variables for deep obsidian surfaces (`#06090e`, `#0a0f18`), neon HUD accents (Emerald Green `#10b981`, Amber `#f59e0b`, Crimson `#ef4444`, Cyan `#06b6d4`), CRT scanline overlay, tactical corner brackets (`.tactical-box`), and pulsing alert glows (`@keyframes pulseGlowRed`).
- **Dashboard Layout (`page.tsx`, `page.module.css`)**: Assembles the sticky command bar, 2-column tactical workspace (camera matrix + live alert queue), and bottom continuous 24h timeline.

### 2.2 Global State & Real-Time Context (`frontend/src/context/SurveillanceContext.tsx`)
- **Adaptive Polling Engine**: Polls health, cameras, live anomaly scores, and incidents concurrently every 2.5 seconds with graceful fallback handling.
- **Audio Alarm Synthesizer (`src/lib/audio.ts`)**: Built with Web Audio API (sine, triangle, and sawtooth oscillators with exponential decay) synthesizing tactical radar pings and 2-tone warning chimes for critical anomalies without requiring external audio assets.
- **Multi-View Modes**: Seamlessly switches between `matrix` (3x3 grid) and `focus` (1 high-res primary feed + 8 thumbnail strip).

### 2.3 9-Channel Live Camera Grid (`src/components/CameraGrid.tsx`, `src/components/CameraPanel.tsx`)
- **Direct HTTP MJPEG Proxy**: Streams decoded video directly from Edge Node RAM ring buffers (`http://localhost:7777/api/v1/stream/{ch}/live`) bypassing RTSP handshakes.
- **Live HUD Overlays**: Decoded FPS counter, latency (ms), channel tag, camera name, and radial anomaly gauge (`AnomalyMeter.tsx`).
- **Tactical Controls**: Instant snapshot download, Qdrant Edge baseline vector seeding, digital zoom, stream refresh, and pulsating red radar aura when $T \ge 0.080$.

### 2.4 Real-Time Alert Queue & Threat Stream (`src/components/AlertQueue.tsx`, `src/components/ActiveIncidentCard.tsx`)
- **Consolidated Threat Stream**: Live incident cards ordered by timestamp and severity score (0–100).
- **Incident Lifecycle Management**: Status filtering (`ALL`, `OPEN`, `ACKNOWLEDGED`, `CLOSED`), channel filtering, and one-click acknowledge/close actions.
- **Pegasus VLM Badging**: Displays natural language scene reasoning chips and instant explain triggers.

### 2.5 Dual-Pane Incident Investigation Modal (`src/components/IncidentModal.tsx`)
- **Left Pane**: High-resolution video clip / snapshot evidence playback with duration and peak score telemetry.
- **Right Top Pane**: Twelve Labs Pegasus VLM structured scene breakdown (narrative summary, identified actors, risk level, re-run trigger).
- **Right Bottom Pane**: **Qdrant Baseline Nearest-Neighbor Comparison Grid** displaying top-3 normal baseline reference vectors with cosine similarity meters ($0.0 - 1.0$).
- **Lifecycle Controls**: Status selector (`OPEN`, `ACKNOWLEDGED`, `CLOSED`, `ARCHIVED`) and operator investigation notes.

### 2.6 24-Hour Continuous Multi-Feed Timeline (`src/components/ContinuousTimeline.tsx`)
- **Multi-Feed Anomaly Heatmap**: 48-block horizontal time tracks across all 9 camera channels simultaneously.
- **Interactive Scrubber**: Synchronized cursor displaying relative time (`T - 2h 14m`) with range selectors (`15m`, `1h`, `6h`, `24h`).

### 2.7 AI Security Copilot & Semantic Search Drawer (`src/components/OpsCopilot.tsx`)
- **Tab 1 (Semantic Video Search)**: Natural language text search combined with spatial sector filters and similarity threshold slider ($0.20 - 0.80$).
- **Tab 2 (Conversational Copilot Chat)**: Interactive conversational dialogue grounded in Dahua surveillance events with quick suggestion chips and multi-turn session memory.
- **Tab 3 (Daily Surveillance Digest)**: 24-hour executive summary briefing, threat level gauge, and key takeaways.

### 2.8 System Governance Hub (`src/components/GovernancePanel.tsx`)
- **Adaptive Load Shedding Controller**: Interactive toggling between Levels 0–3 (`NORMAL`, `SCORE_ONLY`, `PASSTHROUGH`, `SHED_LOAD`) and auto-mode switch.
- **Observation Quarantine Buffer Manager**: View candidate normal vectors, promote matured vectors to Central Qdrant, and execute 7-day retention scrubs.

---

## 3. Verification & Validation Results

### 3.1 Next.js 14 Production Bundle Compilation
```bash
npx next build
```
- **Result**: `✓ Compiled successfully in 16.8s (4/4 static pages generated, 0 TypeScript errors, 0 ESLint errors)`.
- **First Load JS**: `104 kB` (lightweight, zero bloated dependencies).

### 3.2 TypeScript Type Checking
```bash
npx --no-install tsc --noEmit
```
- **Result**: Code 0 (zero type errors across all components, API clients, and contexts).

### 3.3 Backend Integration & API Contract Test Suite
```bash
uv run pytest tests/test_web_ui_api_contract.py
```
- `test_ui_health_contract`: **PASSED** (Cluster health schema validated)
- `test_ui_cameras_contract`: **PASSED** (9 camera channels with direct HTTP stream endpoints validated)
- `test_ui_incident_lifecycle_contract`: **PASSED** (Creation, active retrieval, status transitions `OPEN -> ACKNOWLEDGED -> CLOSED`, and notes persistence validated)
- `test_ui_semantic_search_contract`: **PASSED** (Marengo text-to-video search contract validated)
- `test_ui_copilot_chat_contract`: **PASSED** (Conversational Q&A turn validated)
- `test_ui_daily_digest_contract`: **PASSED** (Executive summary and 24h daily digest validated)
- `test_ui_governance_contract`: **PASSED** (Streaming load-shedding overrides and memory quarantine validated)
- **Result**: `7 passed, 2 warnings in 3.68s (100% test pass rate)`.

---

## 4. Master Checklist Update

With Phase 6 fully implemented and verified on branch `feature/web-ui-option2`:
- **Phase 1: Direct HTTP Ingestion & Core Infrastructure (5/5)**: Complete
- **Phase 2: Edge Triage & Qdrant Edge (4/4)**: Complete
- **Phase 3: Central Cloud Backend & kNN (4/4)**: Complete
- **Phase 4: Incident Formation & VLM Explainer (4/4)**: Complete
- **Phase 5: Semantic Video Search & AI Copilot (3/3)**: Complete
- **Phase 6: Multi-Camera Web Console & Live UI (6/6)**: Complete
- **Overall Progress**: `26 / 31 Tasks Completed (83.9%)`.
