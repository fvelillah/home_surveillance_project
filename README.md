# Home Security Surveillance Analytics Platform

An enterprise-grade, edge-to-cloud AI surveillance and threat intelligence platform designed for 9-channel CCTV systems (Dahua NVR `DH-NVR5216-16P-4K` with PoE IP cameras).

---

## Project Overview

The platform transforms raw security camera footage into real-time threat intelligence through a three-tier hybrid architecture:

- **Edge Node (`port 7777`)**: Direct HTTP video stream ingestion into memory ring buffers, 10-second sliding-window segmentation, PyTorch [MobileNetV3](edge/model.py) feature extraction (576-dim), local Qdrant Edge baseline triage ($T_{\text{edge}} = 0.060$), and crash-resilient SQLite offline queuing.
- **Central Cloud Backend (`port 9876`)**: High-precision vector baseline matching in Qdrant Cloud, 70/30 cloud-edge ensemble scoring, Twelve Labs Pegasus 1.5 VLM visual scene reasoning, Marengo 3.0 semantic video search, and a conversational AI Security Copilot.
- **Cyber-Defense Matrix Web Console (`port 3000`)**: Next.js 14 mission-critical HUD featuring a 9-channel direct MJPEG live grid, real-time threat queue with synthesized Web Audio alarms, 24-hour continuous multi-camera anomaly scrubber, and an interactive AI Security Copilot drawer.

---

## Quickstart: Running the Application

### 1. Prerequisites & Environment Setup

- **Python 3.10+** with [`uv`](https://github.com/astral-sh/uv) installed
- **Node.js 18+** & `npm`
- **Docker & Docker Compose** (optional, for containerized deployment)

Clone and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Dahua NVR IP/credentials, Qdrant Cloud URL/API key, and Twelve Labs API key
```

Install dependencies:
```bash
# Python dependencies (Edge & Backend)
uv sync

# Next.js Web Console dependencies
cd frontend && npm install && cd ..
```

---

### 2. Option A: Run via Docker Compose (Recommended for Production)

Deploy the full production stack (`surveillance-edge`, `surveillance-backend`, `surveillance-frontend`, and optional local `qdrant`) in a single command:

```bash
# 1. Run automated pre-flight audit
./scripts/deploy_production.sh

# 2. Launch all containerized services in background
docker compose -f docker-compose.prod.yml up -d --build

# 3. View live operational logs
docker compose -f docker-compose.prod.yml logs -f
```

Access the Web Console at: **[http://localhost:3000](http://localhost:3000)**

---

### 3. Option B: Run in Local Development Mode

Run the services across separate terminal tabs:

#### Terminal 1 — Central Backend (Port 9876)
Handles Qdrant vector scoring, Twelve Labs VLM reasoning, search, and Copilot:
```bash
uv run uvicorn backend.main:app --host 0.0.0.0 --port 9876 --reload
```

#### Terminal 2 — Edge Node & Streaming Proxy (Port 7777)
Ingests Dahua HTTP streams, performs MobileNetV3 feature extraction, and manages triage:
```bash
uv run uvicorn edge.main:app --host 0.0.0.0 --port 7777 --reload
```

> **Testing without Dahua hardware?** Launch the synthetic 9-camera MJPEG feed simulator:
> ```bash
> uv run python scripts/simulate_dahua_feeds.py
> ```

#### Terminal 3 — Cyber-Defense Web Console (Port 3000)
Next.js 14 reactive tactical matrix UI:
```bash
cd frontend && npm run dev
```

Open your browser at **[http://localhost:3000](http://localhost:3000)**.

---

## Baseline Calibration & Offline Storage Ingestion

To calibrate normal routine centroids directly from external USB storage without copying video files to the local hard drive:

```bash
# Ingest USB footage, wipe previous vectors, and calibrate 500 centroids per channel:
uv run python scripts/embed_and_index_baseline.py \
    --video-dir "/mnt/e/NVR/2026-9-19,/mnt/e/NVR/2026-9-20" \
    --clusters-per-camera 500 \
    --window-stride-s 20.0 \
    --clear-db

# Empirically evaluate distance percentiles and update .env operating thresholds:
uv run python scripts/evaluate_thresholds.py \
    --collection anomaly_baseline \
    --apply-env
```

---

## System Verification & Test Suite

Run the full workspace test suite (148 automated unit, integration, and contract tests):

```bash
uv run pytest
```

Build the optimized Next.js frontend production bundle:
```bash
cd frontend && npm run build
```

---

## Architecture Reference

| Service | Port | Endpoint / Role |
| :--- | :--- | :--- |
| **Web Console** | `3000` | Tactical Cyber-Defense Matrix HUD (`http://localhost:3000`) |
| **Central Backend** | `9876` | OpenAPI docs (`/docs`), Pegasus VLM (`/api/v1/explain`), Copilot (`/api/v1/copilot/chat`) |
| **Edge Node** | `7777` | Health (`/health`), Telemetry SSE (`/api/v1/stream/anomalies`), Live MJPEG (`/api/v1/stream/{ch}/live`) |
| **Qdrant DB** | `6333` | Vector collection `anomaly_baseline` (576-dim cosine similarity) |

For comprehensive architectural deep dives and implementation specifications, see the documentation in `documentation/` (`PHASE1.md` through `PHASE7.md`).