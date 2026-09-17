# SENTINEL — Home Security Surveillance Analytics Platform

AI-powered video analytics for a 9-camera Dahua CCTV home security system. Real-time anomaly detection, natural language incident descriptions, semantic video search, and a live operator console.

## CCTV Hardware

- **NVR**: Dahua DH-NVR5216-16P-4K
- **Cameras**: 9× PoE IP cameras connected to the NVR

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Edge Node (Port 7777)                                  │
│  9× HTTP MJPEG streams → Segmenter → MobileNetV3 →     │
│  Qdrant Edge kNN scorer → SQLite offline queue          │
└──────────────────────┬──────────────────────────────────┘
                       │ Escalated clips
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Central Cloud Backend (Port 9876)                      │
│  Twelve Labs Marengo embeddings → Qdrant Baseline kNN → │
│  Ensemble scorer (70% cloud + 30% edge) → Incidents →   │
│  Pegasus VLM explainer → Copilot chat → Daily digest    │
└──────────────────────┬──────────────────────────────────┘
                       │ REST API
                       ▼
┌─────────────────────────────────────────────────────────┐
│  SENTINEL Web Console (Port 3000)                       │
│  3×3 live camera grid · Alert queue · Incident modal ·  │
│  24h timeline · Semantic search · AI copilot drawer     │
└─────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Dependency | Version | Purpose |
|---|---|---|
| **Python** | ≥ 3.10 | Backend & edge services |
| **uv** | latest | Python package management |
| **Node.js** | ≥ 18 | Frontend dev server |
| **npm** | ≥ 9 | Frontend dependencies |
| **Docker** *(optional)* | latest | Qdrant vector database |

---

## 1. Environment Setup

```bash
# Clone and enter the project
cd home_surveillance_project

# Copy the example environment file and edit with your values
cp .env.example .env

# Install Python dependencies (backend + edge)
uv sync --all-extras

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Key environment variables to configure in `.env`

| Variable | Description |
|---|---|
| `DAHUA_NVR_IP` | IP address of your Dahua NVR on the local network |
| `DAHUA_NVR_USER` / `DAHUA_NVR_PASSWORD` | NVR login credentials |
| `CAMERA_NAMES` | Friendly names for each channel (e.g. `1:Studio,2:Kitchen,...`) |
| `QDRANT_URL` | Qdrant cluster URL (`http://localhost:6333` for local Docker) |
| `QDRANT_API_KEY` | *(Optional)* API key for Qdrant Cloud |
| `TWELVE_LABS_API_KEY` | *(Optional)* API key for Twelve Labs VLM/Marengo. Without it, the system falls back to the local PyTorch model server |

---

## 2. Start Support Services

### Qdrant Vector Database (optional — system falls back to in-memory NumPy if unavailable)

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/data/qdrant_central:/qdrant/storage:z \
  qdrant/qdrant:latest
```

### Local PyTorch Model Server (optional fallback when Twelve Labs is not configured)

```bash
uv run python model_server.py --port 9877
```

---

## 3. Run the Application

Open **three separate terminals**, all from the project root (`home_surveillance_project/`):

### Terminal 1 — Edge Node (Port 7777)

The edge node connects to the Dahua NVR, ingests 9 camera streams, runs real-time anomaly triage, and proxies live MJPEG video to the frontend.

```bash
uv run uvicorn edge.main:app --host 0.0.0.0 --port 7777 --reload
```

> **Note**: If the NVR is not reachable, you can run the mock MJPEG simulator for development:
> ```bash
> uv run python scripts/simulate_dahua_feeds.py
> ```

### Terminal 2 — Central Cloud Backend (Port 9876)

The central backend performs cloud-grade anomaly scoring, incident formation, VLM scene analysis, semantic search, and AI copilot chat.

```bash
uv run uvicorn backend.main:app --host 0.0.0.0 --port 9876 --reload
```

### Terminal 3 — SENTINEL Web Console (Port 3000)

The operator dashboard with live camera grid, alert queue, incident modal, timeline, and copilot.

```bash
cd frontend
npm run dev
```

Then open **http://localhost:3000** in your browser.

---

## 4. Verify Everything is Running

### Health check (central backend)

```bash
curl -s http://localhost:9876/health | python -m json.tool
```

Expected output:
```json
{
    "status": "ok",
    "qdrant_connected": true,
    "model_loaded": true,
    "twelve_labs_enabled": true,
    "edge_devices_count": 9,
    "uptime_s": 12.4
}
```

### Camera catalog

```bash
curl -s http://localhost:9876/api/v1/cameras | python -m json.tool
```

### Edge node health

```bash
curl -s http://localhost:7777/health | python -m json.tool
```

### Frontend

Navigate to http://localhost:3000 — you should see the SENTINEL dashboard with 9 camera panels, alert queue, and the timeline.

---

## 5. Run Tests

```bash
# Full test suite (all phases)
uv run pytest

# Individual phase tests
uv run pytest tests/test_stream_worker.py tests/test_capture_manager.py tests/test_segmenter.py     # Phase 1
uv run pytest tests/test_model.py tests/test_detector.py tests/test_queue.py tests/test_edge_api.py  # Phase 2
uv run pytest tests/test_backend_api.py tests/test_backend_anomaly.py tests/test_backend_ensemble.py # Phase 3
uv run pytest tests/test_backend_incidents.py tests/test_vlm_explainer.py tests/test_memory_governor.py # Phase 4
uv run pytest tests/test_search.py tests/test_copilot.py tests/test_copilot_cli_script.py            # Phase 5
```

---

## 6. CLI Tools

```bash
# Evaluate VLM scene explainer
uv run python scripts/evaluate_vlm.py --video data/tests/video_test.mp4

# Semantic video search
uv run python scripts/evaluate_search.py --query "person near garage at night"

# AI security copilot (interactive REPL)
uv run python scripts/copilot_cli.py

# Daily surveillance digest
uv run python scripts/copilot_cli.py --digest

# NVR connection diagnostic
uv run python scripts/test_nvr_connection.py
```

---

## Project Structure

```
home_surveillance_project/
├── edge/                    # Edge node (Port 7777)
│   ├── stream_worker.py     # Dahua HTTP MJPEG stream reader
│   ├── capture_manager.py   # 9-channel orchestrator
│   ├── segmenter.py         # 10s sliding window clip generator
│   ├── model.py             # MobileNetV3 feature extractor
│   ├── detector.py          # Qdrant two-shard kNN scorer
│   ├── queue.py             # SQLite offline queue + auto-drain
│   └── main.py              # FastAPI + MJPEG stream proxy
├── backend/                 # Central cloud backend (Port 9876)
│   ├── main.py              # FastAPI routes & endpoints
│   ├── models.py            # Pydantic schemas
│   ├── anomaly.py           # Qdrant baseline kNN scorer
│   ├── ensemble.py          # Multi-model ensemble (70/30)
│   ├── escalation.py        # Edge escalation handler
│   ├── incidents.py         # Incident formation engine
│   ├── vlm_explainer.py     # Pegasus VLM scene explainer
│   ├── memory.py            # Memory governor & anti-poisoning
│   ├── streaming.py         # Backpressure & load shedding
│   ├── search.py            # Semantic video search
│   ├── copilot.py           # AI copilot chat + daily digest
│   ├── twelvelabs_client.py # Twelve Labs API wrapper
│   └── config.py            # Configuration from .env
├── frontend/                # SENTINEL web console (Port 3000)
│   └── src/
│       ├── app/             # Next.js 14 App Router pages
│       ├── components/      # React UI components
│       ├── context/         # AppContext, CameraContext
│       └── lib/api.ts       # Typed API client
├── scripts/                 # CLI tools & utilities
├── tests/                   # Automated test suite
├── documentation/           # Phase-by-phase docs (PHASE1–5.md)
├── model_server.py          # Local PyTorch fallback server
├── .env.example             # Environment template
└── pyproject.toml           # Python project config
```