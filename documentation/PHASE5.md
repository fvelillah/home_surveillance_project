# Phase 5: Semantic Video Search & AI Security Copilot

Comprehensive technical documentation for **Phase 5** of the Home Security Surveillance Analytics Platform. This document details the architecture, algorithmic implementations, multi-modal query fusion, conversational security copilot engine, automated 24-hour daily surveillance digest generator, and validation results.

---

## 1. Executive Summary & Objectives

While **Phase 4** established the Incident Formation Engine (EMA score smoothing, dual-threshold hysteresis, and Twelve Labs Pegasus VLM scene explanation), human operators still face cognitive friction when reviewing hours of surveillance history or querying specific events across all 9 camera feeds.

**Phase 5** introduces **foundational semantic intelligence and conversational interaction** across the entire 9-channel Dahua surveillance footprint:

```
                                  +-------------------------------------------------------------+
                                  |                 OPERATOR QUERY / CONSOLE                    |
                                  |   ("Find delivery courier yesterday", "What happened on Ch5?")  |
                                  +-------------------------------------------------------------+
                                                                 |
                                       REST API / POST /api/v1/search/semantic & /copilot/chat
                                                                 v
+=======================================================================================================================+
|                                              PHASE 5 INTELLIGENCE TIER                                                |
|                                                                                                                       |
|  +-----------------------------------------------------------------------------------------------------------------+  |
|  | 1. Multi-Modal Semantic Search Engine (backend/search.py)                                                       |  |
|  |    - Twelve Labs Marengo 3.0 / 2.7 Text-to-Video Visual Search                                                  |  |
|  |    - Tokenized Lexical & Semantic Metadata Matcher (Summary, Actors, Actions, Objects, Scene, Camera)           |  |
|  |    - Multi-Criteria Spatial & Temporal Filtering (Channels 1-9, Time Windows, 0-100 Severity, Badges)          |  |
|  |    - Hybrid Relevance Scoring & Deduplicated Pagination                                                         |  |
|  |    - Automatic Standalone Fallback when Cloud VLM is Offline                                                    |  |
|  +-----------------------------------------------------------------------------------------------------------------+  |
|                                                        |                                                              |
|                         +------------------------------+-------------------------------+                              |
|                         |                                                              |                              |
|                         v                                                              v                              |
|  +-----------------------------------------------------------+   +-------------------------------------------------+  |
|  | 2. Conversational Security Copilot (backend/copilot.py)   |   | 3. Daily Surveillance Digest & Routine Summary  |  |
|  |    - Grounded Q&A over Video Telemetry & Incident Records |   |    - 24-Hour Multi-Camera Event Consolidation   |  |
|  |    - Natural Language Camera & Time Intent Extraction     |   |    - Automated Threat Level (LOW..SEVERE)       |  |
|  |    - Explicit Evidence Citing (Timestamps, Severity, VLM) |   |    - 9-Channel Operational Activity Matrix      |  |
|  |    - Multi-Turn Isolated Session History Management       |   |    - Routine Pattern & Anomaly Highlighting     |  |
|  |    - Actionable Security Operator Recommendations         |   |    - Rich GitHub-Flavored Markdown Synthesis    |  |
|  +-----------------------------------------------------------+   +-------------------------------------------------+  |
+=======================================================================================================================+
```

### Key Architectural Objectives Achieved:
1. **Multi-Modal Semantic Video Search Engine (`backend/search.py`)**: Combines Twelve Labs Marengo 3.0 video embeddings and visual search with multi-camera spatial (channels 1–9), temporal (start/end timestamp), severity (0–100 score, danger badges), and status (`OPEN`, `CLOSED`, `ARCHIVED`) filters.
2. **Hybrid Relevance Ranking & Scoring Fusion**: Blends high-dimensional Marengo visual search scores ($65\%$) with tokenized metadata lexical similarity ($35\%$) across VLM summaries, identified actors, observed actions, visible objects, camera zone names, and operator notes, with a threat severity boost.
3. **Conversational Security Copilot Chat Engine (`backend/copilot.py`)**: Provides interactive natural-language dialogue grounded in actual video footage and telemetry. It accurately parses camera channels (by channel number or location name), extracts temporal context, retrieves factual evidence, and cites specific incident milestones with danger badges and video clip references.
4. **Session-Isolated Multi-Turn Memory**: Manages distinct conversational threads (`session_id`) with persistent message logs, clear session endpoints, and full evidence provenance.
5. **Automated 24-Hour Surveillance Security Digest Generator**: Synthesizes full-day operational telemetry into an executive security briefing. Features a global threat level index (`LOW`, `MODERATE`, `ELEVATED`, `SEVERE`), 9-channel activity breakdowns, chronological critical incident highlights, routine pattern observations, and proactive security recommendations in both JSON and structured Markdown.

---

## 2. Mathematical Formulations & Ranking Algorithms

### 2.1 Lexical & Semantic Metadata Similarity Score
For a natural-language query token set $Q = \{q_1, q_2, \dots, q_m\}$ and incident record $I$:
$$S_{\text{meta}}(Q, I) = w_{\text{actors}} \cdot \frac{|Q \cap T_{\text{actors}}|}{|Q|} + w_{\text{summary}} \cdot \frac{|Q \cap T_{\text{summary}}|}{|Q|} + w_{\text{action}} \cdot \frac{|Q \cap T_{\text{action}}|}{|Q|} + w_{\text{objects}} \cdot \frac{|Q \cap T_{\text{objects}}|}{|Q|} + w_{\text{camera}} \cdot \frac{|Q \cap T_{\text{cam}}|}{|Q|} + \frac{\text{Severity}(I)}{1000}$$

Where weights are tuned for security relevance:
- $w_{\text{actors}} = 0.40$ (subject identification: "intruder", "delivery driver", "vehicle")
- $w_{\text{summary}} = 0.35$ (natural language scene description)
- $w_{\text{action}} = 0.30$ (specific action: "prying door", "climbing fence", "dropping box")
- $w_{\text{camera}} = 0.25$ (camera zone name: "Front Door", "Driveway", "Back Patio")
- $w_{\text{objects}} = 0.20$ (objects visible: "crowbar", "backpack", "van")
- $w_{\text{notes}} = 0.20$ (operator annotations)

### 2.2 Multi-Modal Hybrid Search Score Fusion
When Twelve Labs Marengo returns a visual search relevance score $S_{\text{marengo}} \in [0.0, 1.0]$ for an indexed incident clip:
$$S_{\text{hybrid}} = 0.65 \cdot S_{\text{marengo}} + 0.35 \cdot S_{\text{meta}}$$

When Twelve Labs is offline or searching unindexed metadata:
$$S_{\text{hybrid}} = S_{\text{meta}}$$

### 2.3 Global Surveillance Threat Level Rating
Evaluated across all incidents $I \in \text{Digest Window}$:
$$\text{Threat Level} = \begin{cases} 
\text{SEVERE} & \text{if } \exists I \text{ with Severity Badge } = \text{CRITICAL } (\ge 75) \\
\text{ELEVATED} & \text{if } \exists I \text{ with Severity Badge } = \text{HIGH } (\ge 50) \\
\text{MODERATE} & \text{if } \exists I \text{ with Severity Badge } = \text{MODERATE } (\ge 25) \\
\text{LOW} & \text{otherwise}
\end{cases}$$

---

## 3. Module-by-Module Technical Breakdown

### 3.1 `backend/search.py` — Semantic Video Search Engine
- **Purpose**: Core search processor performing multimodal search, metadata query parsing, spatial/temporal filtering, and score fusion.
- **Key Functions & Classes**:
  - `_tokenize(text) -> Set[str]`: Fast alpha-numeric token extractor.
  - `_compute_text_similarity(tokens, incident) -> (score, reason, matched_items)`: Weighted scoring across VLM explanations and incident fields.
  - `SemanticSearchEngine.search(...) -> SemanticSearchResponse`: Orchestrates Marengo visual search and metadata ranking.
  - `search_engine`: Global singleton instance.

### 3.2 `backend/copilot.py` — AI Security Copilot & Daily Digest
- **Purpose**: Conversational Q&A engine and automated 24-hour surveillance report synthesizer.
- **Key Functions & Classes**:
  - `SecurityCopilotEngine.chat(message, session_id, channel, ...) -> CopilotChatResponse`: Multi-turn conversational interface with grounded evidence attachment.
  - `SecurityCopilotEngine._infer_channel_from_query(query) -> Optional[int]`: Extracts camera channels (e.g. "ch 4", "Driveway", "Studio") from conversational text.
  - `SecurityCopilotEngine.generate_daily_digest(start_time, end_time, ...) -> DailyDigestResponse`: Generates executive surveillance briefings and 9-channel activity matrices.
  - `SecurityCopilotEngine.get_history(session_id) / clear_history(session_id)`: Session lifecycle management.
  - `copilot_engine`: Global singleton instance.

### 3.3 `backend/models.py` — Pydantic Schemas
- **Added Models**:
  - `SemanticSearchRequest`, `SemanticSearchResultItem`, `SemanticSearchResponse`
  - `CopilotChatRequest`, `CopilotEvidence`, `CopilotMessage`, `CopilotChatResponse`
  - `DailyDigestRequest`, `ChannelActivitySummary`, `DigestIncidentSummary`, `DailyDigestResponse`

### 3.4 `backend/main.py` — REST API Endpoints
- **Exposed Endpoints**:
  - `POST /api/v1/search/semantic`: Multimodal video search.
  - `POST /api/v1/copilot/chat`: Conversational Copilot Q&A.
  - `GET /api/v1/copilot/history/{session_id}`: Retrieve session chat turns.
  - `DELETE /api/v1/copilot/history/{session_id}`: Clear session memory.
  - `GET /api/v1/digest/daily`: Retrieve full 24-hour daily report.
  - `GET /api/v1/digest/summary`: Compact telemetry summary for frontend cards.

---

## 4. API Endpoints Reference Matrix

| Method | Endpoint | Description | Key Parameters / Request Body | Response Schema |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/search/semantic` | Natural-language video search | `{"query": str, "channel": int, "min_severity": int, ...}` | `SemanticSearchResponse` |
| `POST` | `/api/v1/copilot/chat` | Conversational Security Copilot | `{"message": str, "session_id": str, "channel": int}` | `CopilotChatResponse` |
| `GET` | `/api/v1/copilot/history/{session_id}` | Get session conversation turns | `session_id: str` | `List[CopilotMessage]` |
| `DELETE` | `/api/v1/copilot/history/{session_id}` | Clear conversation session | `session_id: str` | `{"status": "ok", "deleted": bool}` |
| `GET` | `/api/v1/digest/daily` | 24-hour surveillance digest report | `start_time, end_time, channel, format` | `DailyDigestResponse` |
| `GET` | `/api/v1/digest/summary` | Dashboard summary telemetry | None | `{"threat_level": str, ...}` |

---

## 5. Evaluation CLI Tools & Testing Playbook

### 5.1 Semantic Video Search CLI (`scripts/evaluate_search.py`)
- **Purpose**: Evaluates video discovery, automatic Twelve Labs / metadata indexing from `data/tests/`, and interactive natural-language video clip search.
- **Key Commands**:
  ```bash
  # Search test clips in data/tests/
  uv run python scripts/evaluate_search.py --query "person window tool"

  # Launch interactive search shell over indexed surveillance videos
  uv run python scripts/evaluate_search.py --interactive

  # Query with spatial channel and severity filters
  uv run python scripts/evaluate_search.py --query "delivery" --channel 1 --min-severity 25

  # Run in offline mock mode without Twelve Labs API keys
  uv run python scripts/evaluate_search.py --query "intruder" --mock
  ```

### 5.2 Conversational AI Security Copilot CLI (`scripts/copilot_cli.py`)
- **Purpose**: Provides an interactive terminal conversational shell for operators to query surveillance activity, investigate incidents, and generate 24-hour daily security reports.
- **Key Commands**:
  ```bash
  # Launch interactive Copilot conversational shell
  uv run python scripts/copilot_cli.py

  # Ask a single question directly from CLI
  uv run python scripts/copilot_cli.py --query "Who was near the window with a tool?"

  # Generate 24-hour daily surveillance report
  uv run python scripts/copilot_cli.py --digest

  # Run in offline mock mode
  uv run python scripts/copilot_cli.py --mock
  ```

---

## 6. Verification & Test Results

The test suite validates semantic search, copilot dialogue grounding, multi-turn session management, 24-hour digest report generation, and standalone evaluation scripts.

```bash
uv run pytest tests/test_search.py tests/test_copilot.py tests/test_backend_api.py tests/test_evaluate_search_script.py tests/test_copilot_cli_script.py -v
```

### Verification Highlights:
- **Semantic Search**: Validated lexical matching, actor/action extraction, spatial/temporal filter combinations, severity constraints, pagination, and Marengo visual search score fusion.
- **Conversational Copilot**: Verified channel/location intent detection, grounded evidence citations with danger badges, and multi-turn session isolation.
- **Daily Digest**: Verified 24-hour aggregation across all 9 camera channels, threat level escalation (`LOW` to `SEVERE`), key incident highlights, and rich GitHub-flavored markdown output.
- **CLI Evaluation Scripts**: Verified synthetic video sample generation, metadata extraction, live Twelve Labs API execution, offline mocking, and JSON file export across `scripts/evaluate_search.py` and `scripts/copilot_cli.py`.
- **Full Workspace Regression**: All **123 tests** passed with zero regressions across Phases 1 through 5.

---

## 7. Next Steps (Phase 6)

With Phase 5 complete, the project moves to **Phase 6: Multi-Camera Web Console & Live UI**:
- **Task 6.1**: Next.js 14 project shell, dark mode design system, and state providers (`AppContext.tsx`, `CameraContext.tsx`).
- **Task 6.2**: 9-channel responsive live video grid (`CameraGrid.tsx`, `CameraPanel.tsx`) using direct HTTP MJPEG proxy streams with real-time FPS and anomaly meters.
- **Task 6.3**: Real-time alert queue with severity gauges and interactive incident cards (`AlertQueue.tsx`, `ActiveIncidentCard.tsx`).
- **Task 6.4**: Synchronized incident video playback modal (`IncidentModal.tsx`) with normal baseline nearest-neighbor comparison grid.
- **Task 6.5**: 24-hour continuous multi-camera timeline scrubber and anomaly heatmap (`ContinuousTimeline.tsx`, `AnomalyMeter.tsx`).
- **Task 6.6**: Semantic search console and AI Security Copilot drawer (`OpsCopilot.tsx`).
