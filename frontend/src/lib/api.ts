/**
 * Typed API wrappers for Central Cloud Backend (port 9876) and Edge Node (port 7777).
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:9876";
const EDGE_URL = process.env.NEXT_PUBLIC_EDGE_URL || "http://localhost:7777";

/* ------------------------------------------------------------------ */
/* Shared Types                                                       */
/* ------------------------------------------------------------------ */

export interface HealthResponse {
  status: string;
  qdrant_connected: boolean;
  model_loaded: boolean;
  twelve_labs_enabled: boolean;
  edge_devices_count: number;
  uptime_s: number;
}

export interface CameraStreamInfo {
  channel: number;
  name: string;
  status: string;
  substream_url: string | null;
  mainstream_url: string | null;
  snapshot_url: string | null;
  latest_score: number | null;
}

export interface VLMExplanation {
  summary: string;
  actors: string[];
  action: string;
  objects: string[];
  risk_assessment: string;
  recommended_action: string;
  model: string;
  latency_ms: number;
}

export interface IncidentEvent {
  event_id: string;
  timestamp: number;
  edge_score: number;
  cloud_score: number;
  ensemble_score: number;
  smoothed_score: number;
  snapshot_url: string | null;
  clip_url: string | null;
}

export interface IncidentRecord {
  incident_id: string;
  channel: number;
  camera_id: string;
  camera_name: string;
  start_time: number;
  end_time: number;
  duration_s: number;
  peak_score: number;
  mean_score: number;
  smoothed_score: number;
  severity: number;
  severity_badge: string;
  status: string;
  event_count: number;
  events: IncidentEvent[];
  vlm_explanation: VLMExplanation | null;
  snapshot_url: string | null;
  clip_url: string | null;
  scene_id: string;
  created_at: number;
  updated_at: number;
  notes: string | null;
}

export interface NeighborResponse {
  clip_id: string;
  similarity: number;
  source_video: string;
  scene_id: string;
  camera_id: string;
}

export interface AnomalyResultResponse {
  anomaly_score: number;
  is_anomaly: boolean;
  neighbors: NeighborResponse[];
  latency_ms: Record<string, number>;
}

export interface EscalationStats {
  total_escalations: number;
  confirmed_anomalies: number;
  confirmation_rate: number;
  [key: string]: unknown;
}

export interface SemanticSearchResultItem {
  incident_id: string;
  channel: number;
  camera_id: string;
  camera_name: string;
  score: number;
  confidence: string;
  timestamp: number;
  duration_s: number;
  severity: number;
  severity_badge: string;
  status: string;
  summary: string;
  actors: string[];
  action: string;
  objects: string[];
  snapshot_url: string | null;
  clip_url: string | null;
  video_id: string | null;
  match_reason: string;
}

export interface SemanticSearchResponse {
  query: string;
  total_matches: number;
  results: SemanticSearchResultItem[];
  latency_ms: number;
  filters_applied: Record<string, unknown>;
}

export interface CopilotEvidence {
  incident_id: string;
  channel: number;
  camera_name: string;
  timestamp: number;
  severity: number;
  severity_badge: string;
  summary: string;
  actors: string[];
  action: string;
  clip_url: string | null;
  snapshot_url: string | null;
}

export interface CopilotChatResponse {
  response: string;
  session_id: string;
  cited_incidents: string[];
  evidence: CopilotEvidence[];
  latency_ms: number;
}

export interface DailyDigestResponse {
  digest_id: string;
  period_start: number;
  period_end: number;
  generated_at: number;
  total_incidents: number;
  total_events: number;
  critical_incidents: number;
  high_incidents: number;
  moderate_incidents: number;
  low_incidents: number;
  threat_level: string;
  executive_summary: string;
  markdown_text: string;
}

export interface LoadSheddingStatus {
  current_level: number;
  level_name: string;
  auto_mode: boolean;
  avg_latency_ms: number;
  queue_depth: number;
  total_requests: number;
  shed_requests_count: number;
}

export interface DigestSummary {
  digest_id: string;
  threat_level: string;
  total_incidents: number;
  critical_incidents: number;
  high_incidents: number;
  executive_summary: string;
  generated_at: number;
}

/* ------------------------------------------------------------------ */
/* Fetch helper                                                       */
/* ------------------------------------------------------------------ */

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, { ...init, cache: "no-store" });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

/* ------------------------------------------------------------------ */
/* Central Backend Endpoints (port 9876)                              */
/* ------------------------------------------------------------------ */

export const central = {
  health: () => fetchJson<HealthResponse>(`${API_URL}/health`),

  cameras: () => fetchJson<CameraStreamInfo[]>(`${API_URL}/api/v1/cameras`),

  activeIncidents: () =>
    fetchJson<IncidentRecord[]>(`${API_URL}/api/v1/incidents/active`),

  incidents: (params?: {
    channel?: number;
    status?: string;
    min_severity?: number;
    limit?: number;
    offset?: number;
  }) => {
    const sp = new URLSearchParams();
    if (params?.channel != null) sp.set("channel", String(params.channel));
    if (params?.status) sp.set("status", params.status);
    if (params?.min_severity != null) sp.set("min_severity", String(params.min_severity));
    if (params?.limit != null) sp.set("limit", String(params.limit));
    if (params?.offset != null) sp.set("offset", String(params.offset));
    return fetchJson<IncidentRecord[]>(`${API_URL}/api/v1/incidents?${sp}`);
  },

  incident: (id: string) =>
    fetchJson<IncidentRecord>(`${API_URL}/api/v1/incidents/${id}`),

  updateIncidentStatus: (id: string, status: string, notes?: string) =>
    fetchJson<IncidentRecord>(`${API_URL}/api/v1/incidents/${id}/status`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, notes }),
    }),

  escalationStats: () =>
    fetchJson<EscalationStats>(`${API_URL}/api/v1/escalations/stats`),

  semanticSearch: (query: string, opts?: {
    channel?: number;
    min_severity?: number;
    severity_badge?: string;
    limit?: number;
    threshold?: string;
  }) =>
    fetchJson<SemanticSearchResponse>(`${API_URL}/api/v1/search/semantic`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, ...opts }),
    }),

  copilotChat: (message: string, sessionId?: string, channel?: number) =>
    fetchJson<CopilotChatResponse>(`${API_URL}/api/v1/copilot/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        channel,
      }),
    }),

  dailyDigest: () =>
    fetchJson<DailyDigestResponse>(`${API_URL}/api/v1/digest/daily`),

  digestSummary: () =>
    fetchJson<DigestSummary>(`${API_URL}/api/v1/digest/summary`),

  streamingStatus: () =>
    fetchJson<LoadSheddingStatus>(`${API_URL}/api/v1/streaming/status`),
};

/* ------------------------------------------------------------------ */
/* Edge Node Endpoints (port 7777)                                    */
/* ------------------------------------------------------------------ */

export function edgeLiveStreamUrl(channel: number): string {
  return `${EDGE_URL}/api/v1/stream/${channel}/live`;
}

export function edgeSnapshotUrl(channel: number): string {
  return `${EDGE_URL}/api/v1/stream/${channel}/snapshot`;
}

/* ------------------------------------------------------------------ */
/* Formatting Utilities                                               */
/* ------------------------------------------------------------------ */

export function formatTimestamp(ts: number): string {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

export function formatDate(ts: number): string {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

export function severityColor(badge: string): string {
  switch (badge.toUpperCase()) {
    case "CRITICAL": return "var(--severity-critical)";
    case "HIGH": return "var(--severity-high)";
    case "MODERATE": return "var(--severity-moderate)";
    default: return "var(--severity-low)";
  }
}

export function severityBgColor(badge: string): string {
  switch (badge.toUpperCase()) {
    case "CRITICAL": return "var(--severity-critical-bg)";
    case "HIGH": return "var(--severity-high-bg)";
    case "MODERATE": return "var(--severity-moderate-bg)";
    default: return "var(--severity-low-bg)";
  }
}
