/**
 * Type definitions for Cyber-Defense Surveillance Matrix Console.
 */

export interface CameraInfo {
  channel: number;
  name: string;
  status: 'active' | 'offline' | 'degraded' | 'reconnecting';
  substream_url: string;
  snapshot_url: string;
  latest_score: number;
  fps?: number;
  latency_ms?: number;
  is_connected?: boolean;
  escalations?: number;
  resolution?: string;
}

export interface ScoreHistoryPoint {
  timestamp: number;
  score: number;
  is_escalated: boolean;
}

export interface ClusterHealth {
  status: string;
  qdrant_connected: boolean;
  model_loaded: boolean;
  twelve_labs_enabled: boolean;
  edge_devices_count: number;
  uptime_s: number;
  edge_node_connected?: boolean;
  ingest_fps?: number;
}

export interface VLMExplanation {
  explanation_id?: string;
  incident_id?: string;
  model?: string;
  summary: string;
  actors_identified?: string[];
  threat_level?: string;
  reason?: string;
  generated_at?: string | number;
  confidence?: number;
}

export interface NearestNeighbor {
  clip_id: string;
  similarity: number;
  source_video?: string;
  scene_id?: string;
  camera_id?: string;
  thumbnail_url?: string;
}

export interface IncidentEvent {
  event_id: string;
  timestamp: number;
  score: number;
  raw_score?: number;
  clip_path?: string;
  thumbnail_path?: string;
}

export interface IncidentRecord {
  incident_id: string;
  channel: number;
  camera_name?: string;
  status: 'OPEN' | 'ACKNOWLEDGED' | 'CLOSED' | 'ARCHIVED';
  start_time: number;
  end_time: number;
  duration_seconds: number;
  peak_score: number;
  severity_score: number; // 0..100
  severity_badge: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  events_count: number;
  events?: IncidentEvent[];
  clip_path?: string;
  snapshot_path?: string;
  vlm_explanation?: VLMExplanation;
  nearest_neighbors?: NearestNeighbor[];
  operator_notes?: string;
  acknowledged_at?: number;
  acknowledged_by?: string;
}

export interface SemanticSearchResultItem {
  incident_id: string;
  channel: number;
  camera_name?: string;
  timestamp: number;
  severity_score: number;
  severity_badge: string;
  status: string;
  similarity_score: number;
  summary: string;
  clip_path?: string;
  snapshot_path?: string;
}

export interface SemanticSearchResponse {
  query: string;
  total_matches: number;
  results: SemanticSearchResultItem[];
  latency_ms: number;
}

export interface CopilotMessage {
  id?: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: number;
  citations?: Array<{
    incident_id: string;
    channel: number;
    timestamp: number;
    severity_score: number;
  }>;
}

export interface DailyDigest {
  digest_id: string;
  generated_at: string | number;
  period_start: string | number;
  period_end: string | number;
  threat_level: 'NORMAL' | 'ELEVATED' | 'HIGH' | 'CRITICAL';
  total_incidents: number;
  critical_incidents: number;
  high_incidents: number;
  camera_incident_counts: Record<number, number>;
  executive_summary: string;
  detailed_narrative?: string;
  key_takeaways?: string[];
  recommendations?: string[];
}

export interface LoadSheddingStatus {
  current_level: number;
  level_name: string;
  auto_mode: boolean;
  cpu_percent: number;
  queue_latency_ms: number;
  frames_dropped_total: number;
  active_subscribers: number;
}

export interface QuarantineItem {
  item_id: string;
  camera_id: string;
  channel: number;
  anomaly_score: number;
  staged_at: number;
  age_minutes: number;
  status: 'PENDING' | 'APPROVED' | 'REJECTED';
}

export interface TimelineDataPoint {
  time_label: string;
  timestamp: number;
  channels: Record<number, number>; // channel -> max anomaly score
}
