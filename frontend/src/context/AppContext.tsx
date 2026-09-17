"use client";

import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import {
  central,
  type HealthResponse,
  type IncidentRecord,
  type LoadSheddingStatus,
  type DigestSummary,
} from "@/lib/api";

/* ------------------------------------------------------------------ */
/* Context shape                                                      */
/* ------------------------------------------------------------------ */

interface AppState {
  health: HealthResponse | null;
  activeIncidents: IncidentRecord[];
  allIncidents: IncidentRecord[];
  streamingStatus: LoadSheddingStatus | null;
  digestSummary: DigestSummary | null;
  loading: boolean;
  error: string | null;
  /** Force an immediate refresh of all data */
  refresh: () => void;
  /** The incident currently viewed in modal, null = closed */
  selectedIncident: IncidentRecord | null;
  setSelectedIncident: (inc: IncidentRecord | null) => void;
  /** Copilot drawer visibility */
  copilotOpen: boolean;
  setCopilotOpen: (open: boolean) => void;
}

const AppContext = createContext<AppState>({
  health: null,
  activeIncidents: [],
  allIncidents: [],
  streamingStatus: null,
  digestSummary: null,
  loading: true,
  error: null,
  refresh: () => {},
  selectedIncident: null,
  setSelectedIncident: () => {},
  copilotOpen: false,
  setCopilotOpen: () => {},
});

export const useApp = () => useContext(AppContext);

/* ------------------------------------------------------------------ */
/* Provider                                                           */
/* ------------------------------------------------------------------ */

const POLL_INTERVAL_MS = 5_000;

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [activeIncidents, setActiveIncidents] = useState<IncidentRecord[]>([]);
  const [allIncidents, setAllIncidents] = useState<IncidentRecord[]>([]);
  const [streamingStatus, setStreamingStatus] = useState<LoadSheddingStatus | null>(null);
  const [digestSummary, setDigestSummary] = useState<DigestSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedIncident, setSelectedIncident] = useState<IncidentRecord | null>(null);
  const [copilotOpen, setCopilotOpen] = useState(false);

  // Track previously known incident IDs for new-incident audio alert
  const prevIdsRef = useRef<Set<string>>(new Set());
  const audioCtxRef = useRef<AudioContext | null>(null);

  const playAlertBeep = useCallback(() => {
    try {
      if (!audioCtxRef.current) audioCtxRef.current = new AudioContext();
      const ctx = audioCtxRef.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = 880;
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.4);
    } catch {
      /* audio not available */
    }
  }, []);

  const fetchAll = useCallback(async () => {
    try {
      const [h, active, all, ss, ds] = await Promise.allSettled([
        central.health(),
        central.activeIncidents(),
        central.incidents({ limit: 100 }),
        central.streamingStatus(),
        central.digestSummary(),
      ]);

      if (h.status === "fulfilled") setHealth(h.value);
      if (active.status === "fulfilled") {
        const newActive = active.value;
        // Detect new CRITICAL/HIGH incidents
        const newIds = new Set(newActive.map((i) => i.incident_id));
        for (const inc of newActive) {
          if (
            !prevIdsRef.current.has(inc.incident_id) &&
            (inc.severity_badge === "CRITICAL" || inc.severity_badge === "HIGH")
          ) {
            playAlertBeep();
            break;
          }
        }
        prevIdsRef.current = newIds;
        setActiveIncidents(newActive);
      }
      if (all.status === "fulfilled") setAllIncidents(all.value);
      if (ss.status === "fulfilled") setStreamingStatus(ss.value);
      if (ds.status === "fulfilled") setDigestSummary(ds.value);

      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Backend unreachable");
    } finally {
      setLoading(false);
    }
  }, [playAlertBeep]);

  useEffect(() => {
    fetchAll();
    const iv = setInterval(fetchAll, POLL_INTERVAL_MS);
    return () => clearInterval(iv);
  }, [fetchAll]);

  return (
    <AppContext.Provider
      value={{
        health,
        activeIncidents,
        allIncidents,
        streamingStatus,
        digestSummary,
        loading,
        error,
        refresh: fetchAll,
        selectedIncident,
        setSelectedIncident,
        copilotOpen,
        setCopilotOpen,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}
