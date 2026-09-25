'use client';

import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react';
import {
  CameraInfo,
  ClusterHealth,
  DailyDigest,
  IncidentRecord,
  LoadSheddingStatus,
} from '@/types';
import {
  fetchCameras,
  fetchClusterHealth,
  fetchDailyDigest,
  fetchDigestSummary,
  fetchIncidents,
  fetchLiveScores,
  fetchStreamingStatus,
  triggerSimulatedIncident,
  seedBaselineVectors,
} from '@/lib/api';
import { soundFx } from '@/lib/audio';

export type ViewMode = 'matrix' | 'focus' | 'split';
export type CopilotTab = 'search' | 'copilot' | 'digest';

interface SurveillanceContextType {
  // Telemetry & Cluster State
  health: ClusterHealth | null;
  cameras: CameraInfo[];
  incidents: IncidentRecord[];
  activeIncidents: IncidentRecord[];
  dailyDigest: DailyDigest | null;
  streamingStatus: LoadSheddingStatus | null;
  lastUpdated: Date;
  isLoading: boolean;

  // Viewport & Selection State
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  selectedCamera: number | null;
  setSelectedCamera: (channel: number | null) => void;

  // Modals & Drawers
  inspectIncident: IncidentRecord | null;
  setInspectIncident: (inc: IncidentRecord | null) => void;
  isCopilotOpen: boolean;
  setIsCopilotOpen: (open: boolean) => void;
  copilotTab: CopilotTab;
  setCopilotTab: (tab: CopilotTab) => void;
  isGovernanceOpen: boolean;
  setIsGovernanceOpen: (open: boolean) => void;

  // Audio Control
  isAudioMuted: boolean;
  toggleAudioMute: () => void;

  // Incident Filtering
  incidentStatusFilter: string;
  setIncidentStatusFilter: (status: string) => void;
  incidentChannelFilter: number | null;
  setIncidentChannelFilter: (ch: number | null) => void;

  // Actions
  refreshAll: () => Promise<void>;
  simulateAnomaly: (channel: number, score?: number) => Promise<void>;
  seedBaseline: (channel: number, count?: number) => Promise<void>;
}

const SurveillanceContext = createContext<SurveillanceContextType | undefined>(undefined);

export const SurveillanceProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [health, setHealth] = useState<ClusterHealth | null>(null);
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [incidents, setIncidents] = useState<IncidentRecord[]>([]);
  const [dailyDigest, setDailyDigest] = useState<DailyDigest | null>(null);
  const [streamingStatus, setStreamingStatus] = useState<LoadSheddingStatus | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // View & UI Navigation
  const [viewMode, setViewMode] = useState<ViewMode>('matrix');
  const [selectedCamera, setSelectedCamera] = useState<number | null>(1);
  const [inspectIncident, setInspectIncident] = useState<IncidentRecord | null>(null);
  const [isCopilotOpen, setIsCopilotOpen] = useState<boolean>(false);
  const [copilotTab, setCopilotTab] = useState<CopilotTab>('copilot');
  const [isGovernanceOpen, setIsGovernanceOpen] = useState<boolean>(false);
  const [isAudioMuted, setIsAudioMuted] = useState<boolean>(false);

  // Filters
  const [incidentStatusFilter, setIncidentStatusFilter] = useState<string>('ALL');
  const [incidentChannelFilter, setIncidentChannelFilter] = useState<number | null>(null);

  // Track known incident IDs to trigger audio warning on new incoming critical incidents
  const knownIncidentIds = useRef<Set<string>>(new Set());

  const toggleAudioMute = () => {
    const next = !isAudioMuted;
    setIsAudioMuted(next);
    soundFx.setMuted(next);
    if (!next) {
      soundFx.playRadarPing();
    }
  };

  // Main Data Refresh Cycle
  const refreshAll = useCallback(async () => {
    try {
      const [h, camList, scoreData, incList, digest, streamStat] = await Promise.all([
        fetchClusterHealth(),
        fetchCameras(),
        fetchLiveScores(),
        fetchIncidents({ limit: 40 }),
        fetchDailyDigest('full'),
        fetchStreamingStatus(),
      ]);

      setHealth(h);

      // Merge live score updates into camera list
      const mergedCameras = camList.map((cam) => {
        const liveScore = scoreData.latest_scores[cam.channel] ?? cam.latest_score;
        return {
          ...cam,
          latest_score: liveScore,
        };
      });
      setCameras(mergedCameras);

      // Detect new critical incidents for audio warning
      if (incList && incList.length > 0) {
        let hasNewCritical = false;
        incList.forEach((inc) => {
          if (!knownIncidentIds.current.has(inc.incident_id)) {
            knownIncidentIds.current.add(inc.incident_id);
            if (inc.severity_badge === 'CRITICAL' || inc.severity_score >= 75) {
              hasNewCritical = true;
            }
          }
        });

        if (hasNewCritical && !isAudioMuted) {
          soundFx.playAnomalyAlert();
        }
      }

      setIncidents(incList);
      if (digest) setDailyDigest(digest);
      if (streamStat) setStreamingStatus(streamStat);
      setLastUpdated(new Date());
    } catch (err) {
      console.warn('[SurveillanceContext] Polling cycle error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [isAudioMuted]);

  // Initial load and periodic polling
  useEffect(() => {
    refreshAll();
    // Fast polling for scores & status (every 2.5s)
    const intervalId = setInterval(() => {
      refreshAll();
    }, 2500);

    return () => clearInterval(intervalId);
  }, [refreshAll]);

  // Simulation handlers
  const simulateAnomaly = async (channel: number, score: number = 0.88) => {
    soundFx.playClick();
    await triggerSimulatedIncident(channel, score);
    await refreshAll();
  };

  const seedBaseline = async (channel: number, count: number = 15) => {
    soundFx.playClick();
    await seedBaselineVectors(channel, count);
    await refreshAll();
  };

  const activeIncidents = incidents.filter((i) => i.status === 'OPEN');

  return (
    <SurveillanceContext.Provider
      value={{
        health,
        cameras,
        incidents,
        activeIncidents,
        dailyDigest,
        streamingStatus,
        lastUpdated,
        isLoading,
        viewMode,
        setViewMode,
        selectedCamera,
        setSelectedCamera,
        inspectIncident,
        setInspectIncident,
        isCopilotOpen,
        setIsCopilotOpen,
        copilotTab,
        setCopilotTab,
        isGovernanceOpen,
        setIsGovernanceOpen,
        isAudioMuted,
        toggleAudioMute,
        incidentStatusFilter,
        setIncidentStatusFilter,
        incidentChannelFilter,
        setIncidentChannelFilter,
        refreshAll,
        simulateAnomaly,
        seedBaseline,
      }}
    >
      {children}
    </SurveillanceContext.Provider>
  );
};

export const useSurveillance = () => {
  const context = useContext(SurveillanceContext);
  if (!context) {
    throw new Error('useSurveillance must be used within a SurveillanceProvider');
  }
  return context;
};
