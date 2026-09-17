"use client";

import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import { central, type CameraStreamInfo } from "@/lib/api";

/* ------------------------------------------------------------------ */
/* Context shape                                                      */
/* ------------------------------------------------------------------ */

interface CameraState {
  cameras: CameraStreamInfo[];
  loading: boolean;
  error: string | null;
  getCameraName: (channel: number) => string;
}

const CameraContext = createContext<CameraState>({
  cameras: [],
  loading: true,
  error: null,
  getCameraName: () => "Unknown",
});

export const useCameras = () => useContext(CameraContext);

/* ------------------------------------------------------------------ */
/* Provider                                                           */
/* ------------------------------------------------------------------ */

/** Default camera catalog used when backend is unreachable */
const DEFAULT_CAMERAS: CameraStreamInfo[] = Array.from({ length: 9 }, (_, i) => ({
  channel: i + 1,
  name: `Camera ${i + 1}`,
  status: "offline",
  substream_url: null,
  mainstream_url: null,
  snapshot_url: null,
  latest_score: null,
}));

export function CameraProvider({ children }: { children: React.ReactNode }) {
  const [cameras, setCameras] = useState<CameraStreamInfo[]>(DEFAULT_CAMERAS);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCameras = useCallback(async () => {
    try {
      const data = await central.cameras();
      setCameras(data.length > 0 ? data : DEFAULT_CAMERAS);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load cameras");
      setCameras(DEFAULT_CAMERAS);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCameras();
    // Re-fetch camera catalog every 30s (rarely changes)
    const iv = setInterval(fetchCameras, 30_000);
    return () => clearInterval(iv);
  }, [fetchCameras]);

  const getCameraName = useCallback(
    (channel: number) => {
      const cam = cameras.find((c) => c.channel === channel);
      return cam?.name || `Camera ${channel}`;
    },
    [cameras],
  );

  return (
    <CameraContext.Provider value={{ cameras, loading, error, getCameraName }}>
      {children}
    </CameraContext.Provider>
  );
}
