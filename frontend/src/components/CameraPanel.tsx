'use client';

import React, { useState, useEffect, useRef } from 'react';
import { CameraInfo } from '@/types';
import { useSurveillance } from '@/context/SurveillanceContext';
import { AnomalyMeter } from './AnomalyMeter';
import {
  IconCamera,
  IconZoom,
  IconDownload,
  IconRefresh,
  IconSparkles,
} from '@/components/Icons';
import { soundFx } from '@/lib/audio';
import styles from './CameraPanel.module.css';

interface CameraPanelProps {
  camera: CameraInfo;
  isFocused?: boolean;
  onFocus?: (channel: number) => void;
}

export const CameraPanel: React.FC<CameraPanelProps> = ({
  camera,
  isFocused = false,
  onFocus,
}) => {
  const { selectedCamera, setSelectedCamera, seedBaseline } = useSurveillance();
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const [isWsConnected, setIsWsConnected] = useState<boolean>(false);
  const [streamError, setStreamError] = useState<boolean>(false);
  const objectUrlRef = useRef<string | null>(null);

  const isSelected = selectedCamera === camera.channel;
  const isEscalated = camera.latest_score >= 0.08;

  // Primary: Low-latency full-duplex WebSocket stream (bypasses browser 6-connection HTTP limit)
  useEffect(() => {
    let ws: WebSocket | null = null;
    let isMounted = true;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connectWebSocket = () => {
      if (!isMounted) return;

      try {
        const edgeHost = process.env.NEXT_PUBLIC_EDGE_URL || 'http://localhost:7777';
        const wsUrl = edgeHost.replace(/^http/, 'ws') + `/api/v1/stream/${camera.channel}/ws?fps=12`;

        ws = new WebSocket(wsUrl);
        ws.binaryType = 'blob';

        ws.onopen = () => {
          if (!isMounted) return;
          setIsWsConnected(true);
          setStreamError(false);
        };

        ws.onmessage = (event) => {
          if (!isMounted) return;
          if (event.data instanceof Blob) {
            const nextUrl = URL.createObjectURL(event.data);
            if (objectUrlRef.current) {
              URL.revokeObjectURL(objectUrlRef.current);
            }
            objectUrlRef.current = nextUrl;
            setFrameUrl(nextUrl);
          }
        };

        ws.onerror = () => {
          if (isMounted) {
            setIsWsConnected(false);
          }
        };

        ws.onclose = () => {
          if (isMounted) {
            setIsWsConnected(false);
            reconnectTimer = setTimeout(connectWebSocket, 2000);
          }
        };
      } catch {
        if (isMounted) setIsWsConnected(false);
      }
    };

    connectWebSocket();

    return () => {
      isMounted = false;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) {
        ws.onclose = null;
        ws.onerror = null;
        ws.close();
      }
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
    };
  }, [camera.channel]);

  const handleSelect = () => {
    setSelectedCamera(camera.channel);
    soundFx.playRadarPing();
    if (onFocus) {
      onFocus(camera.channel);
    }
  };

  const handleRefresh = (e: React.MouseEvent) => {
    e.stopPropagation();
    soundFx.playClick();
    setStreamError(false);
  };

  const handleDownloadSnapshot = (e: React.MouseEvent) => {
    e.stopPropagation();
    soundFx.playClick();
    const link = document.createElement('a');
    link.href = `${camera.snapshot_url}?t=${Date.now()}`;
    link.download = `snapshot_ch${camera.channel}_${Date.now()}.jpg`;
    link.target = '_blank';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleSeed = (e: React.MouseEvent) => {
    e.stopPropagation();
    seedBaseline(camera.channel, 10);
  };

  // If WebSocket is delivering frames, use blob URL; otherwise fallback to snapshot/MJPEG
  const displaySrc = frameUrl || camera.substream_url;

  return (
    <div
      className={`${styles.cameraPanel} ${isSelected ? styles.selected : ''} ${
        isEscalated ? styles.escalated : ''
      }`}
      onClick={handleSelect}
    >
      {/* Top Tactical HUD Overlay */}
      <div className={styles.topHud}>
        <div className={styles.cameraMeta}>
          <span className={styles.channelTag}>CH-{String(camera.channel).padStart(2, '0')}</span>
          <span className={styles.channelName}>{camera.name}</span>
        </div>

        <div className={styles.telemetryBadge}>
          <div className={styles.liveDot} />
          <span>{camera.fps || 25} FPS</span>
          <span style={{ color: 'var(--text-muted)' }}>•</span>
          <span>{camera.latency_ms || 18}ms</span>
        </div>
      </div>

      {/* Video Viewport */}
      <div className={styles.videoContainer}>
        {!streamError && displaySrc ? (
          /* eslint-disable-next-line @next/next/no-img-element */
          <img
            src={displaySrc}
            alt={`Live Feed ${camera.channel} - ${camera.name}`}
            className={styles.liveVideo}
            onError={() => {
              if (!isWsConnected) setStreamError(true);
            }}
          />
        ) : (
          <div className={styles.fallbackOverlay}>
            <IconCamera size={28} style={{ color: 'var(--text-muted)' }} />
            <span>CONNECTING TO DIRECT STREAM...</span>
            <button
              className={styles.iconBtn}
              onClick={handleRefresh}
              title="Retry Stream Connection"
            >
              <IconRefresh size={14} />
            </button>
          </div>
        )}
      </div>

      {/* Bottom Tactical HUD Overlay */}
      <div className={styles.bottomHud}>
        <div className={styles.meterContainer}>
          <AnomalyMeter score={camera.latest_score} threshold={0.08} size="sm" showLabel={true} />
        </div>

        <div className={styles.actionToolbar}>
          <button
            className={styles.iconBtn}
            onClick={handleDownloadSnapshot}
            title="Download Instant Hi-Res Snapshot"
          >
            <IconDownload size={13} />
          </button>
          <button
            className={styles.iconBtn}
            onClick={handleSeed}
            title="Seed Baseline Reference Vectors (Qdrant Edge)"
          >
            <IconSparkles size={13} />
          </button>
          <button
            className={styles.iconBtn}
            onClick={(e) => {
              e.stopPropagation();
              handleSelect();
            }}
            title="Focus Camera Feed"
          >
            <IconZoom size={13} />
          </button>
        </div>
      </div>
    </div>
  );
};
