"use client";

import React, { useEffect, useRef, useState } from "react";
import { edgeWsStreamUrl, edgeSnapshotUrl, severityColor } from "@/lib/api";
import type { CameraStreamInfo } from "@/lib/api";
import styles from "./CameraPanel.module.css";

interface CameraPanelProps {
  camera: CameraStreamInfo;
  anomalyScore?: number;
  onClick?: (camera: CameraStreamInfo) => void;
}

export default function CameraPanel({ camera, anomalyScore = 0, onClick }: CameraPanelProps) {
  const [isOnline, setIsOnline] = useState(true);
  const [hasFrame, setHasFrame] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const score = anomalyScore ?? camera.latest_score ?? 0;
  const severity = score >= 70 ? "CRITICAL" : score >= 45 ? "HIGH" : score >= 20 ? "MODERATE" : "LOW";

  useEffect(() => {
    let active = true;
    let ws: WebSocket | null = null;
    let pollTimer: ReturnType<typeof setTimeout> | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let isWsStreaming = false;

    // Draw frame onto canvas
    const drawFrame = async (blob: Blob) => {
      if (!active) return;
      try {
        if (typeof createImageBitmap === "function") {
          const bitmap = await createImageBitmap(blob);
          if (!active) {
            bitmap.close();
            return;
          }
          const canvas = canvasRef.current;
          if (canvas) {
            if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
              canvas.width = bitmap.width;
              canvas.height = bitmap.height;
            }
            const ctx = canvas.getContext("2d");
            if (ctx) {
              ctx.drawImage(bitmap, 0, 0);
            }
          }
          bitmap.close();
        } else {
          // Fallback for environments without createImageBitmap
          const url = URL.createObjectURL(blob);
          const img = new Image();
          img.onload = () => {
            if (active && canvasRef.current) {
              const canvas = canvasRef.current;
              if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
              }
              const ctx = canvas.getContext("2d");
              if (ctx) ctx.drawImage(img, 0, 0);
            }
            URL.revokeObjectURL(url);
          };
          img.src = url;
        }

        if (active) {
          setIsOnline(true);
          setHasFrame(true);
        }
      } catch {
        // frame decode error, ignore individual frame drops
      }
    };

    // Snapshot polling fallback when WebSocket is unavailable
    const pollSnapshot = async () => {
      if (!active || isWsStreaming) return;
      try {
        const snapUrl = `${edgeSnapshotUrl(camera.channel)}?t=${Date.now()}`;
        const res = await fetch(snapUrl, { cache: "no-store" });
        if (!res.ok) throw new Error("Snapshot failed");
        const blob = await res.blob();
        await drawFrame(blob);
        if (active && !isWsStreaming) {
          pollTimer = setTimeout(pollSnapshot, 120); // ~8 FPS
        }
      } catch {
        if (active && !isWsStreaming) {
          setIsOnline(false);
          pollTimer = setTimeout(pollSnapshot, 2000); // retry in 2s
        }
      }
    };

    // WebSocket connection
    const connectWs = () => {
      if (!active) return;
      try {
        const wsUrl = edgeWsStreamUrl(camera.channel);
        ws = new WebSocket(wsUrl);
        ws.binaryType = "blob";

        ws.onopen = () => {
          if (!active) return;
          isWsStreaming = true;
          if (pollTimer) {
            clearTimeout(pollTimer);
            pollTimer = null;
          }
        };

        ws.onmessage = (event) => {
          if (!active) return;
          const blob =
            event.data instanceof Blob
              ? event.data
              : new Blob([event.data], { type: "image/jpeg" });
          drawFrame(blob);
        };

        ws.onerror = () => {
          if (!active) return;
          isWsStreaming = false;
          // Trigger snapshot fallback if not already running
          if (!pollTimer) {
            pollTimer = setTimeout(pollSnapshot, 100);
          }
        };

        ws.onclose = () => {
          if (!active) return;
          isWsStreaming = false;
          // Trigger snapshot fallback
          if (!pollTimer) {
            pollTimer = setTimeout(pollSnapshot, 100);
          }
          // Attempt to reconnect WS in 4 seconds
          reconnectTimer = setTimeout(connectWs, 4000);
        };
      } catch {
        isWsStreaming = false;
        if (!pollTimer) {
          pollTimer = setTimeout(pollSnapshot, 100);
        }
      }
    };

    connectWs();

    return () => {
      active = false;
      if (ws) {
        ws.close();
        ws = null;
      }
      if (pollTimer) clearTimeout(pollTimer);
      if (reconnectTimer) clearTimeout(reconnectTimer);
    };
  }, [camera.channel]);

  const statusClass = isOnline
    ? score >= 45
      ? styles.dotWarning
      : styles.dotOnline
    : styles.dotOffline;

  return (
    <div
      className={styles.panel}
      onClick={() => onClick?.(camera)}
      role="button"
      tabIndex={0}
      aria-label={`Camera ${camera.channel}: ${camera.name}`}
    >
      {/* Top overlay */}
      <div className={styles.overlay}>
        <span className={styles.cameraName}>
          <span className={`${styles.statusDot} ${statusClass}`} />
          {camera.name}
        </span>
        <span className={styles.channel}>CH{camera.channel}</span>
      </div>

      {/* Video stream canvas */}
      <canvas
        ref={canvasRef}
        className={styles.stream}
        style={{ display: isOnline && hasFrame ? "block" : "none" }}
      />

      {/* Offline / connecting placeholder */}
      {(!isOnline || !hasFrame) && (
        <div className={styles.offline}>
          <div className={styles.offlineIcon}>⦿</div>
          <span>{isOnline ? "Connecting..." : "Stream Offline"}</span>
        </div>
      )}

      {/* Bottom bar */}
      <div className={styles.bottomBar}>
        <div className={styles.liveTag}>
          <span className={styles.liveDot} />
          LIVE
        </div>
        <div className={styles.scoreBar}>
          <div className={styles.scoreTrack}>
            <div
              className={styles.scoreFill}
              style={{
                width: `${Math.min(score, 100)}%`,
                background: severityColor(severity),
              }}
            />
          </div>
          <span className={styles.scoreLabel}>{Math.round(score)}</span>
        </div>
      </div>
    </div>
  );
}
