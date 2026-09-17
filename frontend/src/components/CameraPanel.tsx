"use client";

import React, { useCallback, useRef, useState } from "react";
import { edgeLiveStreamUrl, severityColor } from "@/lib/api";
import type { CameraStreamInfo } from "@/lib/api";
import styles from "./CameraPanel.module.css";

interface CameraPanelProps {
  camera: CameraStreamInfo;
  anomalyScore?: number;
  onClick?: (camera: CameraStreamInfo) => void;
}

export default function CameraPanel({ camera, anomalyScore = 0, onClick }: CameraPanelProps) {
  const [isOnline, setIsOnline] = useState(true);
  const imgRef = useRef<HTMLImageElement>(null);

  const handleError = useCallback(() => setIsOnline(false), []);
  const handleLoad = useCallback(() => setIsOnline(true), []);

  const score = anomalyScore ?? camera.latest_score ?? 0;
  const severity = score >= 70 ? "CRITICAL" : score >= 45 ? "HIGH" : score >= 20 ? "MODERATE" : "LOW";

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

      {/* Video stream or offline placeholder */}
      {isOnline ? (
        <img
          ref={imgRef}
          className={styles.stream}
          src={edgeLiveStreamUrl(camera.channel)}
          alt={`Live feed: ${camera.name}`}
          onError={handleError}
          onLoad={handleLoad}
        />
      ) : (
        <div className={styles.offline}>
          <div className={styles.offlineIcon}>⦿</div>
          <span>Stream Offline</span>
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
