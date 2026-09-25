'use client';

import React from 'react';
import { useSurveillance } from '@/context/SurveillanceContext';
import { CameraPanel } from './CameraPanel';
import styles from './CameraGrid.module.css';

export const CameraGrid: React.FC = () => {
  const { cameras, viewMode, selectedCamera, setSelectedCamera } = useSurveillance();

  const activeCount = cameras.filter((c) => c.status === 'active' || c.is_connected).length;
  const primaryCam = cameras.find((c) => c.channel === selectedCamera) || cameras[0];
  const otherCams = cameras.filter((c) => c.channel !== primaryCam?.channel);

  return (
    <div className={styles.gridContainer}>
      {/* Grid Sub-Header Stats */}
      <div className={styles.gridToolbar}>
        <div className={styles.channelStats}>
          <span>
            SURVEILLANCE SECTORS:{' '}
            <span className={styles.statHighlight}>
              {activeCount} / {cameras.length} ONLINE
            </span>
          </span>
          <span style={{ color: 'var(--text-muted)' }}>•</span>
          <span>PROTOCOL: DIRECT MULTIPART HTTP (NO RTSP)</span>
          <span style={{ color: 'var(--text-muted)' }}>•</span>
          <span>SUB-STREAM: D1 @ 25 FPS</span>
        </div>
      </div>

      {/* Mode 1: 3x3 Tactical Matrix Grid */}
      {viewMode === 'matrix' && (
        <div className={styles.matrixGrid}>
          {cameras.map((camera) => (
            <CameraPanel
              key={camera.channel}
              camera={camera}
              onFocus={(ch) => setSelectedCamera(ch)}
            />
          ))}
        </div>
      )}

      {/* Mode 2: 1+8 Cinema Focus View */}
      {viewMode === 'focus' && primaryCam && (
        <div className={styles.focusLayout}>
          <div className={styles.primaryViewport}>
            <CameraPanel camera={primaryCam} isFocused={true} />
          </div>

          <div className={styles.thumbnailStrip}>
            {otherCams.map((camera) => (
              <CameraPanel
                key={camera.channel}
                camera={camera}
                onFocus={(ch) => setSelectedCamera(ch)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
