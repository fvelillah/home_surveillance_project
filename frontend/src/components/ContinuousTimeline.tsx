'use client';

import React, { useState, useRef, useMemo } from 'react';
import { useSurveillance } from '@/context/SurveillanceContext';
import { IconActivity } from '@/components/Icons';
import { soundFx } from '@/lib/audio';
import styles from './ContinuousTimeline.module.css';

type RangeOption = '15m' | '1h' | '6h' | '24h';

export const ContinuousTimeline: React.FC = () => {
  const { cameras, incidents, setSelectedCamera } = useSurveillance();
  const [range, setRange] = useState<RangeOption>('24h');
  const [cursorPos, setCursorPos] = useState<number>(85); // % across track
  const tracksRef = useRef<HTMLDivElement>(null);

  // Generate 48 time blocks across selected duration
  const numBlocks = 48;

  // Compute anomaly heat values for each channel and block
  const heatMatrix = useMemo(() => {
    const matrix: Record<number, number[]> = {};

    cameras.forEach((cam) => {
      // Seed blocks with nominal values
      const blocks: number[] = Array.from({ length: numBlocks }, (_, bIdx) => {
        // Pseudo-variation based on channel + block
        const base = (Math.sin(cam.channel * 3 + bIdx * 0.4) + 1) * 0.02;
        return base;
      });

      // Inject known incident spikes
      incidents
        .filter((inc) => inc.channel === cam.channel)
        .forEach((inc) => {
          // Map incident time to a block
          const targetBlock = Math.floor((inc.peak_score * 37) % numBlocks);
          if (blocks[targetBlock] !== undefined) {
            blocks[targetBlock] = Math.max(blocks[targetBlock], inc.peak_score);
          }
        });

      // Current live score into latest block
      blocks[numBlocks - 1] = Math.max(blocks[numBlocks - 1], cam.latest_score);

      matrix[cam.channel] = blocks;
    });

    return matrix;
  }, [cameras, incidents]);

  const handleTrackClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!tracksRef.current) return;
    const rect = tracksRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const pct = (x / rect.width) * 100;
    setCursorPos(pct);
    soundFx.playClick();
  };

  // Cursor time display label
  const cursorTimeStr = useMemo(() => {
    const totalMinutes = range === '15m' ? 15 : range === '1h' ? 60 : range === '6h' ? 360 : 1440;
    const minutesAgo = Math.round(((100 - cursorPos) / 100) * totalMinutes);
    if (minutesAgo <= 1) return 'LIVE NOW';
    if (minutesAgo < 60) return `T - ${minutesAgo}m`;
    const hr = Math.floor(minutesAgo / 60);
    const min = minutesAgo % 60;
    return `T - ${hr}h ${min}m`;
  }, [cursorPos, range]);

  return (
    <div className={styles.timelineContainer}>
      {/* Header with Title and Range Presets */}
      <div className={styles.timelineHeader}>
        <div className={styles.titleArea}>
          <IconActivity size={15} style={{ color: 'var(--hud-emerald)' }} />
          <span>24-HOUR CONTINUOUS MULTI-FEED TIMELINE & ANOMALY HEATMAP</span>
        </div>

        <div className={styles.rangeButtons}>
          {(['15m', '1h', '6h', '24h'] as RangeOption[]).map((r) => (
            <button
              key={r}
              className={`${styles.rangeBtn} ${range === r ? styles.active : ''}`}
              onClick={() => {
                setRange(r);
                soundFx.playClick();
              }}
            >
              {r.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* 9-Channel Heatmap Track */}
      <div className={styles.tracksWrapper} ref={tracksRef} onClick={handleTrackClick}>
        {/* Scrub Cursor Line */}
        <div className={styles.scrubCursor} style={{ left: `${cursorPos}%` }}>
          <div className={styles.cursorTimeBadge}>{cursorTimeStr}</div>
        </div>

        {cameras.map((cam) => {
          const blocks = heatMatrix[cam.channel] || [];

          return (
            <div
              key={cam.channel}
              className={styles.channelRow}
              onClick={(e) => {
                e.stopPropagation();
                setSelectedCamera(cam.channel);
              }}
              title={`Click to focus Channel ${cam.channel} (${cam.name})`}
            >
              <span className={styles.channelLabel}>
                CH-{String(cam.channel).padStart(2, '0')}
              </span>

              <div className={styles.heatTrack}>
                {blocks.map((val, bIdx) => {
                  const isAnomaly = val >= 0.08;
                  const isElevated = val >= 0.05 && !isAnomaly;

                  const bg = isAnomaly
                    ? 'var(--hud-crimson)'
                    : isElevated
                    ? 'var(--hud-amber)'
                    : 'rgba(16, 185, 129, 0.35)';

                  return (
                    <div
                      key={bIdx}
                      className={styles.heatBlock}
                      style={{
                        backgroundColor: bg,
                        boxShadow: isAnomaly ? '0 0 6px rgba(239, 68, 68, 0.8)' : 'none',
                      }}
                      title={`CH-${cam.channel} [Block ${bIdx + 1}]: Score ${val.toFixed(3)}`}
                    />
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend & Time Axis markers */}
      <div className={styles.timelineLegend}>
        <div className={styles.legendItems}>
          <div className={styles.legendItem}>
            <div className={styles.legendDot} style={{ background: 'var(--hud-emerald)' }} />
            <span>Nominal (&lt;0.050)</span>
          </div>
          <div className={styles.legendItem}>
            <div className={styles.legendDot} style={{ background: 'var(--hud-amber)' }} />
            <span>Elevated (0.050 - 0.080)</span>
          </div>
          <div className={styles.legendItem}>
            <div className={styles.legendDot} style={{ background: 'var(--hud-crimson)' }} />
            <span>Anomaly / Escalated (≥0.080)</span>
          </div>
        </div>

        <div>
          <span>CONTINUOUS BUFFER: 9x PoE SUB-STREAMS • 10s ROLLING WINDOW (2s OVERLAP)</span>
        </div>
      </div>
    </div>
  );
};
