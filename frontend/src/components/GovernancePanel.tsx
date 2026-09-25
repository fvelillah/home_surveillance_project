'use client';

import React, { useState, useEffect } from 'react';
import { useSurveillance } from '@/context/SurveillanceContext';
import {
  setStreamingLevel,
  fetchQuarantineList,
  promoteQuarantinedVectors,
  triggerRetentionScrub,
} from '@/lib/api';
import { QuarantineItem } from '@/types';
import { IconClose, IconSliders, IconShield, IconRefresh } from '@/components/Icons';
import { soundFx } from '@/lib/audio';
import styles from './GovernancePanel.module.css';

export const GovernancePanel: React.FC = () => {
  const { isGovernanceOpen, setIsGovernanceOpen, streamingStatus, refreshAll } = useSurveillance();
  const [quarantineItems, setQuarantineItems] = useState<QuarantineItem[]>([]);
  const [isPromoting, setIsPromoting] = useState(false);
  const [isScrubbing, setIsScrubbing] = useState(false);

  useEffect(() => {
    if (isGovernanceOpen) {
      fetchQuarantineList().then((items) => setQuarantineItems(items));
    }
  }, [isGovernanceOpen]);

  if (!isGovernanceOpen) return null;

  const handleLevelChange = async (lvl: number) => {
    soundFx.playClick();
    await setStreamingLevel(lvl, streamingStatus?.auto_mode ?? true);
    await refreshAll();
  };

  const handleToggleAuto = async () => {
    soundFx.playClick();
    await setStreamingLevel(
      streamingStatus?.current_level ?? 0,
      !streamingStatus?.auto_mode
    );
    await refreshAll();
  };

  const handlePromoteAll = async () => {
    setIsPromoting(true);
    soundFx.playClick();
    const ids = quarantineItems.map((q) => q.item_id);
    await promoteQuarantinedVectors(ids, true);
    const updated = await fetchQuarantineList();
    setQuarantineItems(updated);
    setIsPromoting(false);
  };

  const handleScrub = async () => {
    setIsScrubbing(true);
    soundFx.playClick();
    await triggerRetentionScrub(7);
    const updated = await fetchQuarantineList();
    setQuarantineItems(updated);
    setIsScrubbing(false);
  };

  return (
    <div className={styles.overlay} onClick={() => setIsGovernanceOpen(false)}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <div className={styles.title}>
            <IconSliders size={18} style={{ color: 'var(--hud-cyan)' }} />
            <span>EDGE & MEMORY GOVERNANCE CONTROL</span>
          </div>
          <button
            className={styles.closeBtn}
            onClick={() => setIsGovernanceOpen(false)}
            title="Close"
          >
            <IconClose size={16} />
          </button>
        </div>

        <div className={styles.body}>
          {/* Section 1: Adaptive Streaming & Load Shedding */}
          <div className={styles.sectionBox}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className={styles.sectionTitle}>
                ADAPTIVE STREAMING BACKPRESSURE & LOAD SHEDDING (PHASE 4.4)
              </span>
              <button
                style={{
                  background: streamingStatus?.auto_mode
                    ? 'rgba(16, 185, 129, 0.2)'
                    : 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.15)',
                  color: streamingStatus?.auto_mode ? 'var(--hud-emerald)' : '#fff',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.68rem',
                  padding: '3px 8px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                }}
                onClick={handleToggleAuto}
              >
                AUTO-ADAPT: {streamingStatus?.auto_mode ? 'ENABLED' : 'MANUAL'}
              </button>
            </div>

            <div className={styles.levelGrid}>
              {[
                { lvl: 0, name: 'LEVEL 0: NORMAL', desc: 'Full MJPEG + AI scoring' },
                { lvl: 1, name: 'LEVEL 1: SCORE_ONLY', desc: 'Throttle stream FPS' },
                { lvl: 2, name: 'LEVEL 2: PASSTHROUGH', desc: 'Skip cloud kNN' },
                { lvl: 3, name: 'LEVEL 3: SHED_LOAD', desc: 'Drop non-critical frames' },
              ].map((item) => (
                <button
                  key={item.lvl}
                  className={`${styles.levelBtn} ${
                    streamingStatus?.current_level === item.lvl ? styles.active : ''
                  }`}
                  onClick={() => handleLevelChange(item.lvl)}
                >
                  <span style={{ fontWeight: 700 }}>{item.name}</span>
                  <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)' }}>
                    {item.desc}
                  </span>
                </button>
              ))}
            </div>

            <div className={styles.telemetryGrid}>
              <div className={styles.telemetryCard}>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>CPU USAGE</span>
                <span style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fff' }}>
                  {streamingStatus?.cpu_percent.toFixed(1)}%
                </span>
              </div>
              <div className={styles.telemetryCard}>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                  QUEUE LATENCY
                </span>
                <span style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fff' }}>
                  {streamingStatus?.queue_latency_ms.toFixed(1)} ms
                </span>
              </div>
              <div className={styles.telemetryCard}>
                <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                  FRAMES SHED
                </span>
                <span style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fff' }}>
                  {streamingStatus?.frames_dropped_total || 0}
                </span>
              </div>
            </div>
          </div>

          {/* Section 2: Memory Governor & Anti-Poisoning Quarantine */}
          <div className={styles.sectionBox}>
            <span className={styles.sectionTitle}>
              OBSERVATION QUARANTINE BUFFER & RETENTION SCRUB (PHASE 4.3)
            </span>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Candidate baseline vectors are held in a 1-hour observation quarantine to prevent
              environmental drift and model poisoning before promotion to Central Qdrant.
            </p>

            <div className={styles.actionRow}>
              <button
                className={styles.actionBtn}
                onClick={handlePromoteAll}
                disabled={isPromoting}
              >
                {isPromoting ? 'PROMOTING...' : 'PROMOTE MATURED VECTORS'}
              </button>
              <button
                className={styles.actionBtn}
                style={{ borderColor: 'rgba(239, 68, 68, 0.4)', color: '#fca5a5' }}
                onClick={handleScrub}
                disabled={isScrubbing}
              >
                {isScrubbing ? 'SCRUBBING...' : 'RUN 7-DAY RETENTION SCRUB'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
