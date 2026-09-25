'use client';

import React from 'react';
import { useSurveillance } from '@/context/SurveillanceContext';
import {
  IconRadar,
  IconSparkles,
  IconFileText,
  IconSettings,
  IconBell,
  IconBellOff,
  IconGrid,
  IconZoom,
  IconActivity,
} from '@/components/Icons';
import styles from './CommandHeader.module.css';

export const CommandHeader: React.FC = () => {
  const {
    health,
    dailyDigest,
    viewMode,
    setViewMode,
    setIsCopilotOpen,
    setCopilotTab,
    setIsGovernanceOpen,
    isAudioMuted,
    toggleAudioMute,
    simulateAnomaly,
    selectedCamera,
  } = useSurveillance();

  const threatLevel = dailyDigest?.threat_level || 'NORMAL';

  const threatClass =
    threatLevel === 'CRITICAL' || threatLevel === 'HIGH'
      ? styles.threatCritical
      : threatLevel === 'ELEVATED'
      ? styles.threatElevated
      : styles.threatNormal;

  return (
    <header className={styles.header}>
      {/* Brand & System Tier */}
      <div className={styles.brandSection}>
        <div className={styles.logoIcon}>
          <IconRadar size={24} className="radar-sweep" />
        </div>
        <div className={styles.titleArea}>
          <div className={styles.systemTitle}>
            CYBER-DEFENSE SURVEILLANCE MATRIX
            <span className={styles.versionBadge}>DIRECT HTTP ENGINE</span>
          </div>
          <div className={styles.subtitle}>
            Dahua DH-NVR5216 • 9-PoE Sub-Stream Cluster • Pegasus VLM
          </div>
        </div>
      </div>

      {/* Cluster Health Matrix */}
      <div className={styles.clusterStatusGrid}>
        <div className={styles.statusNode} title="Central Cloud Analytics Backend (Port 9876)">
          <div className={`${styles.nodeDot} ${health?.status !== 'offline' ? styles.active : styles.offline}`} />
          <span>BACKEND 9876</span>
        </div>
        <div className={styles.divider} />
        <div className={styles.statusNode} title="Edge Ingestion Node (Port 7777)">
          <div className={`${styles.nodeDot} ${health?.edge_node_connected ? styles.active : styles.warning}`} />
          <span>EDGE 7777</span>
        </div>
        <div className={styles.divider} />
        <div className={styles.statusNode} title="Central Qdrant Baseline Vector Cluster">
          <div className={`${styles.nodeDot} ${health?.qdrant_connected ? styles.active : styles.offline}`} />
          <span>QDRANT CLOUD</span>
        </div>
        <div className={styles.divider} />
        <div className={styles.statusNode} title="Twelve Labs Pegasus v1.5 VLM Reasoning Engine">
          <div className={`${styles.nodeDot} ${health?.twelve_labs_enabled ? styles.active : styles.warning}`} />
          <span>PEGASUS VLM</span>
        </div>
        <div className={styles.divider} />
        <div className={styles.statusNode} title="Decoded Ingest Video Stream FPS">
          <IconActivity size={14} style={{ color: 'var(--hud-emerald)' }} />
          <span>25 FPS</span>
        </div>
      </div>

      {/* DEFCON Threat Barometer */}
      <div className={styles.threatMeter}>
        <span style={{ color: 'var(--text-muted)' }}>THREAT STATUS:</span>
        <span className={`${styles.threatBadge} ${threatClass}`}>{threatLevel}</span>
      </div>

      {/* View Switcher & Action Commands */}
      <div className={styles.actionGroup}>
        {/* View mode toggle */}
        <div className={styles.viewModeToggle}>
          <button
            className={`${styles.viewBtn} ${viewMode === 'matrix' ? styles.active : ''}`}
            onClick={() => setViewMode('matrix')}
            title="3x3 Tactical Matrix Grid"
          >
            <IconGrid size={14} />
            <span>3x3</span>
          </button>
          <button
            className={`${styles.viewBtn} ${viewMode === 'focus' ? styles.active : ''}`}
            onClick={() => setViewMode('focus')}
            title="1+8 Focus Cinema View"
          >
            <IconZoom size={14} />
            <span>FOCUS</span>
          </button>
        </div>

        {/* Audio Mute */}
        <button
          className={`${styles.audioBtn} ${isAudioMuted ? styles.muted : ''}`}
          onClick={toggleAudioMute}
          title={isAudioMuted ? 'Unmute Audio Alarms' : 'Mute Audio Alarms'}
        >
          {isAudioMuted ? <IconBellOff size={16} /> : <IconBell size={16} />}
        </button>

        {/* AI Copilot Drawer Trigger */}
        <button
          className={`${styles.hudButton} ${styles.copilotButton}`}
          onClick={() => {
            setCopilotTab('copilot');
            setIsCopilotOpen(true);
          }}
          title="Open AI Security Copilot & Semantic Search"
        >
          <IconSparkles size={15} />
          <span>AI COPILOT</span>
        </button>

        {/* Daily Digest Trigger */}
        <button
          className={styles.hudButton}
          onClick={() => {
            setCopilotTab('digest');
            setIsCopilotOpen(true);
          }}
          title="View 24h Daily Surveillance Digest"
        >
          <IconFileText size={15} />
          <span>DAILY DIGEST</span>
        </button>

        {/* System Governance Trigger */}
        <button
          className={styles.hudButton}
          onClick={() => setIsGovernanceOpen(true)}
          title="Memory Governor & Load Shedding Controls"
        >
          <IconSettings size={15} />
        </button>

        {/* Test Anomaly Trigger */}
        <button
          className={styles.hudButton}
          style={{ borderColor: 'rgba(239, 68, 68, 0.4)', color: 'var(--hud-crimson)' }}
          onClick={() => simulateAnomaly(selectedCamera || 1, 0.89)}
          title="Inject synthetic test incident for verification"
        >
          <span>SIMULATE</span>
        </button>
      </div>
    </header>
  );
};
