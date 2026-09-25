'use client';

import React from 'react';
import { IncidentRecord } from '@/types';
import { useSurveillance } from '@/context/SurveillanceContext';
import { updateIncidentStatus, triggerIncidentExplanation } from '@/lib/api';
import { IconSparkles, IconCheck, IconZoom } from '@/components/Icons';
import { soundFx } from '@/lib/audio';
import styles from './ActiveIncidentCard.module.css';

interface ActiveIncidentCardProps {
  incident: IncidentRecord;
}

export const ActiveIncidentCard: React.FC<ActiveIncidentCardProps> = ({ incident }) => {
  const { setInspectIncident, refreshAll } = useSurveillance();

  const handleCardClick = () => {
    soundFx.playRadarPing();
    setInspectIncident(incident);
  };

  const handleAcknowledge = async (e: React.MouseEvent) => {
    e.stopPropagation();
    soundFx.playClick();
    await updateIncidentStatus(incident.incident_id, 'ACKNOWLEDGED');
    await refreshAll();
  };

  const handleExplain = async (e: React.MouseEvent) => {
    e.stopPropagation();
    soundFx.playClick();
    await triggerIncidentExplanation(incident.incident_id);
    await refreshAll();
  };

  // Format relative timestamp
  const formatTime = (ts: number) => {
    if (!ts) return 'Just now';
    const sec = Math.floor((Date.now() / 1000) - ts);
    if (sec < 60) return `${Math.max(sec, 1)}s ago`;
    const min = Math.floor(sec / 60);
    if (min < 60) return `${min}m ago`;
    const hr = Math.floor(min / 60);
    return `${hr}h ago`;
  };

  const badgeClass =
    incident.severity_badge === 'CRITICAL'
      ? styles.scoreCritical
      : incident.severity_badge === 'HIGH'
      ? styles.scoreHigh
      : incident.severity_badge === 'MEDIUM'
      ? styles.scoreMedium
      : styles.scoreLow;

  const severityCardClass =
    incident.severity_badge === 'CRITICAL'
      ? styles.critical
      : incident.severity_badge === 'HIGH'
      ? styles.high
      : incident.severity_badge === 'MEDIUM'
      ? styles.medium
      : styles.low;

  return (
    <div className={`${styles.card} ${severityCardClass}`} onClick={handleCardClick}>
      {/* Header: Channel, Time, Severity Badge */}
      <div className={styles.cardHeader}>
        <div className={styles.metaGroup}>
          <span className={styles.channelBadge}>
            CH-{String(incident.channel).padStart(2, '0')}
          </span>
          <span className={styles.timeLabel}>{formatTime(incident.start_time)}</span>
        </div>

        <div className={styles.severityScore}>
          <span className={`${styles.scorePill} ${badgeClass}`}>
            {incident.severity_badge} • {incident.severity_score}
          </span>
        </div>
      </div>

      {/* Body: Thumbnail + Title + VLM Summary */}
      <div className={styles.cardBody}>
        {/* Thumbnail fallback to camera snapshot proxy */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`http://localhost:7777/api/v1/stream/${incident.channel}/snapshot`}
          alt={`Incident on Channel ${incident.channel}`}
          className={styles.snapshotThumbnail}
          onError={(e) => {
            (e.currentTarget as HTMLElement).style.display = 'none';
          }}
        />

        <div className={styles.descriptionBox}>
          <span className={styles.cameraTitle}>
            {incident.camera_name || `Camera ${incident.channel}`}
          </span>
          <p className={styles.vlmSummary}>
            {incident.vlm_explanation?.summary ||
              `Anomaly detected with peak score ${incident.peak_score.toFixed(
                3
              )} exceeding triage threshold.`}
          </p>
          {incident.vlm_explanation && (
            <span className={styles.vlmChip}>
              <IconSparkles size={11} />
              PEGASUS VLM REASONED
            </span>
          )}
        </div>
      </div>

      {/* Footer: Status + Actions */}
      <div className={styles.cardFooter}>
        <span className={styles.statusTag}>STATUS: {incident.status}</span>

        <div className={styles.actionButtons}>
          {!incident.vlm_explanation && (
            <button
              className={styles.cardBtn}
              onClick={handleExplain}
              title="Generate Pegasus VLM natural-language scene breakdown"
            >
              <IconSparkles size={11} />
            </button>
          )}

          {incident.status === 'OPEN' && (
            <button
              className={styles.cardBtn}
              onClick={handleAcknowledge}
              title="Mark Incident Acknowledged"
            >
              <IconCheck size={11} />
              ACK
            </button>
          )}

          <button className={`${styles.cardBtn} ${styles.inspect}`} onClick={handleCardClick}>
            <IconZoom size={11} />
            INSPECT
          </button>
        </div>
      </div>
    </div>
  );
};
