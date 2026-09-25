'use client';

import React from 'react';
import { useSurveillance } from '@/context/SurveillanceContext';
import { ActiveIncidentCard } from './ActiveIncidentCard';
import { IconShield, IconSparkles } from '@/components/Icons';
import styles from './AlertQueue.module.css';

export const AlertQueue: React.FC = () => {
  const {
    incidents,
    activeIncidents,
    incidentStatusFilter,
    setIncidentStatusFilter,
    incidentChannelFilter,
    setIncidentChannelFilter,
    simulateAnomaly,
    selectedCamera,
  } = useSurveillance();

  // Filtered incidents
  const filteredIncidents = incidents.filter((inc) => {
    const statusMatch =
      incidentStatusFilter === 'ALL' || inc.status === incidentStatusFilter;
    const channelMatch =
      incidentChannelFilter === null || inc.channel === incidentChannelFilter;
    return statusMatch && channelMatch;
  });

  const activeCount = activeIncidents.length;

  return (
    <div className={styles.queueContainer}>
      {/* Header */}
      <div className={styles.queueHeader}>
        <div className={styles.titleRow}>
          <div className={styles.queueTitle}>
            <span>REAL-TIME THREAT STREAM</span>
            <span
              className={`${styles.badgeCount} ${
                activeCount === 0 ? styles.zero : ''
              }`}
            >
              {activeCount} ACTIVE
            </span>
          </div>
        </div>

        {/* Filters */}
        <div className={styles.filterBar}>
          <div className={styles.tabGroup}>
            {['ALL', 'OPEN', 'ACKNOWLEDGED', 'CLOSED'].map((tab) => (
              <button
                key={tab}
                className={`${styles.tabBtn} ${
                  incidentStatusFilter === tab ? styles.active : ''
                }`}
                onClick={() => setIncidentStatusFilter(tab)}
              >
                {tab}
              </button>
            ))}
          </div>

          <select
            className={styles.channelSelect}
            value={incidentChannelFilter ?? ''}
            onChange={(e) =>
              setIncidentChannelFilter(
                e.target.value === '' ? null : Number(e.target.value)
              )
            }
          >
            <option value="">ALL CHANNELS</option>
            {Array.from({ length: 9 }, (_, i) => i + 1).map((ch) => (
              <option key={ch} value={ch}>
                CH-{String(ch).padStart(2, '0')}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Incident List */}
      <div className={styles.incidentList}>
        {filteredIncidents.length > 0 ? (
          filteredIncidents.map((incident) => (
            <ActiveIncidentCard key={incident.incident_id} incident={incident} />
          ))
        ) : (
          <div className={styles.emptyState}>
            <div className={styles.secureIcon}>
              <IconShield size={26} />
            </div>
            <div className={styles.emptyText}>
              <span style={{ fontWeight: 600, color: '#fff' }}>
                ALL SECTORS SECURE
              </span>
              <span>Zero unacknowledged anomalies in the surveillance queue.</span>
            </div>
            <button
              className={styles.simulateBtn}
              onClick={() => simulateAnomaly(selectedCamera || 1, 0.88)}
            >
              <IconSparkles size={13} style={{ display: 'inline', marginRight: '4px' }} />
              Trigger Test Incident (Edge Anomaly)
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
