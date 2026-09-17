"use client";

import React, { useCallback } from "react";
import { useApp } from "@/context/AppContext";
import {
  central,
  formatTimestamp,
  formatDate,
  formatDuration,
  severityColor,
} from "@/lib/api";
import AnomalyMeter from "./AnomalyMeter";
import styles from "./IncidentModal.module.css";

export default function IncidentModal() {
  const { selectedIncident, setSelectedIncident, refresh } = useApp();

  const handleClose = useCallback(() => setSelectedIncident(null), [setSelectedIncident]);

  const handleStatus = useCallback(
    async (status: string) => {
      if (!selectedIncident) return;
      try {
        await central.updateIncidentStatus(selectedIncident.incident_id, status);
        refresh();
        handleClose();
      } catch (err) {
        console.error("Failed to update status:", err);
      }
    },
    [selectedIncident, refresh, handleClose],
  );

  if (!selectedIncident) return null;
  const inc = selectedIncident;
  const vlm = inc.vlm_explanation;

  return (
    <div className={styles.overlay} onClick={handleClose}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            <span
              className={`badge badge-${inc.severity_badge.toLowerCase()}`}
            >
              {inc.severity_badge}
            </span>
            <span className={styles.cameraName}>{inc.camera_name}</span>
            <span className={styles.channel}>CH{inc.channel}</span>
          </div>
          <button className={styles.closeBtn} onClick={handleClose} aria-label="Close">
            ✕
          </button>
        </div>

        {/* Body */}
        <div className={styles.body}>
          {/* Score row */}
          <div className={styles.scoreRow}>
            <AnomalyMeter score={inc.severity} size={80} label="Severity" />
            <div className={styles.scoreItem}>
              <span className={styles.scoreValue} style={{ color: severityColor(inc.severity_badge) }}>
                {inc.peak_score.toFixed(3)}
              </span>
              <span className={styles.scoreCaption}>Peak Score</span>
            </div>
            <div className={styles.scoreItem}>
              <span className={styles.scoreValue}>{inc.mean_score.toFixed(3)}</span>
              <span className={styles.scoreCaption}>Mean Score</span>
            </div>
            <div className={styles.scoreItem}>
              <span className={styles.scoreValue}>{inc.event_count}</span>
              <span className={styles.scoreCaption}>Events</span>
            </div>
          </div>

          {/* Meta info */}
          <div className={styles.metaRow}>
            <span className={styles.metaItem}>
              <span className={styles.metaLabel}>Start:</span>
              <span className={styles.metaValue}>
                {formatDate(inc.start_time)} {formatTimestamp(inc.start_time)}
              </span>
            </span>
            <span className={styles.metaItem}>
              <span className={styles.metaLabel}>Duration:</span>
              <span className={styles.metaValue}>{formatDuration(inc.duration_s)}</span>
            </span>
            <span className={styles.metaItem}>
              <span className={styles.metaLabel}>Status:</span>
              <span className={styles.metaValue}>{inc.status}</span>
            </span>
            <span className={styles.metaItem}>
              <span className={styles.metaLabel}>ID:</span>
              <span className={styles.metaValue}>{inc.incident_id.slice(0, 8)}</span>
            </span>
          </div>

          {/* VLM explanation */}
          {vlm && (
            <div className={styles.vlmSection}>
              <div className={styles.vlmTitle}>🧠 AI Scene Analysis</div>
              <p className={styles.vlmSummary}>{vlm.summary}</p>
              <div className={styles.vlmGrid}>
                {vlm.actors.length > 0 && (
                  <div className={styles.vlmField}>
                    <span className={styles.vlmLabel}>Actors</span>
                    <span className={styles.vlmValue}>{vlm.actors.join(", ")}</span>
                  </div>
                )}
                {vlm.action && (
                  <div className={styles.vlmField}>
                    <span className={styles.vlmLabel}>Action</span>
                    <span className={styles.vlmValue}>{vlm.action}</span>
                  </div>
                )}
                {vlm.objects.length > 0 && (
                  <div className={styles.vlmField}>
                    <span className={styles.vlmLabel}>Objects</span>
                    <span className={styles.vlmValue}>{vlm.objects.join(", ")}</span>
                  </div>
                )}
                <div className={styles.vlmField}>
                  <span className={styles.vlmLabel}>Risk Assessment</span>
                  <span className={styles.vlmValue} style={{ textTransform: "capitalize" }}>
                    {vlm.risk_assessment}
                  </span>
                </div>
                {vlm.recommended_action && (
                  <div className={styles.vlmField}>
                    <span className={styles.vlmLabel}>Recommended</span>
                    <span className={styles.vlmValue}>{vlm.recommended_action}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Events timeline */}
          {inc.events.length > 0 && (
            <div className={styles.eventsSection}>
              <div className={styles.eventsTitle}>📊 Event Timeline ({inc.events.length} events)</div>
              {inc.events.slice(0, 12).map((ev) => {
                const pct = Math.min(ev.ensemble_score * 100 * 5, 100); // scale for visibility
                const badge =
                  ev.ensemble_score > 0.15
                    ? "CRITICAL"
                    : ev.ensemble_score > 0.08
                      ? "HIGH"
                      : ev.ensemble_score > 0.04
                        ? "MODERATE"
                        : "LOW";
                return (
                  <div key={ev.event_id} className={styles.eventRow}>
                    <span className={styles.eventTime}>{formatTimestamp(ev.timestamp)}</span>
                    <span
                      className={styles.eventScore}
                      style={{ color: severityColor(badge) }}
                    >
                      {ev.ensemble_score.toFixed(3)}
                    </span>
                    <div className={styles.eventBar}>
                      <div
                        className={styles.eventBarFill}
                        style={{
                          width: `${pct}%`,
                          background: severityColor(badge),
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Action buttons */}
          {inc.status === "OPEN" && (
            <div className={styles.actions}>
              <button
                className={`${styles.actionBtn} ${styles.btnAcknowledge}`}
                onClick={() => handleStatus("ACKNOWLEDGED")}
              >
                ✓ Acknowledge
              </button>
              <button
                className={`${styles.actionBtn} ${styles.btnClose}`}
                onClick={() => handleStatus("CLOSED")}
              >
                ✕ Close
              </button>
              <button
                className={`${styles.actionBtn} ${styles.btnArchive}`}
                onClick={() => handleStatus("ARCHIVED")}
              >
                📦 Archive
              </button>
            </div>
          )}
          {inc.status === "ACKNOWLEDGED" && (
            <div className={styles.actions}>
              <button
                className={`${styles.actionBtn} ${styles.btnClose}`}
                onClick={() => handleStatus("CLOSED")}
              >
                ✕ Close Incident
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
