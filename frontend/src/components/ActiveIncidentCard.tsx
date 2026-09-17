"use client";

import React from "react";
import { formatTimestamp, severityColor } from "@/lib/api";
import type { IncidentRecord } from "@/lib/api";
import styles from "./ActiveIncidentCard.module.css";

interface ActiveIncidentCardProps {
  incident: IncidentRecord;
  onClick?: (incident: IncidentRecord) => void;
}

export default function ActiveIncidentCard({ incident, onClick }: ActiveIncidentCardProps) {
  const isCritical = incident.severity_badge === "CRITICAL";
  const summary =
    incident.vlm_explanation?.summary || `Anomaly detected on ${incident.camera_name}`;

  const badgeClass =
    incident.severity_badge === "CRITICAL"
      ? "badge-critical"
      : incident.severity_badge === "HIGH"
        ? "badge-high"
        : incident.severity_badge === "MODERATE"
          ? "badge-moderate"
          : "badge-low";

  return (
    <div
      className={`${styles.card} ${isCritical ? styles.critical : ""}`}
      onClick={() => onClick?.(incident)}
      role="button"
      tabIndex={0}
      aria-label={`Incident on ${incident.camera_name}, severity ${incident.severity}`}
    >
      <div
        className={styles.severityBar}
        style={{ background: severityColor(incident.severity_badge) }}
      />
      <div className={styles.body}>
        <div className={styles.header}>
          <span className={styles.cameraName}>{incident.camera_name}</span>
          <span className={styles.timestamp}>{formatTimestamp(incident.start_time)}</span>
        </div>
        <p className={styles.summary}>{summary}</p>
        <div className={styles.footer}>
          <span className={`badge ${badgeClass}`}>{incident.severity_badge}</span>
          <span className={styles.severity}>{incident.severity}/100</span>
          <span className={styles.channel}>CH{incident.channel}</span>
        </div>
      </div>
    </div>
  );
}
