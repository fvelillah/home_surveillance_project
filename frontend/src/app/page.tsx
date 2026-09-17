"use client";

import React, { useEffect, useState } from "react";
import { useApp } from "@/context/AppContext";
import CameraGrid from "@/components/CameraGrid";
import AlertQueue from "@/components/AlertQueue";
import IncidentModal from "@/components/IncidentModal";
import ContinuousTimeline from "@/components/ContinuousTimeline";
import OpsCopilot from "@/components/OpsCopilot";
import styles from "./page.module.css";

export default function Dashboard() {
  const { health, loading, error, digestSummary } = useApp();
  const [clock, setClock] = useState("");

  // Live clock
  useEffect(() => {
    const tick = () =>
      setClock(
        new Date().toLocaleTimeString("en-US", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        }),
      );
    tick();
    const iv = setInterval(tick, 1000);
    return () => clearInterval(iv);
  }, []);

  // Loading screen
  if (loading && !health) {
    return (
      <div className={styles.loadingScreen}>
        <div className={styles.loadingSpinner} />
        <span className={styles.loadingText}>Connecting to surveillance backend...</span>
      </div>
    );
  }

  // Determine system status
  const isOnline = !!health;
  const threatLevel = digestSummary?.threat_level || "LOW";
  const threatColor =
    threatLevel === "SEVERE"
      ? "var(--severity-critical)"
      : threatLevel === "ELEVATED"
        ? "var(--severity-high)"
        : threatLevel === "MODERATE"
          ? "var(--severity-moderate)"
          : "var(--severity-low)";
  const threatBg =
    threatLevel === "SEVERE"
      ? "var(--severity-critical-bg)"
      : threatLevel === "ELEVATED"
        ? "var(--severity-high-bg)"
        : threatLevel === "MODERATE"
          ? "var(--severity-moderate-bg)"
          : "var(--severity-low-bg)";

  return (
    <div className={styles.dashboard}>
      {/* ---- Top Bar ---- */}
      <header className={styles.topBar}>
        <div className={styles.brandRow}>
          <span className={styles.logo}>◉ SENTINEL</span>
          <span className={styles.subtitle}>Home Surveillance Console</span>
        </div>

        <div className={styles.statusRow}>
          {/* Threat level */}
          <span
            className={styles.threatBadge}
            style={{ background: threatBg, color: threatColor }}
          >
            {threatLevel}
          </span>

          {/* System health */}
          <span className={styles.statusChip}>
            <span
              className={styles.statusDot}
              style={{
                background: isOnline ? "var(--status-online)" : "var(--status-offline)",
                boxShadow: isOnline
                  ? "0 0 6px var(--status-online)"
                  : "0 0 6px var(--status-offline)",
              }}
            />
            {isOnline ? "System Online" : "Offline"}
          </span>

          {/* Qdrant */}
          {health && (
            <span className={styles.statusChip}>
              <span
                className={styles.statusDot}
                style={{
                  background: health.qdrant_connected ? "var(--status-online)" : "var(--status-warning)",
                  boxShadow: `0 0 6px ${health.qdrant_connected ? "var(--status-online)" : "var(--status-warning)"}`,
                }}
              />
              Qdrant
            </span>
          )}

          {/* Twelve Labs */}
          {health && (
            <span className={styles.statusChip}>
              <span
                className={styles.statusDot}
                style={{
                  background: health.twelve_labs_enabled ? "var(--status-online)" : "var(--text-muted)",
                  boxShadow: health.twelve_labs_enabled ? "0 0 6px var(--status-online)" : "none",
                }}
              />
              VLM
            </span>
          )}

          {/* Clock */}
          <span className={styles.clock}>{clock}</span>
        </div>
      </header>

      {/* Error banner */}
      {error && !health && (
        <div
          style={{
            padding: "8px 20px",
            background: "var(--severity-high-bg)",
            color: "var(--severity-high)",
            fontSize: "0.75rem",
            borderBottom: "1px solid rgba(249,115,22,0.2)",
          }}
        >
          ⚠ Backend connection error: {error}
        </div>
      )}

      {/* ---- Main Area ---- */}
      <div className={styles.mainArea}>
        <div className={styles.gridArea}>
          <CameraGrid />
        </div>
        <div className={styles.alertArea}>
          <AlertQueue />
        </div>
      </div>

      {/* ---- Bottom Timeline ---- */}
      <div className={styles.timelineArea}>
        <ContinuousTimeline />
      </div>

      {/* ---- Floating Components ---- */}
      <IncidentModal />
      <OpsCopilot />
    </div>
  );
}
