"use client";

import React, { useMemo, useState } from "react";
import { useApp } from "@/context/AppContext";
import { useCameras } from "@/context/CameraContext";
import { formatTimestamp, severityColor } from "@/lib/api";
import type { IncidentRecord } from "@/lib/api";
import styles from "./ContinuousTimeline.module.css";

const HOURS_SHOWN = 24;

export default function ContinuousTimeline() {
  const { allIncidents, setSelectedIncident } = useApp();
  const { cameras } = useCameras();
  const [hoveredIncident, setHoveredIncident] = useState<string | null>(null);

  // Time window: last 24 hours
  const now = Date.now() / 1000;
  const windowStart = now - HOURS_SHOWN * 3600;

  // Group incidents by channel
  const byChannel = useMemo(() => {
    const map: Record<number, IncidentRecord[]> = {};
    for (const cam of cameras) map[cam.channel] = [];
    for (const inc of allIncidents) {
      if (inc.start_time >= windowStart) {
        if (!map[inc.channel]) map[inc.channel] = [];
        map[inc.channel].push(inc);
      }
    }
    return map;
  }, [allIncidents, cameras, windowStart]);

  // Hour labels (every 4 hours)
  const hourLabels = useMemo(() => {
    const labels: string[] = [];
    for (let h = 0; h <= HOURS_SHOWN; h += 4) {
      const t = new Date((windowStart + h * 3600) * 1000);
      labels.push(
        t.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false }),
      );
    }
    return labels;
  }, [windowStart]);

  // Calculate position of NOW indicator
  const nowPct = ((now - windowStart) / (HOURS_SHOWN * 3600)) * 100;

  const calcPosition = (ts: number) => ((ts - windowStart) / (HOURS_SHOWN * 3600)) * 100;

  return (
    <div className={styles.wrapper}>
      <div className={styles.header}>
        <span className={styles.title}>📊 24-Hour Activity Timeline</span>
        <span className={styles.timeRange}>
          {new Date(windowStart * 1000).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false })}
          {" — "}
          {new Date(now * 1000).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false })}
        </span>
      </div>

      <div className={styles.timeline}>
        {/* Hour labels */}
        <div className={styles.hourLabels}>
          {hourLabels.map((label, i) => (
            <span key={i} className={styles.hourLabel}>{label}</span>
          ))}
        </div>

        {/* Camera rows */}
        <div className={styles.rows}>
          {cameras.map((cam) => {
            const incidents = byChannel[cam.channel] || [];
            return (
              <div key={cam.channel} className={styles.row}>
                <span className={styles.rowLabel} title={cam.name}>
                  {cam.name}
                </span>
                <div className={styles.rowTrack}>
                  {/* NOW line */}
                  {nowPct >= 0 && nowPct <= 100 && (
                    <div className={styles.nowLine} style={{ left: `${nowPct}%` }} />
                  )}

                  {/* Incident blocks */}
                  {incidents.map((inc) => {
                    const left = Math.max(0, calcPosition(inc.start_time));
                    const right = Math.min(100, calcPosition(inc.end_time));
                    const width = Math.max(right - left, 0.3);

                    return (
                      <div
                        key={inc.incident_id}
                        className={styles.incidentBlock}
                        style={{
                          left: `${left}%`,
                          width: `${width}%`,
                          background: severityColor(inc.severity_badge),
                          opacity: hoveredIncident === inc.incident_id ? 0.9 : 0.65,
                        }}
                        onClick={() => setSelectedIncident(inc)}
                        onMouseEnter={() => setHoveredIncident(inc.incident_id)}
                        onMouseLeave={() => setHoveredIncident(null)}
                        title={`${inc.camera_name} — ${inc.severity_badge} (${inc.severity}/100)`}
                      >
                        {hoveredIncident === inc.incident_id && (
                          <div className={styles.tooltip}>
                            <span>{inc.camera_name} — {inc.severity_badge}</span>
                            <span className={styles.tooltipTime}>
                              {formatTimestamp(inc.start_time)} · {inc.severity}/100
                            </span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
