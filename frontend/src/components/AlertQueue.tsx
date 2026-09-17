"use client";

import React, { useMemo, useState } from "react";
import { useApp } from "@/context/AppContext";
import type { IncidentRecord } from "@/lib/api";
import ActiveIncidentCard from "./ActiveIncidentCard";
import styles from "./AlertQueue.module.css";

const FILTERS = ["ALL", "CRITICAL", "HIGH", "MODERATE", "LOW"] as const;
type Filter = (typeof FILTERS)[number];

export default function AlertQueue() {
  const { activeIncidents, allIncidents, setSelectedIncident } = useApp();
  const [filter, setFilter] = useState<Filter>("ALL");

  // Combine active + recent closed for a fuller view, deduplicated
  const combined = useMemo(() => {
    const map = new Map<string, IncidentRecord>();
    for (const inc of activeIncidents) map.set(inc.incident_id, inc);
    for (const inc of allIncidents) map.set(inc.incident_id, inc);
    const arr = Array.from(map.values());
    arr.sort((a, b) => {
      // Active first, then by severity desc, then by time desc
      const aActive = a.status === "OPEN" ? 0 : 1;
      const bActive = b.status === "OPEN" ? 0 : 1;
      if (aActive !== bActive) return aActive - bActive;
      if (b.severity !== a.severity) return b.severity - a.severity;
      return b.start_time - a.start_time;
    });
    return arr;
  }, [activeIncidents, allIncidents]);

  const filtered = useMemo(() => {
    if (filter === "ALL") return combined;
    return combined.filter((i) => i.severity_badge === filter);
  }, [combined, filter]);

  const handleClick = (inc: IncidentRecord) => {
    setSelectedIncident(inc);
  };

  return (
    <div className={styles.sidebar}>
      <div className={styles.header}>
        <span className={styles.title}>
          🔔 Alert Queue
          {activeIncidents.length > 0 && (
            <span className={styles.count}>{activeIncidents.length}</span>
          )}
        </span>
      </div>

      <div className={styles.filterBar}>
        {FILTERS.map((f) => (
          <button
            key={f}
            className={`${styles.filterBtn} ${filter === f ? styles.filterBtnActive : ""}`}
            onClick={() => setFilter(f)}
          >
            {f}
          </button>
        ))}
      </div>

      <div className={styles.list}>
        {filtered.length === 0 ? (
          <div className={styles.empty}>
            <span className={styles.emptyIcon}>🛡️</span>
            <span>No incidents to display</span>
            <span style={{ fontSize: "0.7rem" }}>All cameras operating normally</span>
          </div>
        ) : (
          filtered.map((inc) => (
            <ActiveIncidentCard key={inc.incident_id} incident={inc} onClick={handleClick} />
          ))
        )}
      </div>
    </div>
  );
}
