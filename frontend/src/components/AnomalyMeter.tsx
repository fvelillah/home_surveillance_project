"use client";

import React from "react";
import { severityColor } from "@/lib/api";
import styles from "./AnomalyMeter.module.css";

interface AnomalyMeterProps {
  score: number; // 0–100
  size?: number; // px
  strokeWidth?: number;
  label?: string;
}

export default function AnomalyMeter({
  score,
  size = 64,
  strokeWidth = 5,
  label = "Score",
}: AnomalyMeterProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const clampedScore = Math.max(0, Math.min(100, score));
  const offset = circumference - (clampedScore / 100) * circumference;

  const badge =
    clampedScore >= 70
      ? "CRITICAL"
      : clampedScore >= 45
        ? "HIGH"
        : clampedScore >= 20
          ? "MODERATE"
          : "LOW";

  const color = severityColor(badge);

  return (
    <div className={styles.wrapper} style={{ width: size, height: size }}>
      <svg className={styles.svg} width={size} height={size}>
        <circle
          className={styles.trackCircle}
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
        />
        <circle
          className={styles.valueCircle}
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          stroke={color}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className={styles.label}>
        <span className={styles.value} style={{ fontSize: size * 0.28, color }}>
          {Math.round(clampedScore)}
        </span>
        <span className={styles.caption}>{label}</span>
      </div>
    </div>
  );
}
