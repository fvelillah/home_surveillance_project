"use client";

import React from "react";
import { useCameras } from "@/context/CameraContext";
import CameraPanel from "./CameraPanel";
import type { CameraStreamInfo } from "@/lib/api";
import styles from "./CameraGrid.module.css";

interface CameraGridProps {
  onCameraClick?: (camera: CameraStreamInfo) => void;
}

export default function CameraGrid({ onCameraClick }: CameraGridProps) {
  const { cameras } = useCameras();

  return (
    <div className={styles.grid}>
      {cameras.map((cam) => (
        <CameraPanel
          key={cam.channel}
          camera={cam}
          anomalyScore={cam.latest_score ?? 0}
          onClick={onCameraClick}
        />
      ))}
    </div>
  );
}
