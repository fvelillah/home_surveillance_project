'use client';

import React from 'react';

interface AnomalyMeterProps {
  score: number;
  threshold?: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
}

export const AnomalyMeter: React.FC<AnomalyMeterProps> = ({
  score,
  threshold = 0.08,
  size = 'md',
  showLabel = true,
}) => {
  const normalized = Math.min(Math.max(score, 0), 1.0);
  const isAnomaly = normalized >= threshold;
  const isElevated = normalized >= 0.05 && !isAnomaly;

  const color = isAnomaly
    ? 'var(--hud-crimson)'
    : isElevated
    ? 'var(--hud-amber)'
    : 'var(--hud-emerald)';

  const glowColor = isAnomaly
    ? 'var(--hud-crimson-glow)'
    : isElevated
    ? 'var(--hud-amber-glow)'
    : 'var(--hud-emerald-glow)';

  // Dimensions based on size
  const radius = size === 'sm' ? 16 : size === 'lg' ? 32 : 22;
  const strokeWidth = size === 'sm' ? 3 : size === 'lg' ? 5 : 4;
  const center = radius + strokeWidth;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - normalized * circumference;
  const svgSize = (radius + strokeWidth) * 2;

  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '8px',
        fontFamily: 'var(--font-mono)',
      }}
    >
      <div
        style={{
          position: 'relative',
          width: svgSize,
          height: svgSize,
          filter: `drop-shadow(0 0 6px ${glowColor})`,
        }}
      >
        <svg width={svgSize} height={svgSize} style={{ transform: 'rotate(-90deg)' }}>
          {/* Background circle */}
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="transparent"
            stroke="rgba(255, 255, 255, 0.1)"
            strokeWidth={strokeWidth}
          />
          {/* Active progress arc */}
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="transparent"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            style={{
              transition: 'stroke-dashoffset 0.4s ease, stroke 0.3s ease',
            }}
          />
        </svg>

        {/* Center score readout */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: size === 'sm' ? '0.62rem' : size === 'lg' ? '0.85rem' : '0.7rem',
            fontWeight: 700,
            color: '#fff',
          }}
        >
          {normalized.toFixed(2)}
        </div>
      </div>

      {showLabel && (
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <span
            style={{
              fontSize: '0.65rem',
              color: 'var(--text-muted)',
              letterSpacing: '0.04em',
            }}
          >
            ANOMALY
          </span>
          <span
            style={{
              fontSize: '0.78rem',
              fontWeight: 700,
              color: color,
            }}
          >
            {isAnomaly ? 'ESCALATED' : isElevated ? 'ELEVATED' : 'NOMINAL'}
          </span>
        </div>
      )}
    </div>
  );
};
