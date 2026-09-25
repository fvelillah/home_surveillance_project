'use client';

import React from 'react';
import { SurveillanceProvider, useSurveillance } from '@/context/SurveillanceContext';
import { CommandHeader } from '@/components/CommandHeader';
import { CameraGrid } from '@/components/CameraGrid';
import { AlertQueue } from '@/components/AlertQueue';
import { ContinuousTimeline } from '@/components/ContinuousTimeline';
import { IncidentModal } from '@/components/IncidentModal';
import { OpsCopilot } from '@/components/OpsCopilot';
import { GovernancePanel } from '@/components/GovernancePanel';
import styles from './page.module.css';

const DashboardContent: React.FC = () => {
  const { inspectIncident, setInspectIncident } = useSurveillance();

  return (
    <div className={styles.dashboardRoot}>
      {/* Top Tactical Command Bar */}
      <CommandHeader />

      {/* Main Viewport: Camera Grid + Real-Time Threat Stream */}
      <main className={styles.mainLayout}>
        <section className={styles.cameraSection}>
          <CameraGrid />
        </section>

        <aside className={styles.alertSection}>
          <AlertQueue />
        </aside>
      </main>

      {/* Bottom Dock: 24-Hour Continuous Multi-Feed Timeline & Anomaly Heatmap */}
      <section className={styles.timelineSection}>
        <ContinuousTimeline />
      </section>

      {/* Deep Investigation Modal */}
      {inspectIncident && (
        <IncidentModal
          incident={inspectIncident}
          onClose={() => setInspectIncident(null)}
        />
      )}

      {/* Intelligence & AI Copilot Drawer */}
      <OpsCopilot />

      {/* System Governance Modal */}
      <GovernancePanel />
    </div>
  );
};

export default function Home() {
  return (
    <SurveillanceProvider>
      <DashboardContent />
    </SurveillanceProvider>
  );
}
