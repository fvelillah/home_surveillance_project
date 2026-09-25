import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'CYBER-DEFENSE SURVEILLANCE MATRIX | 9-Feed AI Command Console',
  description:
    'Tactical real-time 9-channel Dahua surveillance console, Direct HTTP video streaming, Qdrant vector anomaly detection, and Twelve Labs Pegasus AI Security Copilot.',
  keywords: [
    'Surveillance',
    'AI Security',
    'Twelve Labs',
    'Pegasus VLM',
    'Qdrant',
    'Dahua HTTP Streaming',
    'Anomaly Detection',
  ],
};

export const viewport: Viewport = {
  themeColor: '#06090e',
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <div id="portal-root" />
        {children}
      </body>
    </html>
  );
}
