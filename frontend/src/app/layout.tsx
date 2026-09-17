import type { Metadata } from "next";
import "./globals.css";
import { AppProvider } from "@/context/AppContext";
import { CameraProvider } from "@/context/CameraContext";

export const metadata: Metadata = {
  title: "SENTINEL — Home Surveillance Console",
  description:
    "AI-powered 9-channel home security surveillance dashboard with live camera feeds, anomaly detection, VLM scene analysis, and natural language search.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              // Intercept and suppress runtime errors injected by rogue browser extensions (e.g. Urban VPN)
              if (typeof window !== 'undefined') {
                window.addEventListener('error', function(e) {
                  if (e.filename && (e.filename.startsWith('chrome-extension://') || e.filename.startsWith('moz-extension://') || e.filename.includes('executors/200.js'))) {
                    e.stopImmediatePropagation();
                    e.preventDefault();
                    return true;
                  }
                  if (e.message && e.message.includes("reading 'M_ID'")) {
                    e.stopImmediatePropagation();
                    e.preventDefault();
                    return true;
                  }
                }, true);
                window.addEventListener('unhandledrejection', function(e) {
                  var stack = (e.reason && (e.reason.stack || e.reason.message)) || '';
                  if (stack.indexOf('chrome-extension://') !== -1 || stack.indexOf('moz-extension://') !== -1 || stack.indexOf('M_ID') !== -1) {
                    e.stopImmediatePropagation();
                    e.preventDefault();
                  }
                }, true);
              }
            `,
          }}
        />
      </head>
      <body>
        <AppProvider>
          <CameraProvider>{children}</CameraProvider>
        </AppProvider>
      </body>
    </html>
  );
}
