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
      <body>
        <AppProvider>
          <CameraProvider>{children}</CameraProvider>
        </AppProvider>
      </body>
    </html>
  );
}
