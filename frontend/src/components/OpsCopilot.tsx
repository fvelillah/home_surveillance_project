"use client";

import React, { useCallback, useRef, useState, useEffect } from "react";
import { useApp } from "@/context/AppContext";
import {
  central,
  formatTimestamp,
  formatDate,
  type SemanticSearchResultItem,
  type CopilotEvidence,
} from "@/lib/api";
import styles from "./OpsCopilot.module.css";

type TabId = "search" | "copilot";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  evidence?: CopilotEvidence[];
}

export default function OpsCopilot() {
  const { copilotOpen, setCopilotOpen, setSelectedIncident } = useApp();

  // Tab state
  const [activeTab, setActiveTab] = useState<TabId>("search");

  // Search state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SemanticSearchResultItem[]>([]);
  const [searching, setSearching] = useState(false);

  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  /* ---- Search ---- */
  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    try {
      const resp = await central.semanticSearch(searchQuery.trim(), { limit: 15 });
      setSearchResults(resp.results);
    } catch (err) {
      console.error("Search failed:", err);
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  }, [searchQuery]);

  /* ---- Chat ---- */
  const handleSend = useCallback(async () => {
    if (!chatInput.trim()) return;
    const msg = chatInput.trim();
    setChatInput("");
    setChatMessages((prev) => [...prev, { role: "user", content: msg }]);
    setChatLoading(true);
    try {
      const resp = await central.copilotChat(msg, sessionId);
      setSessionId(resp.session_id);
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: resp.response, evidence: resp.evidence },
      ]);
    } catch {
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: "⚠️ Failed to get a response. Please try again." },
      ]);
    } finally {
      setChatLoading(false);
    }
  }, [chatInput, sessionId]);

  /* ---- Digest ---- */
  const handleDigest = useCallback(async () => {
    setChatLoading(true);
    try {
      const resp = await central.dailyDigest();
      setActiveTab("copilot");
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: `📋 **Daily Digest**\n\n${resp.executive_summary}` },
      ]);
    } catch {
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: "⚠️ Failed to generate daily digest." },
      ]);
    } finally {
      setChatLoading(false);
    }
  }, []);

  /* ---- FAB or Drawer ---- */
  if (!copilotOpen) {
    return (
      <button
        className={styles.fab}
        onClick={() => setCopilotOpen(true)}
        aria-label="Open AI Security Copilot"
        title="AI Security Copilot"
      >
        🤖
      </button>
    );
  }

  return (
    <>
      <div className={styles.drawerOverlay} onClick={() => setCopilotOpen(false)} />
      <div className={styles.drawer}>
        {/* Header */}
        <div className={styles.drawerHeader}>
          <span className={styles.drawerTitle}>🤖 AI Security Copilot</span>
          <button className={styles.closeBtn} onClick={() => setCopilotOpen(false)}>
            ✕
          </button>
        </div>

        {/* Tabs */}
        <div className={styles.tabs}>
          <button
            className={`${styles.tab} ${activeTab === "search" ? styles.tabActive : ""}`}
            onClick={() => setActiveTab("search")}
          >
            🔍 Search
          </button>
          <button
            className={`${styles.tab} ${activeTab === "copilot" ? styles.tabActive : ""}`}
            onClick={() => setActiveTab("copilot")}
          >
            💬 Copilot
          </button>
        </div>

        {/* Content */}
        <div className={styles.content}>
          {activeTab === "search" ? (
            <div className={styles.searchPanel}>
              <div className={styles.searchInputRow}>
                <input
                  className={styles.searchInput}
                  type="text"
                  placeholder='Search footage... e.g. "person near garage at night"'
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                />
                <button
                  className={styles.searchBtn}
                  onClick={handleSearch}
                  disabled={searching || !searchQuery.trim()}
                >
                  {searching ? "..." : "Search"}
                </button>
              </div>

              {searching && <div className={styles.spinner} />}

              {!searching && searchResults.length === 0 && (
                <div className={styles.emptySearch}>
                  <span className={styles.emptyIcon}>🔍</span>
                  <span>Search your surveillance footage</span>
                  <span style={{ fontSize: "0.68rem" }}>
                    Use natural language to find events
                  </span>
                </div>
              )}

              <div className={styles.searchResults}>
                {searchResults.map((r) => (
                  <div
                    key={r.incident_id}
                    className={styles.resultCard}
                    onClick={() => {
                      // Try to open incident modal
                      central
                        .incident(r.incident_id)
                        .then(setSelectedIncident)
                        .catch(() => {});
                    }}
                  >
                    <div className={styles.resultHeader}>
                      <span className={styles.resultCamera}>{r.camera_name}</span>
                      <span className={styles.resultScore}>
                        {(r.score * 100).toFixed(0)}% match
                      </span>
                    </div>
                    <p className={styles.resultSummary}>{r.summary}</p>
                    <div className={styles.resultMeta}>
                      <span className={`badge badge-${r.severity_badge.toLowerCase()}`}>
                        {r.severity_badge}
                      </span>
                      <span>
                        {formatDate(r.timestamp)} {formatTimestamp(r.timestamp)}
                      </span>
                      <span>CH{r.channel}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Digest button */}
              <button className={styles.digestBtn} onClick={handleDigest}>
                📊 Generate Daily Digest
              </button>
            </div>
          ) : (
            <div className={styles.chatPanel}>
              <div className={styles.chatMessages}>
                {chatMessages.length === 0 && (
                  <div className={styles.emptySearch}>
                    <span className={styles.emptyIcon}>💬</span>
                    <span>Ask your AI security assistant</span>
                    <span style={{ fontSize: "0.68rem" }}>
                      &quot;Who was near the front door today?&quot;
                    </span>
                  </div>
                )}
                {chatMessages.map((msg, i) => (
                  <div key={i}>
                    <div
                      className={`${styles.chatBubble} ${
                        msg.role === "user" ? styles.userBubble : styles.assistantBubble
                      }`}
                    >
                      {msg.content}
                    </div>
                    {/* Evidence citations */}
                    {msg.evidence &&
                      msg.evidence.map((ev, j) => (
                        <div key={j} className={styles.evidenceCard}>
                          <span className={styles.evidenceCamera}>
                            📹 {ev.camera_name} — {ev.severity_badge}
                          </span>
                          <span>{ev.summary}</span>
                          <span style={{ fontSize: "0.6rem", color: "var(--text-muted)" }}>
                            {formatTimestamp(ev.timestamp)}
                          </span>
                        </div>
                      ))}
                  </div>
                ))}
                {chatLoading && <div className={styles.spinner} />}
                <div ref={chatEndRef} />
              </div>

              <div className={styles.chatInputRow}>
                <input
                  className={styles.chatInput}
                  type="text"
                  placeholder="Ask about your surveillance..."
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSend()}
                />
                <button
                  className={styles.sendBtn}
                  onClick={handleSend}
                  disabled={chatLoading || !chatInput.trim()}
                >
                  ↑
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
