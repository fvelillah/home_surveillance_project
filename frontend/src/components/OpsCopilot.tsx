'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useSurveillance } from '@/context/SurveillanceContext';
import {
  executeSemanticSearch,
  sendCopilotMessage,
  fetchDailyDigest,
  fetchDigestSummary,
} from '@/lib/api';
import {
  CopilotMessage,
  DailyDigest,
  SemanticSearchResultItem,
} from '@/types';
import {
  IconClose,
  IconSearch,
  IconSparkles,
  IconFileText,
  IconSend,
  IconZoom,
  IconShield,
} from '@/components/Icons';
import { soundFx } from '@/lib/audio';
import styles from './OpsCopilot.module.css';

export const OpsCopilot: React.FC = () => {
  const {
    isCopilotOpen,
    setIsCopilotOpen,
    copilotTab,
    setCopilotTab,
    setInspectIncident,
    incidents,
  } = useSurveillance();

  // Search State
  const [searchQuery, setSearchQuery] = useState('');
  const [searchChannel, setSearchChannel] = useState<number | undefined>(undefined);
  const [searchThreshold, setSearchThreshold] = useState<number>(0.35);
  const [searchResults, setSearchResults] = useState<SemanticSearchResultItem[]>([]);
  const [isSearching, setIsSearching] = useState(false);

  // Chat State
  const [sessionId] = useState<string>(() => `session-${Date.now()}`);
  const [chatInput, setChatInput] = useState('');
  const [messages, setMessages] = useState<CopilotMessage[]>([
    {
      role: 'assistant',
      content:
        'Greetings Operator. I am your AI Surveillance Copilot powered by Twelve Labs Pegasus & Qdrant. Ask me to search footage, analyze threat posture, or summarize camera activity.',
      timestamp: Date.now(),
    },
  ]);
  const [isChatting, setIsChatting] = useState(false);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Digest State
  const [digest, setDigest] = useState<DailyDigest | null>(null);
  const [isLoadingDigest, setIsLoadingDigest] = useState(false);

  // Load digest on tab switch
  useEffect(() => {
    if (copilotTab === 'digest' && !digest) {
      setIsLoadingDigest(true);
      fetchDailyDigest('full')
        .then((d) => setDigest(d))
        .finally(() => setIsLoadingDigest(false));
    }
  }, [copilotTab, digest]);

  // Scroll chat to bottom
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (!isCopilotOpen) return null;

  // Search Submit
  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!searchQuery.trim()) return;

    soundFx.playClick();
    setIsSearching(true);
    try {
      const res = await executeSemanticSearch({
        query: searchQuery,
        channel: searchChannel,
        threshold: searchThreshold,
      });
      setSearchResults(res.results || []);
    } finally {
      setIsSearching(false);
    }
  };

  // Chat Submit
  const handleSendMessage = async (textToSend?: string) => {
    const msg = (textToSend || chatInput).trim();
    if (!msg) return;

    soundFx.playClick();
    const userMsg: CopilotMessage = {
      role: 'user',
      content: msg,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setChatInput('');
    setIsChatting(true);

    try {
      const response = await sendCopilotMessage({
        message: msg,
        session_id: sessionId,
      });

      const assistantMsg: CopilotMessage = {
        role: 'assistant',
        content: response.reply,
        citations: response.citations,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } finally {
      setIsChatting(false);
    }
  };

  return (
    <div className={styles.drawerOverlay} onClick={() => setIsCopilotOpen(false)}>
      <div className={styles.drawer} onClick={(e) => e.stopPropagation()}>
        {/* Header & Navigation Tabs */}
        <div className={styles.drawerHeader}>
          <div className={styles.headerTop}>
            <div className={styles.drawerTitle}>
              <IconSparkles size={18} style={{ color: '#38bdf8' }} />
              <span>INTELLIGENCE & AI COPILOT</span>
            </div>
            <button
              className={styles.closeBtn}
              onClick={() => setIsCopilotOpen(false)}
              title="Close Drawer"
            >
              <IconClose size={16} />
            </button>
          </div>

          <div className={styles.tabBar}>
            <button
              className={`${styles.tabItem} ${copilotTab === 'copilot' ? styles.active : ''}`}
              onClick={() => {
                setCopilotTab('copilot');
                soundFx.playClick();
              }}
            >
              <IconSparkles size={14} />
              <span>COPILOT CHAT</span>
            </button>
            <button
              className={`${styles.tabItem} ${copilotTab === 'search' ? styles.active : ''}`}
              onClick={() => {
                setCopilotTab('search');
                soundFx.playClick();
              }}
            >
              <IconSearch size={14} />
              <span>SEMANTIC SEARCH</span>
            </button>
            <button
              className={`${styles.tabItem} ${copilotTab === 'digest' ? styles.active : ''}`}
              onClick={() => {
                setCopilotTab('digest');
                soundFx.playClick();
              }}
            >
              <IconFileText size={14} />
              <span>DAILY DIGEST</span>
            </button>
          </div>
        </div>

        {/* Tab 1: Conversational AI Copilot */}
        {copilotTab === 'copilot' && (
          <div className={styles.contentArea}>
            {/* Suggestion prompt chips */}
            <div className={styles.promptChips}>
              <button
                className={styles.chipBtn}
                onClick={() => handleSendMessage("Summarize today's anomalies across all 9 cameras")}
              >
                Summarize Today&apos;s Anomalies
              </button>
              <button
                className={styles.chipBtn}
                onClick={() => handleSendMessage('Which camera feed recorded the highest peak score?')}
              >
                Highest Peak Score
              </button>
              <button
                className={styles.chipBtn}
                onClick={() => handleSendMessage('Analyze threat posture near the front gate and perimeter')}
              >
                Check Front Gate
              </button>
            </div>

            {/* Chat Stream */}
            <div className={styles.chatStream}>
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`${styles.chatMsg} ${
                    msg.role === 'user' ? styles.user : styles.assistant
                  }`}
                >
                  <div
                    className={`${styles.bubble} ${
                      msg.role === 'user' ? styles.user : styles.assistant
                    }`}
                  >
                    {msg.content}
                  </div>
                  <span className={styles.msgTime}>
                    {msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString() : ''}
                  </span>
                </div>
              ))}
              {isChatting && (
                <div className={`${styles.chatMsg} ${styles.assistant}`}>
                  <div className={`${styles.bubble} ${styles.assistant}`}>
                    <span className="badge-blink">Twelve Labs Pegasus reasoning...</span>
                  </div>
                </div>
              )}
              <div ref={chatBottomRef} />
            </div>

            {/* Chat Input Bar */}
            <div className={styles.chatInputRow}>
              <input
                type="text"
                placeholder="Ask Pegasus Copilot about surveillance events..."
                className={styles.chatInput}
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleSendMessage();
                }}
              />
              <button
                className={styles.sendBtn}
                onClick={() => handleSendMessage()}
                disabled={isChatting}
              >
                <IconSend size={16} />
              </button>
            </div>
          </div>
        )}

        {/* Tab 2: Semantic Video Search */}
        {copilotTab === 'search' && (
          <div className={styles.contentArea}>
            <form onSubmit={handleSearch} style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div className={styles.searchBarWrapper}>
                <input
                  type="text"
                  placeholder="e.g. delivery driver, animal near pool, person walking at night"
                  className={styles.searchInput}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                <button type="submit" className={styles.searchSubmitBtn} disabled={isSearching}>
                  <IconSearch size={14} />
                  <span>{isSearching ? 'SEARCHING...' : 'SEARCH'}</span>
                </button>
              </div>

              <div className={styles.filterRow}>
                <label>FILTER SECTOR:</label>
                <select
                  className={styles.filterSelect}
                  value={searchChannel ?? ''}
                  onChange={(e) =>
                    setSearchChannel(e.target.value === '' ? undefined : Number(e.target.value))
                  }
                >
                  <option value="">ALL CHANNELS</option>
                  {Array.from({ length: 9 }, (_, i) => i + 1).map((ch) => (
                    <option key={ch} value={ch}>
                      CH-{String(ch).padStart(2, '0')}
                    </option>
                  ))}
                </select>

                <label style={{ marginLeft: 'auto' }}>
                  THRESHOLD: {searchThreshold.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.2"
                  max="0.8"
                  step="0.05"
                  value={searchThreshold}
                  onChange={(e) => setSearchThreshold(parseFloat(e.target.value))}
                  style={{ width: '80px', accentColor: '#38bdf8' }}
                />
              </div>
            </form>

            {/* Results */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '6px' }}>
              <span style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                {searchResults.length} MULTI-MODAL MATCHES FOUND
              </span>

              {searchResults.map((item, idx) => (
                <div
                  key={idx}
                  className={styles.searchResultCard}
                  onClick={() => {
                    const matchInc = incidents.find((inc) => inc.incident_id === item.incident_id);
                    if (matchInc) setInspectIncident(matchInc);
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--hud-emerald)' }}>
                      CH-{String(item.channel).padStart(2, '0')}
                    </span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: '#38bdf8' }}>
                      SIMILARITY: {(item.similarity_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-primary)' }}>
                    {item.summary || 'Video match retrieved via Twelve Labs Marengo embedding search.'}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 3: Daily Surveillance Digest */}
        {copilotTab === 'digest' && (
          <div className={styles.contentArea}>
            {isLoadingDigest ? (
              <div style={{ textAlign: 'center', padding: '40px', fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                GENERATING 24-HOUR EXECUTIVE REPORT...
              </div>
            ) : digest ? (
              <>
                <div className={styles.digestCard}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '0.95rem' }}>
                      24-HOUR SURVEILLANCE REPORT
                    </span>
                    <span
                      style={{
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.7rem',
                        fontWeight: 700,
                        background:
                          digest.threat_level === 'CRITICAL'
                            ? 'rgba(239, 68, 68, 0.25)'
                            : 'rgba(16, 185, 129, 0.2)',
                        color:
                          digest.threat_level === 'CRITICAL' ? 'var(--hud-crimson)' : 'var(--hud-emerald)',
                      }}
                    >
                      DEFCON: {digest.threat_level}
                    </span>
                  </div>

                  <div className={styles.statGrid}>
                    <div className={styles.statItem}>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>TOTAL</span>
                      <span className={styles.statVal}>{digest.total_incidents}</span>
                    </div>
                    <div className={styles.statItem}>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>CRITICAL</span>
                      <span className={styles.statVal} style={{ color: 'var(--hud-crimson)' }}>
                        {digest.critical_incidents}
                      </span>
                    </div>
                    <div className={styles.statItem}>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>HIGH SEV</span>
                      <span className={styles.statVal} style={{ color: 'var(--hud-amber)' }}>
                        {digest.high_incidents}
                      </span>
                    </div>
                  </div>
                </div>

                <div className={styles.digestCard}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: '#a5f3fc' }}>
                    EXECUTIVE SUMMARY
                  </span>
                  <p style={{ fontSize: '0.8rem', lineHeight: '1.45' }}>
                    {digest.executive_summary}
                  </p>
                </div>

                {digest.key_takeaways && digest.key_takeaways.length > 0 && (
                  <div className={styles.digestCard}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: '#a5f3fc' }}>
                      KEY TAKEAWAYS
                    </span>
                    <ul style={{ paddingLeft: '16px', fontSize: '0.78rem', lineHeight: '1.4' }}>
                      {digest.key_takeaways.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                No digest report generated yet.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
