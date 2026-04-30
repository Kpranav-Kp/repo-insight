// src/lib/sessionStore.js
// Saves chat sessions to localStorage so the user can resume them later.

const KEY = "repoinsight_sessions";

export function loadSessions() {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveSessions(sessions) {
  localStorage.setItem(KEY, JSON.stringify(sessions));
  // notify other components in the same tab
  window.dispatchEvent(new Event("sessions-updated"));
}

export function upsertSession(session) {
  const all = loadSessions();
  const idx = all.findIndex((s) => s.localId === session.localId);
  if (idx >= 0) all[idx] = session;
  else all.unshift(session);
  saveSessions(all);
}

export function deleteSession(localId) {
  saveSessions(loadSessions().filter((s) => s.localId !== localId));
}

export function newLocalId() {
  return `s_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}
