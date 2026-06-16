// frontend/src/lib/sessionStore.js
// Saves chat sessions to localStorage so the user can resume them later.

function getUserKey() {
  const email = localStorage.getItem("email");
  if (email) return email.toLowerCase();

  let userId = localStorage.getItem("repoinsight_user_id");
  if (!userId) {
    userId = `user_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    localStorage.setItem("repoinsight_user_id", userId);
  }
  return userId;
}

const getKey = () => `repoinsight_sessions_${getUserKey()}`;

export function loadSessions() {
  try {
    const raw = localStorage.getItem(getKey());
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveSessions(sessions) {
  localStorage.setItem(getKey(), JSON.stringify(sessions));
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

export function migrateSessionsFromOldKey() {
  const oldKey = `repoinsight_sessions_${localStorage.getItem("username")}`;
  const newKey = getKey();
  if (oldKey !== newKey) {
    try {
      const raw = localStorage.getItem(oldKey);
      if (raw) {
        localStorage.setItem(newKey, raw);
        localStorage.removeItem(oldKey);
      }
    } catch {
      // Ignore migration errors
    }
  }
}
