// Tiny API client for the Django backend.
// Configure base URL via VITE_API_BASE_URL (defaults to http://localhost:8000).
// Auth: JWT access token stored in localStorage under "access_token".

const BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api";

export function getToken() {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("access_token");
}

async function request(path, options = {}) {
  const token = getToken();
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${BASE_URL}${path}`, { ...options, headers });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;

  if (!res.ok) {
    const msg =
      (data && (data.error || data.detail)) || `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

export const api = {
  analyzeRepository: (url) =>
    request("/repositories/analyze/", {
      method: "POST",
      body: JSON.stringify({ url }),
    }),

  repositoryStatus: (repoId) => request(`/repositories/${repoId}/status/`),

  createSession: (repositoryId) =>
    request("/chat/session/", {
      method: "POST",
      body: JSON.stringify({ repository_id: repositoryId }),
    }),

  sendMessage: (sessionId, message) =>
    request("/chat/message/", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, message }),
    }),

  chatResult: (taskId) => request(`/chat/result/${taskId}/`),
};

// Poll helper
export async function poll(
  fn,
  done,
  { intervalMs = 1500, timeoutMs = 120000 } = {},
) {
  const start = Date.now();
  while (true) {
    const v = await fn();
    if (done(v)) return v;
    if (Date.now() - start > timeoutMs) throw new Error("Timed out");
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}
