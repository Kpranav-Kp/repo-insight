// frontend/src/lib/api.js
import { getStoredToken } from "./auth";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

export async function request(path, options = {}) {
  const token = getStoredToken();
  const headers = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options.headers || {}),
  };
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers,
    credentials: "include",
  });
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    throw new Error(
      `Server returned non-JSON response (${res.status}): ${text.slice(0, 120)}`,
    );
  }
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

  submitStructuredSkills: (sessionId, skills) =>
    request(`/chat/session/${sessionId}/skills/structured/`, {
      method: "PUT",
      body: JSON.stringify({ skills }),
    }),

  getRecommendations: (sessionId) =>
    request(`/chat/session/${sessionId}/recommendations/`, {
      method: "GET",
    }),

  sendRecommendationFeedback: (recId, feedback) =>
    request(`/recommendations/${recId}/feedback/`, {
      method: "PATCH",
      body: JSON.stringify({ feedback }),
    }),

  flushFeedback: () =>
    request("/feedback/flush/", {
      method: "POST",
    }),

  selectIssue: (sessionId, issue) =>
    request(`/chat/session/${sessionId}/select-issue/`, {
      method: "PUT",
      body: JSON.stringify({ issue }),
    }),

  submitNoSkills: (sessionId) =>
    request(`/chat/session/${sessionId}/no-skills/`, {
      method: "GET",
    }),

  submitExtraSkills: (sessionId, skills) =>
    request(`/chat/session/${sessionId}/skills/structured/`, {
      method: "PUT",
      body: JSON.stringify({ skills }),
    }),
};

export async function poll(
  fn,
  done,
  { intervalMs = 1500, timeoutMs = 120000, signal } = {},
) {
  const start = Date.now();
  while (true) {
    if (signal?.aborted) throw new Error("Aborted");
    const v = await fn();
    if (done(v)) return v;
    if (Date.now() - start > timeoutMs) throw new Error("Timed out");
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}
