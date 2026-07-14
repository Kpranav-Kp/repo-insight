import { supabase } from "./supabase";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api";

export async function loginWithGoogle() {
  const { error } = await supabase.auth.signInWithOAuth({
    provider: "google",
    options: {
      redirectTo: `${window.location.origin}/auth/callback`,
    },
  });
  if (error) throw error;
}

export async function signup(email, password) {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      emailRedirectTo: `${window.location.origin}/auth/callback`,
    },
  });
  if (error) throw error;
  return data;
}

export async function login(email, password) {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });
  if (error) throw error;
  return data;
}

export async function resetPassword(email) {
  const { error } = await supabase.auth.resetPasswordForEmail(email, {
    redirectTo: `${window.location.origin}/auth/callback`,
  });
  if (error) throw error;
}

export async function exchangeSupabaseToken(accessToken) {
  const res = await fetch(`${API_BASE}/auth/supabase/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ access_token: accessToken }),
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const msg =
      (data && (data.error || data.detail)) || `Auth failed (${res.status})`;
    throw new Error(msg);
  }
  return data;
}

export async function getSession() {
  const {
    data: { session },
  } = await supabase.auth.getSession();
  return session;
}

export async function signOut() {
  const { error } = await supabase.auth.signOut();
  if (error) throw error;
}

export async function checkSession() {
  try {
    const res = await fetch(`${API_BASE}/auth/session/`, {
      credentials: "include",
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function backendLogout() {
  try {
    await fetch(`${API_BASE}/auth/logout/`, {
      method: "POST",
      credentials: "include",
    });
  } catch {
    // ignore
  }
}
