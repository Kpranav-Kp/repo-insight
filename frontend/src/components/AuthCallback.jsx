import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { useTheme } from "@/components/ThemeToggle";
import { exchangeSupabaseToken } from "@/lib/auth";
import { migrateSessionsFromOldKey } from "@/lib/sessionStore";
import { supabase } from "@/lib/supabase";

import SpacePanel from "./SpacePanel";

export default function AuthCallback() {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const processed = useRef(false);

  useEffect(() => {
    let cancelled = false;

    async function processSession(session) {
      if (!session?.access_token || processed.current) return false;
      processed.current = true;
      try {
        const data = await exchangeSupabaseToken(session.access_token);
        if (!cancelled) {
          localStorage.setItem("username", data.username);
          if (data.email) localStorage.setItem("email", data.email);
          migrateSessionsFromOldKey();
          navigate("/chat", { replace: true });
        }
        return true;
      } catch (err) {
        if (!cancelled) {
          const msg =
            err instanceof Error ? err.message : "Authentication failed";
          if (msg.toLowerCase().includes("verify your email")) {
            navigate("/verify-email", { replace: true });
            return true;
          }
          setError(msg);
        }
        return false;
      }
    }

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      if ((event === "SIGNED_IN" || event === "INITIAL_SESSION") && session) {
        processSession(session);
      }
      if (event === "SIGNED_OUT" && !cancelled) {
        setError("No session found. Please try logging in again.");
      }
    });

    async function tryGetSession(attempts = 0) {
      if (cancelled) return;
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (session) {
        processSession(session);
      } else if (attempts < 10) {
        setTimeout(() => tryGetSession(attempts + 1), 500);
      }
    }
    tryGetSession();

    return () => {
      cancelled = true;
      subscription.unsubscribe();
    };
  }, [navigate]);

  return (
    <div
      className={`min-h-screen flex ${isDark ? "bg-[#000000]" : "bg-[#FFFFFF]"}`}
    >
      <div className="hidden md:block md:w-1/2 h-screen sticky top-0">
        <SpacePanel />
      </div>
      <div
        className={`w-full md:w-1/2 min-h-screen flex items-center justify-center p-8 border-l ${
          isDark ? "bg-[#030107] border-white/5" : "bg-[#FAFAFA] border-black/5"
        }`}
      >
        <div className="text-center">
          {error ? (
            <div>
              <p className="text-red-500 text-sm font-mono mb-4">{error}</p>
              <button
                onClick={() => navigate("/login")}
                className={`font-mono text-xs uppercase tracking-widest py-3 px-6 rounded-lg cursor-pointer ${
                  isDark
                    ? "bg-white text-black hover:bg-white/90"
                    : "bg-[#000000] text-white"
                }`}
              >
                Back to Login
              </button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4">
              <div
                className={`w-8 h-8 border-2 rounded-full animate-spin ${
                  isDark
                    ? "border-white/20 border-t-white"
                    : "border-black/20 border-t-black"
                }`}
              />
              <p
                className={`text-sm font-mono ${isDark ? "text-white/60" : "text-black/60"}`}
              >
                Completing sign in...
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
