import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { useTheme } from "@/components/ThemeToggle";

import SpacePanel from "./SpacePanel";

export default function Signup() {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  return (
    <div
      className={`min-h-screen flex flex-col md:flex-row ${isDark ? "bg-[#000000]" : "bg-[#FFFFFF]"}`}
    >
      <div className="hidden md:block md:w-1/2 h-screen sticky top-0">
        <SpacePanel />
      </div>

      <div
        className={`w-full md:w-1/2 min-h-screen flex items-center justify-center p-8 border-l ${
          isDark ? "bg-[#030107] border-white/5" : "bg-[#FAFAFA] border-black/5"
        }`}
      >
        <div className="w-full max-w-sm space-y-8">
          <div>
            <h2
              className={`text-2xl font-black tracking-tight font-sans ${isDark ? "text-white" : "text-[#000000]"}`}
            >
              Create Account
            </h2>
            <p
              className={`text-xs mt-1 font-mono ${isDark ? "text-[#1098F7]/70" : "text-[#2541B2]/70"}`}
            >
              Get started with contributing to real open source projects.
            </p>
          </div>

          <form
            className="space-y-4"
            onSubmit={async (e) => {
              e.preventDefault();
              setError(null);

              try {
                const res = await fetch(`/api/auth/signup/`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  credentials: "include",
                  body: JSON.stringify({ username, email, password }),
                });

                const data = await res.json().catch(() => null);

                if (!res.ok) {
                  const msg =
                    (data && (data.error || data.detail)) ||
                    `Signup failed (${res.status})`;
                  setError(msg);
                  return;
                }

                // After signup, go to login (no auto-login cookies currently)
                navigate("/login");
              } catch (err) {
                setError(err instanceof Error ? err.message : "Signup failed");
              }
            }}
          >
            <div className="space-y-1">
              <label
                htmlFor="signup-username"
                className={`text-[10px] font-mono tracking-widest uppercase ${isDark ? "text-[#1098F7]" : "text-[#2541B2]"}`}
              >
                Username
              </label>
              <input
                id="signup-username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="your_username"
                className={`w-full border rounded-lg px-4 py-3 text-sm font-mono focus:outline-none transition-all ${
                  isDark
                    ? "bg-[#0A0712] border-white/10 text-white focus:border-[#1098F7]"
                    : "bg-white border-black/10 text-black focus:border-[#2541B2]"
                }`}
                required
              />
            </div>

            <div className="space-y-1">
              <label
                htmlFor="signup-email"
                className={`text-[10px] font-mono tracking-widest uppercase ${isDark ? "text-[#1098F7]" : "text-[#2541B2]"}`}
              >
                Email
              </label>
              <input
                id="signup-email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="developer@example.com"
                className={`w-full border rounded-lg px-4 py-3 text-sm font-mono focus:outline-none transition-all ${
                  isDark
                    ? "bg-[#0A0712] border-white/10 text-white focus:border-[#1098F7]"
                    : "bg-white border-black/10 text-black focus:border-[#2541B2]"
                }`}
                required
              />
            </div>

            <div className="space-y-1">
              <label
                htmlFor="signup-password"
                className={`text-[10px] font-mono tracking-widest uppercase ${isDark ? "text-[#1098F7]" : "text-[#2541B2]"}`}
              >
                Password
              </label>
              <input
                id="signup-password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Create a strong password"
                className={`w-full border rounded-lg px-4 py-3 text-sm font-mono focus:outline-none transition-all ${
                  isDark
                    ? "bg-[#0A0712] border-white/10 text-white focus:border-[#1098F7]"
                    : "bg-white border-black/10 text-black focus:border-[#2541B2]"
                }`}
                required
              />
            </div>

            {error && (
              <p
                className={`text-xs text-red-500 mb-2 ${isDark ? "text-red-400" : "text-red-600"}`}
              >
                {error}
              </p>
            )}
            <button
              type="submit"
              className={`w-full font-mono font-bold text-xs uppercase tracking-widest py-3.5 rounded-lg transition-all cursor-pointer ${
                isDark
                  ? "bg-white text-black hover:bg-white/90"
                  : "bg-[#000000] text-white hover:bg-[#2541B2]"
              }`}
            >
              Create Account
            </button>
          </form>

          <p
            className={`text-center text-xs font-mono ${isDark ? "text-[#1098F7]/50" : "text-[#2541B2]/50"}`}
          >
            Already have an account?{" "}
            <Link
              to="/login"
              className={`${isDark ? "text-white" : "text-[#000000]"} hover:underline`}
            >
              Sign In
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
