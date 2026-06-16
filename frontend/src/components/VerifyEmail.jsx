import { useState } from "react";
import { Link } from "react-router-dom";

import { useTheme } from "@/components/ThemeToggle";
import { request } from "@/lib/api";

import SpacePanel from "./SpacePanel";

export default function VerifyEmail() {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const [resending, setResending] = useState(false);
  const [message, setMessage] = useState(null);

  const handleResend = async () => {
    const email = localStorage.getItem("email");
    if (!email) {
      setMessage({
        type: "error",
        text: "No email found. Please sign up again.",
      });
      return;
    }
    setResending(true);
    setMessage(null);
    try {
      await request("/auth/resend-verification/", {
        method: "POST",
        body: JSON.stringify({ email }),
      });
      setMessage({ type: "success", text: "Verification email sent!" });
    } catch (err) {
      setMessage({
        type: "error",
        text: err instanceof Error ? err.message : "Failed to resend email",
      });
    } finally {
      setResending(false);
    }
  };

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
        <div className="w-full max-w-sm text-center space-y-6">
          <div>
            <h2
              className={`text-2xl font-black tracking-tight font-sans ${isDark ? "text-white" : "text-[#000000]"}`}
            >
              Check Your Email
            </h2>
            <p
              className={`text-xs mt-3 font-mono leading-relaxed ${isDark ? "text-[#1098F7]/70" : "text-[#2541B2]/70"}`}
            >
              We sent a verification link to your email.
              <br />
              Click the link to activate your account, then sign in.
            </p>
          </div>

          <div className="pt-4">
            <Link
              to="/login"
              className={`inline-block font-mono font-bold text-xs uppercase tracking-widest py-3.5 px-8 rounded-lg transition-all cursor-pointer ${
                isDark
                  ? "bg-white text-black hover:bg-white/90"
                  : "bg-[#000000] text-white hover:bg-[#2541B2]"
              }`}
            >
              Go to Login
            </Link>
          </div>

          <div className="space-y-3">
            <button
              onClick={handleResend}
              disabled={resending}
              className={`w-full font-mono font-bold text-xs uppercase tracking-widest py-3.5 px-8 rounded-lg transition-all cursor-pointer disabled:opacity-50 ${
                isDark
                  ? "bg-white/10 text-white hover:bg-white/20 border border-white/20"
                  : "bg-black/10 text-black hover:bg-black/20 border border-black/20"
              }`}
            >
              {resending ? "Sending..." : "Resend Verification Email"}
            </button>
            {message && (
              <p
                className={`text-center text-xs font-mono ${message.type === "success" ? "text-emerald-500" : "text-red-500"}`}
              >
                {message.text}
              </p>
            )}
          </div>

          <p
            className={`text-center text-xs font-mono ${isDark ? "text-[#1098F7]/50" : "text-[#2541B2]/50"}`}
          >
            Didn&apos;t receive it?{" "}
            <span className="text-inherit opacity-70">
              Check your spam folder
            </span>
          </p>
        </div>
      </div>
    </div>
  );
}
