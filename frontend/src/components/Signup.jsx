// frontend/src/components/SignupPage.jsx
// Signup page component - split screen layout with SpacePanel and clean white form panel
import { Mail, Lock, User, ArrowRight, Eye, EyeOff } from "lucide-react";
import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

import SpacePanel from "./SpacePanel";

export default function SignupPage() {
  const [showPassword, setShowPassword] = useState(false);
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  // Temporary light mode override
  useEffect(() => {
    const root = document.documentElement;
    const hadDark = root.classList.contains("dark");
    if (hadDark) {
      root.classList.remove("dark");
    }
    return () => {
      if (hadDark) {
        root.classList.add("dark");
      }
    };
  }, []);

  const handleSignup = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const res = await fetch("/api/auth/signup/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, email, password }),
      });

      if (!res.ok) {
        setError("Signup failed. Try different credentials.");
        setLoading(false);
        return;
      }

      // Auto-login after signup
      try {
        const loginRes = await fetch("/api/auth/login/", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });

        if (loginRes.ok) {
          const loginData = await loginRes.json();
          localStorage.setItem("username", loginData.username);
          localStorage.setItem("email", email);
          setLoading(false);
          navigate("/chat");
        } else {
          setLoading(false);
          navigate("/login");
        }
      } catch (_err) {
        setLoading(false);
        navigate("/login");
      }
    } catch (_err) {
      setError("Network error — is backend running?");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full flex flex-col md:flex-row bg-white text-slate-900 font-sans">
      {/* Left side: SpacePanel */}
      <div className="hidden md:block md:w-1/2 relative overflow-hidden">
        <SpacePanel />
      </div>

      {/* Right side: Form Panel */}
      <div className="w-full md:w-1/2 flex items-center justify-center p-8 md:p-16 bg-white text-slate-900">
        <div className="w-full max-w-md space-y-6">
          {/* Brand */}
          <div className="flex items-center gap-2 mb-2">
            <div className="h-9 w-9 rounded-lg bg-linear-to-br from-indigo-600 to-violet-600 flex items-center justify-center font-bold text-white shadow-md">
              R
            </div>
            <span className="font-display text-lg font-bold tracking-tight text-slate-900">
              RepoInsight
            </span>
          </div>

          <div className="space-y-1">
            <h1 className="font-display text-3xl font-bold leading-tight text-slate-900">
              Start{" "}
              <span className="bg-linear-to-r from-indigo-600 via-violet-600 to-purple-600 bg-clip-text text-transparent animate-gradient">
                building.
              </span>
            </h1>
            <p className="text-sm text-slate-500">
              Create your account and begin your open-source journey.
            </p>
          </div>

          {/* Divider */}
          <div className="my-4 flex items-center gap-3">
            <div className="h-px flex-1 bg-slate-200" />
            <span className="font-mono text-[10px] uppercase tracking-widest text-slate-400">
              signup with email
            </span>
            <div className="h-px flex-1 bg-slate-200" />
          </div>

          {/* FORM */}
          <form className="space-y-4" onSubmit={handleSignup}>
            {/* Username */}
            <div className="space-y-1.5">
              <label
                htmlFor="signup-username"
                className="text-xs font-semibold text-slate-600"
              >
                Username
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                <Input
                  type="text"
                  placeholder="yourusername"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  className="pl-9 bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 focus-visible:border-violet-600 focus-visible:ring-violet-600/20"
                />
              </div>
            </div>

            {/* Email */}
            <div className="space-y-1.5">
              <label
                htmlFor="signup-email"
                className="text-xs font-semibold text-slate-600"
              >
                Email
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                <Input
                  type="email"
                  placeholder="you@repo.dev"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="pl-9 bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 focus-visible:border-violet-600 focus-visible:ring-violet-600/20"
                />
              </div>
            </div>

            {/* Password */}
            <div className="space-y-1.5">
              <label
                htmlFor="signup-password"
                className="text-xs font-semibold text-slate-600"
              >
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                <Input
                  type={showPassword ? "text" : "password"}
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="pl-9 pr-9 bg-slate-50 border-slate-200 text-slate-900 placeholder:text-slate-400 focus-visible:border-violet-600 focus-visible:ring-violet-600/20"
                />

                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>

            {error && (
              <p className="text-sm text-red-500 mb-2 font-medium">{error}</p>
            )}

            <Button
              type="submit"
              disabled={loading}
              className="group w-full bg-linear-to-r from-indigo-600 via-violet-600 to-purple-600 text-white hover:opacity-95 shadow-md shadow-violet-500/20 transition cursor-pointer"
            >
              {loading ? "Creating..." : "Sign up"}
              <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </Button>
          </form>

          <p className="text-center text-sm text-slate-600">
            Already have an account?{" "}
            <Link
              to="/login"
              className="text-violet-600 hover:text-violet-700 font-semibold hover:underline"
            >
              Log in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
