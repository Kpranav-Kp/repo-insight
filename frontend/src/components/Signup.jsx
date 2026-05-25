//frontend/src/components/Signup.jsx
import { Mail, Lock, User, ArrowRight, Eye, EyeOff } from "lucide-react";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

export default function Signup({ onSignupSuccess, goToLogin }) {
  const [showPassword, setShowPassword] = useState(false);
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSignup = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:8000/api/auth/signup/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, email, password }),
      });

      if (!res.ok) {
        setError("Signup failed. Try different credentials.");
        setLoading(false);
        return;
      }
      setLoading(false);
      onSignupSuccess?.();
    } catch (_err) {
      setError("Network error — is backend running?");
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-background bg-grid flex items-center justify-center px-4 py-12">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-24 -left-24 h-96 w-96 rounded-full bg-primary/30 blur-3xl animate-blob" />
        <div className="absolute top-1/3 -right-24 h-96 w-96 rounded-full bg-primary/20 blur-3xl animate-blob animation-delay-2000" />
        <div className="absolute -bottom-24 left-1/3 h-96 w-96 rounded-full bg-violet-600/15 blur-3xl animate-blob animation-delay-4000" />
      </div>

      <Card className="relative z-10 w-full max-w-md border-border/60 bg-card/70 backdrop-blur-xl shadow-2xl shadow-primary/10 p-8 rounded-2xl">
        <div className="pointer-events-none absolute -inset-px rounded-2xl bg-linear-to-br from-primary/40 via-transparent to-fuchsia-500/30 opacity-60 p-px" />

        {/* Brand */}
        <div className="flex items-center gap-2 mb-6">
          <div className="h-9 w-9 rounded-lg bg-linear-to-br from-primary to-violet-600 flex items-center justify-center font-bold text-primary-foreground">
            R
          </div>
          <span className="font-display text-lg font-semibold">
            RepoInsight
          </span>
        </div>

        <Badge className="mb-4 text-[10px] uppercase border-primary/40 text-primary">
          v1.0 · create account
        </Badge>

        <h1 className="font-display text-3xl font-bold">
          Start{" "}
          <span className="bg-linear-to-r from-indigo-600 via-violet-600 to-purple-600 bg-clip-text text-transparent">
            building.
          </span>
        </h1>

        <p className="mt-2 text-sm text-muted-foreground">
          Create your account and begin your open-source journey.
        </p>

        {/* Divider */}
        <div className="my-6 flex items-center gap-3">
          <div className="h-px flex-1 bg-border" />
          <span className="text-[10px] uppercase text-muted-foreground">
            signup with email
          </span>
          <div className="h-px flex-1 bg-border" />
        </div>

        {/* FORM */}
        <form className="space-y-4" onSubmit={handleSignup}>
          {/* Username */}
          <div className="space-y-1.5">
            <label
              htmlFor="signup-username"
              className="text-xs text-muted-foreground"
            >
              Username
            </label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                type="text"
                placeholder="yourusername"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                className="pl-9 bg-secondary/40"
              />
            </div>
          </div>

          {/* Email */}
          <div className="space-y-1.5">
            <label
              htmlFor="signup-email"
              className="text-xs text-muted-foreground"
            >
              Email
            </label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                type="email"
                placeholder="you@repo.dev"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="pl-9 bg-secondary/40"
              />
            </div>
          </div>

          {/* Password */}
          <div className="space-y-1.5">
            <label
              htmlFor="signup-password"
              className="text-xs text-muted-foreground"
            >
              Password
            </label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="pl-9 pr-9 bg-secondary/40"
              />

              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2"
              >
                {showPassword ? <EyeOff /> : <Eye />}
              </button>
            </div>
          </div>

          {error && <p className="text-sm text-red-400">{error}</p>}

          <Button
            type="submit"
            disabled={loading}
            className="w-full bg-linear-to-r from-indigo-600 via-violet-600 to-purple-600"
          >
            {loading ? "Creating..." : "Sign up"}
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </form>

        <p className="mt-6 text-center text-sm text-muted-foreground">
          Already have an account?{" "}
          <button
            type="button"
            onClick={goToLogin}
            className="text-primary hover:underline cursor-pointer bg-transparent border-none p-0 text-sm"
          >
            Log in
          </button>
        </p>
      </Card>
    </div>
  );
}
