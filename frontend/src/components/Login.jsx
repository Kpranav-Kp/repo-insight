import { Mail, Lock, ArrowRight, Eye, EyeOff } from "lucide-react";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

export default function Login() {
  const [showPassword, setShowPassword] = useState(false);

  return (
    <div className="relative min-h-screen w-full overflow-hidden bg-background bg-grid flex items-center justify-center px-4 py-12">
      {/* Animated gradient blobs */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-24 -left-24 h-96 w-96 rounded-full bg-primary/30 blur-3xl animate-blob" />
        <div className="absolute top-1/3 -right-24 h-96 w-96 rounded-full bg-primary/20 blur-3xl animate-blob animation-delay-2000" />
        <div className="absolute -bottom-24 left-1/3 h-96 w-96 rounded-full bg-violet-600/15 blur-3xl animate-blob animation-delay-4000" />
      </div>

      {/* Card */}
      <Card className="relative z-10 w-full max-w-md border-border/60 bg-card/70 backdrop-blur-xl shadow-2xl shadow-primary/10 p-8 rounded-2xl">
        {/* Glow border accent */}
        <div className="pointer-events-none absolute -inset-px rounded-2xl bg-gradient-to-br from-primary/40 via-transparent to-fuchsia-500/30 opacity-60 [mask:linear-gradient(#000,#000)_content-box,linear-gradient(#000,#000)] [mask-composite:exclude] p-px" />

        {/* Brand */}
        <div className="flex items-center gap-2 mb-6">
          <div className="h-9 w-9 rounded-lg bg-gradient-to-br from-primary to-violet-600 flex items-center justify-center font-bold text-primary-foreground shadow-lg shadow-primary/30">
            R
          </div>
          <span className="font-display text-lg font-semibold tracking-tight">
            RepoInsight
          </span>
        </div>

        <Badge
          variant="outline"
          className="mb-4 font-mono text-[10px] tracking-widest uppercase border-primary/40 text-primary"
        >
          v1.0 · welcome back
        </Badge>

        <h1 className="font-display text-3xl font-bold leading-tight">
          Log in to{" "}
          <span
            className="bg-gradient-to-r from-indigo-600 via-violet-600 to-purple-600 
dark:from-indigo-400 dark:via-violet-400 dark:to-purple-400 bg-clip-text text-transparent animate-gradient"
          >
            keep building.
          </span>
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Continue your open-source journey — your skill graph is waiting.
        </p>

        {/* Divider */}
        <div className="my-6 flex items-center gap-3">
          <div className="h-px flex-1 bg-border" />
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            login with email
          </span>
          <div className="h-px flex-1 bg-border" />
        </div>

        {/* Form */}
        <form className="space-y-4" onSubmit={(e) => e.preventDefault()}>
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-muted-foreground">
              Email
            </label>
            <div className="relative">
              <Mail className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                type="email"
                placeholder="you@repo.dev"
                className="pl-9 bg-secondary/40 border-border/70 focus-visible:ring-primary/60"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-muted-foreground">
                Password
              </label>
              <a href="#" className="text-xs text-primary hover:underline">
                Forgot?
              </a>
            </div>
            <div className="relative">
              <Lock className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                type={showPassword ? "text" : "password"}
                placeholder="••••••••"
                className="pl-9 pr-9 bg-secondary/40 border-border/70 focus-visible:ring-primary/60"
              />
              <button
                type="button"
                onClick={() => setShowPassword((s) => !s)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                {showPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            </div>
          </div>

          <Button
            type="submit"
            className="group w-full bg-gradient-to-r from-indigo-600 via-violet-600 to-purple-600 text-primary-foreground hover:opacity-95 shadow-lg shadow-primary/30 transition"
          >
            Log in
            <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-0.5" />
          </Button>
        </form>

        <p className="mt-6 text-center text-sm text-muted-foreground">
          New here?{" "}
          <a href="#" className="font-medium text-primary hover:underline">
            Create an account
          </a>
        </p>

        {/* Terminal-like footer chip (matches dark screenshot) */}
        <div className="mt-6 rounded-md border border-border/70 bg-secondary/40 px-3 py-2 font-mono text-[11px] text-muted-foreground">
          <span className="text-primary">$</span> repoinsight auth --login
          <span className="ml-1 inline-block h-3 w-1.5 translate-y-0.5 bg-primary animate-pulse" />
        </div>
      </Card>
    </div>
  );
}
