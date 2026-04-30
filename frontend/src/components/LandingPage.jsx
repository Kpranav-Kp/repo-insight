import { useRef, useEffect, useState } from "react";

import Login from "@/components/Login";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog";

const TerminalTypewriter = ({ text, delay = 65, pause = 2200 }) => {
  const [displayed, setDisplayed] = useState("");
  const [phase, setPhase] = useState("typing");

  useEffect(() => {
    let timer;
    if (phase === "typing") {
      if (displayed.length < text.length) {
        timer = setTimeout(
          () => setDisplayed(text.slice(0, displayed.length + 1)),
          delay,
        );
      } else {
        timer = setTimeout(() => setPhase("deleting"), pause);
      }
    } else if (phase === "deleting") {
      if (displayed.length > 0) {
        timer = setTimeout(
          () => setDisplayed(text.slice(0, displayed.length - 1)),
          delay / 2,
        );
      } else {
        timer = setTimeout(() => setPhase("typing"), 600);
      }
    }
    return () => clearTimeout(timer);
  }, [displayed, phase, text, delay, pause]);

  return (
    <span className="inline-flex items-center font-mono text-[13px]">
      {displayed}
      <span className="animate-pulse text-primary ml-0.5">▊</span>
    </span>
  );
};

const SemanticGraph = () => {
  const [gPhase, setGPhase] = useState("nodes");

  useEffect(() => {
    const cycle = () => {
      setGPhase("nodes");
      setTimeout(() => setGPhase("edge"), 3000);
      setTimeout(() => setGPhase("score"), 6000);
    };
    cycle();
    const id = setInterval(cycle, 9500);
    return () => clearInterval(id);
  }, []);

  const toRad = (deg) => (deg * Math.PI) / 180;

  const skills = [
    { name: "python", angle: -70, dist: 105, color: "#8b5cf6" },
    { name: "http", angle: -10, dist: 120, color: "#f59e0b", issue: true },
    { name: "auth", angle: 50, dist: 110, color: "#06b6d4" },
    { name: "json", angle: 110, dist: 100, color: "#ec4899" },
    { name: "test", angle: 160, dist: 115, color: "#10b981" },
  ];

  return (
    <div className="relative w-full h-full min-h-105 flex items-center justify-center">
      <div className="absolute inset-0 bg-linear-to-tr from-primary/10 via-violet-500/5 to-blue-500/10 rounded-full blur-3xl animate-pulse-slow" />

      <div
        className={`absolute inset-0 flex items-center justify-center transition-opacity duration-700 ${gPhase === "score" ? "opacity-0 scale-95" : "opacity-100 scale-100"}`}
      >
        <svg viewBox="0 0 400 400" className="w-full h-full max-w-110">
          <defs>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <linearGradient id="edgeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop
                offset="0%"
                stopColor="hsl(var(--primary))"
                stopOpacity="0.35"
              />
              <stop
                offset="100%"
                stopColor="hsl(var(--primary))"
                stopOpacity="0.02"
              />
            </linearGradient>
          </defs>

          <circle
            cx="200"
            cy="200"
            r="150"
            fill="none"
            stroke="hsl(var(--border))"
            strokeWidth="0.5"
            opacity="0.2"
          />
          <circle
            cx="200"
            cy="200"
            r="100"
            fill="none"
            stroke="hsl(var(--border))"
            strokeWidth="0.5"
            opacity="0.15"
          />

          {skills.map((s) => {
            const x = 200 + s.dist * Math.cos(toRad(s.angle));
            const y = 200 + s.dist * Math.sin(toRad(s.angle));
            const isIssue = s.issue;
            return (
              <g key={`e-${s.name}`}>
                <line
                  x1="200"
                  y1="200"
                  x2={x}
                  y2={y}
                  stroke={isIssue ? "#f59e0b" : "url(#edgeGrad)"}
                  strokeWidth={isIssue ? 3 : 1}
                  strokeDasharray={isIssue ? "none" : "4 6"}
                  opacity={(() => {
                    if (isIssue) return gPhase === "edge" ? 1 : 0.6;
                    return 0.35;
                  })()}
                  className="transition-all duration-500"
                />
                {isIssue && gPhase === "edge" && (
                  <circle r="3.5" fill="#f59e0b" filter="url(#glow)">
                    <animateMotion
                      dur="1.4s"
                      repeatCount="indefinite"
                      path={`M200,200 L${x},${y}`}
                    />
                  </circle>
                )}
              </g>
            );
          })}

          {skills.map((s, i) => {
            const x = 200 + s.dist * Math.cos(toRad(s.angle));
            const y = 200 + s.dist * Math.sin(toRad(s.angle));
            return (
              <g
                key={`n-${s.name}`}
                className="animate-float-1"
                style={{ animationDelay: `${i * 0.7}s` }}
              >
                <circle
                  cx={x}
                  cy={y}
                  r="18"
                  fill="hsl(var(--card))"
                  stroke={s.color}
                  strokeWidth="2.5"
                />
                <text
                  x={x}
                  y={y + 4}
                  textAnchor="middle"
                  fill="hsl(var(--foreground))"
                  style={{
                    fontSize: "10px",
                    fontFamily: "JetBrains Mono, monospace",
                    fontWeight: 700,
                  }}
                >
                  {s.name}
                </text>
              </g>
            );
          })}

          <g>
            <circle
              cx="200"
              cy="200"
              r="55"
              fill="none"
              stroke="hsl(var(--primary))"
              strokeWidth="0.5"
              opacity="0.12"
            >
              <animate
                attributeName="r"
                values="55;68;55"
                dur="4s"
                repeatCount="indefinite"
              />
              <animate
                attributeName="opacity"
                values="0.12;0.03;0.12"
                dur="4s"
                repeatCount="indefinite"
              />
            </circle>
            <circle
              cx="200"
              cy="200"
              r="28"
              fill="hsl(var(--card))"
              stroke="hsl(var(--primary))"
              strokeWidth="3"
              filter="url(#glow)"
            />
            <text
              x="200"
              y="205"
              textAnchor="middle"
              fill="hsl(var(--primary))"
              style={{
                fontSize: "12px",
                fontFamily: "JetBrains Mono, monospace",
                fontWeight: 700,
                letterSpacing: "0.12em",
              }}
            >
              REPO
            </text>
          </g>
        </svg>
      </div>

      <div
        className={`absolute inset-0 flex flex-col items-center justify-center transition-all duration-700 ${gPhase === "score" ? "opacity-100 scale-100" : "opacity-0 scale-95 pointer-events-none"}`}
      >
        <div className="relative w-44 h-44">
          <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
            <circle
              cx="60"
              cy="60"
              r="52"
              fill="none"
              stroke="hsl(var(--border))"
              strokeWidth="7"
              opacity="0.3"
            />
            <circle
              cx="60"
              cy="60"
              r="52"
              fill="none"
              stroke="hsl(var(--primary))"
              strokeWidth="7"
              strokeLinecap="round"
              strokeDasharray={326.73}
              strokeDashoffset={gPhase === "score" ? 19.6 : 326.73}
              className="transition-all duration-1200 ease-out"
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-4xl font-bold font-display text-foreground tracking-tight">
              94%
            </span>
            <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-[0.2em] mt-1">
              skill match
            </span>
          </div>
        </div>
        <p className="mt-6 text-sm text-muted-foreground font-light max-w-50 text-center leading-relaxed">
          Your profile aligns strongly with{" "}
          <span className="text-primary font-medium">http</span> and
          authentication patterns.
        </p>
      </div>

      <div className="absolute bottom-5 left-1/2 -translate-x-1/2 transition-all duration-500">
        {(() => {
          if (gPhase === "score") {
            return (
              <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 backdrop-blur-md">
                <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                <span className="text-xs font-medium text-primary font-mono tracking-wide">
                  94% alignment found
                </span>
              </div>
            );
          } else if (gPhase === "edge") {
            return (
              <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-amber-500/10 border border-amber-500/20 backdrop-blur-md">
                <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
                <span className="text-xs font-medium text-amber-500 font-mono tracking-wide">
                  issue #42 selected
                </span>
              </div>
            );
          } else {
            return (
              <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-primary/5 border border-primary/10 backdrop-blur-md">
                <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                <span className="text-xs font-medium text-primary font-mono tracking-wide">
                  semantic graph active
                </span>
              </div>
            );
          }
        })()}
      </div>
    </div>
  );
};

const Features = [
  {
    title: "Skill",
    titleAccent: "Graph",
    lead: "Interactive proficiency mapping.",
    body: "See exactly how your skills connect to repository needs in real time. No guesswork.",
    tag: "visual onboarding",
    icon: (
      <svg
        width="22"
        height="22"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="12" cy="5" r="3" />
        <circle cx="6" cy="12" r="3" />
        <circle cx="18" cy="12" r="3" />
        <circle cx="12" cy="19" r="3" />
        <path d="M12 8v2M9.5 10.5l-1.5 1.5M14.5 10.5l1.5 1.5M12 16v2" />
      </svg>
    ),
  },
  {
    title: "Socratic",
    titleAccent: "Guidance",
    lead: "Questions, not answers.",
    body: "The agent probes your understanding until you can explain the fix in your own words.",
    tag: "never gives solutions",
    icon: (
      <svg
        width="22"
        height="22"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      >
        <circle cx="12" cy="12" r="10" />
        <path d="M12 16v-4M12 8h.01" />
      </svg>
    ),
  },
  {
    title: "Guarded",
    titleAccent: "Code Assist",
    lead: "Boilerplate with guardrails.",
    body: "Stuck? You get three TODO-laden snippets per session. Then it's back to thinking.",
    tag: "ethical guardrails",
    icon: (
      <svg
        width="22"
        height="22"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      >
        <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
        <path d="M7 11V7a5 5 0 0 1 10 0v4" />
      </svg>
    ),
  },
  {
    title: "Learner",
    titleAccent: "Profile",
    lead: "Memory that persists.",
    body: "Mastered skills are remembered across sessions. You never revisit the same basics twice.",
    tag: "long‑term memory",
    icon: (
      <svg
        width="22"
        height="22"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      >
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
        <circle cx="12" cy="7" r="4" />
      </svg>
    ),
  },
  {
    title: "Semantic",
    titleAccent: "Graph",
    lead: "FAISS-powered matching.",
    body: "Issues, skills, and PRs are embedded into a living graph that adapts to your interests.",
    tag: "knowledge graph",
    icon: (
      <svg
        width="22"
        height="22"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      >
        <path d="M18 10a3 3 0 0 0 3-3 3 3 0 0 0-3-3 3 3 0 0 0-3 3 3 3 0 0 0 3 3ZM6 20a3 3 0 0 0 3-3 3 3 0 0 0-3-3 3 3 0 0 0-3 3 3 3 0 0 0 3 3Z" />
        <path d="M9 7h6v2H9zM14.83 14.83 18 17" />
      </svg>
    ),
  },
  {
    title: "PR Readiness",
    titleAccent: "Score",
    lead: "Review before you push.",
    body: "Get a novelty check against past PRs and a ready-to-use template before you commit.",
    tag: "review before code",
    icon: (
      <svg
        width="22"
        height="22"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      >
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
        <polyline points="22 4 12 14.01 9 11.01" />
      </svg>
    ),
  },
];

const LandingPage = ({ onLoginSuccess }) => {
  const scrollContainerRef = useRef(null);

  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;

    let target = el.scrollLeft;
    let vel = 0;
    let rafId;
    let lastTime = performance.now();

    const friction = 0.92;
    const pull = 0.08;

    function step(now) {
      const dt = Math.min((now - lastTime) / 16.67, 2);
      lastTime = now;

      const diff = target - el.scrollLeft;
      vel += diff * pull * dt;
      vel *= Math.pow(friction, dt);
      el.scrollLeft += vel * dt;

      if (Math.abs(vel) > 0.2 || Math.abs(diff) > 0.5) {
        rafId = requestAnimationFrame(step);
      }
    }

    const onWheel = (e) => {
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        e.preventDefault();
        const maxScroll = el.scrollWidth - el.clientWidth;
        target = Math.max(0, Math.min(target + e.deltaY * 1.4, maxScroll));
        cancelAnimationFrame(rafId);
        lastTime = performance.now();
        rafId = requestAnimationFrame(step);
      }
    };

    el.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      el.removeEventListener("wheel", onWheel);
      cancelAnimationFrame(rafId);
    };
  }, []);

  const terminalCommand = "repoinsight analyze https://github.com/psf/requests";

  return (
    <div className="relative min-h-screen bg-background text-foreground overflow-hidden selection:bg-primary/20 selection:text-primary-foreground">
      <div className="absolute inset-0 bg-linear-to-br from-violet-50/70 via-white/50 to-blue-50/70 dark:from-violet-950/25 dark:via-transparent dark:to-blue-950/20 transition-colors duration-500" />
      <div className="absolute inset-0 bg-[size:32px_32px] bg-[linear-gradient(to_right,var(--grid-color)_1px,transparent_1px),linear-gradient(to_bottom,var(--grid-color)_1px,transparent_1px)]" />
      <div className="absolute -top-28 left-1/2 -translate-x-1/2 w-280 h-152 rounded-b-full bg-linear-to-b from-violet-500/30 via-purple-500/18 to-transparent blur-3xl pointer-events-none dark:from-violet-500/25 dark:via-purple-500/15 dark:to-transparent transition-colors duration-500" />

      <header className="fixed top-0 left-0 right-0 z-50 border-b border-border/40 bg-background/70 backdrop-blur-xl transition-colors duration-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-linear-to-br from-primary to-violet-600 flex items-center justify-center text-primary-foreground font-bold text-sm font-mono shadow-lg shadow-primary/20">
              R
            </div>
            <span className="font-bold text-base tracking-tight font-display">
              RepoInsight
            </span>
          </div>

          <nav className="hidden md:flex items-center gap-8">
            <a
              href="/"
              className="text-sm font-semibold text-foreground hover:text-primary transition-colors"
            >
              Home
            </a>
            <a
              href="/chat"
              className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              Contribute
            </a>
            <a
              href="/profile"
              className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              Profile
            </a>
          </nav>

          <div className="flex items-center gap-3">
            {/*} <Button
              variant="ghost"
              size="sm"
              className="hidden sm:inline-flex text-sm font-medium"
            >
              Log in
            </Button>*/}

            <Dialog>
              <DialogTrigger className="hidden sm:inline-flex items-center justify-center rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-accent transition-colors">
                Log in
              </DialogTrigger>
              <DialogContent className="p-0 sm:max-w-md border-border/70 bg-transparent shadow-none">
                <Login onLoginSuccess={onLoginSuccess} />
              </DialogContent>
            </Dialog>

            <Button
              size="sm"
              className="hidden sm:inline-flex text-sm font-semibold shadow-md shadow-primary/15"
            >
              Sign up
            </Button>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-32 pb-16">
        <section className="grid lg:grid-cols-2 gap-14 lg:gap-20 items-center mb-32 md:mb-40">
          <div className="space-y-8">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-primary/8 border border-primary/15 backdrop-blur-sm">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
              </span>
              <span className="text-[11px] font-mono text-primary font-medium tracking-wide uppercase">
                v1.0 · Open Source Tutor
              </span>
            </div>

            <h1 className="space-y-2">
              <span className="block text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight leading-[1.05] font-display text-foreground">
                Learn by doing.
              </span>
              <span className="block text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight leading-[1.05] font-display italic bg-linear-to-r from-indigo-600 via-violet-600 to-purple-600 dark:from-indigo-400 dark:via-violet-400 dark:to-purple-400 text-transparent bg-clip-text">
                Not by copying.
              </span>
            </h1>

            <p className="text-lg md:text-xl text-muted-foreground max-w-lg leading-relaxed font-light font-sans">
              RepoInsight is an AI tutor that guides you through real
              open‑source issues. Socratic questions, guarded code assists, and
              a skill graph that grows with you.
            </p>

            <div className="flex flex-wrap gap-4 pt-2">
              <Button
                size="lg"
                className="shadow-xl shadow-primary/20 hover:shadow-primary/40 hover:scale-105 transition-all duration-200 bg-primary text-primary-foreground h-12 px-8 text-base font-semibold font-display"
                onClick={() => {
                  window.location.href = "/chat";
                }}
              >
                Start contributing
                <svg
                  className="ml-2 w-5 h-5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="hover:bg-secondary/80 hover:border-primary/30 transition-all duration-200 h-12 px-8 text-base font-medium font-display"
                onClick={() =>
                  document
                    .getElementById("features")
                    ?.scrollIntoView({ behavior: "smooth" })
                }
              >
                See how it works
              </Button>
            </div>

            <div className="pt-2">
              <div className="inline-flex items-center font-mono text-[13px] text-muted-foreground/90 border border-border/60 rounded-xl px-5 py-3.5 bg-secondary/50 backdrop-blur-sm shadow-sm">
                <span className="text-primary font-bold mr-2.5 select-none">
                  $
                </span>
                <TerminalTypewriter text={terminalCommand} delay={65} />
              </div>
            </div>
          </div>

          {/* Right Visual */}
          <div className="relative flex items-center justify-center order-first lg:order-last">
            <div className="relative w-full max-w-md aspect-square rounded-4xl border border-border/60 bg-card/60 backdrop-blur-2xl shadow-2xl shadow-primary/10 flex flex-col items-center justify-center p-6 overflow-hidden">
              <SemanticGraph />
            </div>
          </div>
        </section>

        {/* Features */}
        <section id="features" className="relative mb-32 md:mb-40">
          <div className="text-center max-w-3xl mx-auto mb-16 md:mb-20 space-y-5">
            <Badge
              variant="outline"
              className="text-[11px] font-mono font-medium tracking-wide uppercase"
            >
              Core Philosophy
            </Badge>
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold tracking-tight font-display">
              Designed for{" "}
              <span className="bg-linear-to-r from-indigo-600 via-violet-600 to-purple-600 dark:from-indigo-400 dark:via-violet-400 dark:to-purple-400 bg-clip-text text-transparent">
                real contribution
              </span>
            </h2>
            <p className="text-lg text-muted-foreground font-light font-sans max-w-xl mx-auto">
              Every feature exists to help you grow — no shortcuts, no
              spoon‑feeding.
            </p>
          </div>

          <div className="relative -mx-4 px-4">
            <div className="absolute right-0 top-0 bottom-0 w-28 bg-linear-to-l from-background via-background/90 to-transparent z-10 pointer-events-none dark:from-background dark:via-background/95" />

            <div
              ref={scrollContainerRef}
              className="flex overflow-x-auto gap-6 pb-10 snap-x snap-mandatory scrollbar-hide -mx-4 px-4"
            >
              {Features.map((feature, idx) => (
                <Card
                  key={idx}
                  className="min-w-75 md:min-w-[320px] max-w-85 snap-start group relative overflow-hidden transition-all duration-300 hover:-translate-y-2 hover:shadow-2xl hover:shadow-primary/5 border-border/60 bg-card/60 backdrop-blur-md hover:border-primary/25 shrink-0"
                >
                  <div className="absolute top-0 left-0 right-0 h-0.5 bg-linear-to-r from-transparent via-primary/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

                  <CardHeader className="space-y-5 pb-4">
                    <div className="flex justify-between items-start">
                      <div className="w-12 h-12 rounded-xl bg-linear-to-br from-primary/12 to-primary/5 flex items-center justify-center text-primary group-hover:scale-110 transition-transform duration-300 border border-primary/15 shadow-sm">
                        {feature.icon}
                      </div>
                      <span className="text-5xl font-bold text-muted/20 font-display select-none leading-none">
                        {String(idx + 1).padStart(2, "0")}
                      </span>
                    </div>

                    <CardTitle className="font-display text-xl tracking-tight">
                      <span className="font-light text-muted-foreground">
                        {feature.title}
                      </span>{" "}
                      <span className="font-bold text-foreground">
                        {feature.titleAccent}
                      </span>
                    </CardTitle>

                    <CardDescription asChild>
                      <div className="space-y-2">
                        <p className="text-sm font-medium text-foreground font-sans leading-relaxed">
                          {feature.lead}
                        </p>
                        <p className="text-sm leading-relaxed text-muted-foreground/90 font-light font-sans">
                          {feature.body}
                        </p>
                      </div>
                    </CardDescription>
                  </CardHeader>

                  <CardContent>
                    <div className="text-[11px] font-mono text-muted-foreground/60 border-t border-border/50 pt-3 flex justify-between items-center uppercase tracking-widest">
                      <span>{feature.tag}</span>
                      <span className="opacity-0 group-hover:opacity-100 transition-all duration-300 text-primary transform group-hover:translate-x-1">
                        →
                      </span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            <div className="text-center mt-4">
              <p className="text-[11px] text-muted-foreground/50 font-mono tracking-widest uppercase animate-pulse">
                ← Swipe to explore →
              </p>
            </div>
          </div>
        </section>
        <section className="mb-16">
          <div className="relative rounded-4xl border border-border/40 bg-linear-to-r from-primary/5 via-violet-500/5 to-blue-500/5 backdrop-blur-sm p-10 md:p-20 text-center overflow-hidden">
            <div className="absolute inset-0 bg-linear-to-br from-primary/5 via-transparent to-blue-500/5" />

            <div className="relative z-10 max-w-2xl mx-auto space-y-7">
              <h3 className="text-2xl md:text-3xl lg:text-4xl font-bold tracking-tight font-display">
                Ready to contribute?
              </h3>
              <p className="text-lg text-muted-foreground font-light font-sans">
                Join the next generation of open‑source contributors who learn
                by building.
              </p>
              <Button
                className="shadow-xl shadow-primary/20 hover:shadow-primary/40 hover:scale-105 transition-all duration-200 h-12 px-8 text-base font-semibold font-display"
                size="lg"
                onClick={() => {
                  window.location.href = "/chat";
                }}
              >
                Start your journey
                <svg
                  className="ml-2 w-4 h-4"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </Button>
            </div>
          </div>
        </section>

        <footer className="pt-10 border-t border-border/30 text-center space-y-3">
          <p className="text-sm text-muted-foreground/60 font-mono">
            Built with Django, Celery, FAISS, LangGraph & shadcn/ui
          </p>
          <p className="text-xs text-muted-foreground/50 font-mono tracking-wide">
            No code spoon‑feeding · Socratic guidance · Ethical guardrails
          </p>
        </footer>
      </main>
    </div>
  );
};

export default LandingPage;
