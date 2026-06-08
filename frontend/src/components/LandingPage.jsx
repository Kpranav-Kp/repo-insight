// frontend/src/components/LandingPage.jsx
import { useRef, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { ThemeToggle } from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";

/* ─── Particle Canvas (hero background) ─── */
const HeroCanvas = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let w = (canvas.width = canvas.offsetWidth);
    let h = (canvas.height = canvas.offsetHeight);
    let raf;

    // Particles
    const N = 80;
    const particles = Array.from({ length: N }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      r: 0.6 + Math.random() * 1.4,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25,
      alpha: 0.2 + Math.random() * 0.6,
    }));

    const MAX_DIST = 130;

    const resize = () => {
      w = canvas.width = canvas.offsetWidth;
      h = canvas.height = canvas.offsetHeight;
      particles.forEach((p) => {
        p.x = Math.random() * w;
        p.y = Math.random() * h;
      });
    };
    window.addEventListener("resize", resize);

    const draw = () => {
      ctx.clearRect(0, 0, w, h);

      // Move particles
      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = w;
        if (p.x > w) p.x = 0;
        if (p.y < 0) p.y = h;
        if (p.y > h) p.y = 0;
      });

      // Draw connections
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < MAX_DIST) {
            const alpha = (1 - dist / MAX_DIST) * 0.18;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(139,92,246,${alpha})`;
            ctx.lineWidth = 0.8;
            ctx.stroke();
          }
        }
      }

      // Draw dots
      particles.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(167,139,250,${p.alpha})`;
        ctx.fill();
      });

      raf = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
    />
  );
};

/* ─── Terminal typewriter ─── */
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
      <span className="animate-pulse text-violet-400 ml-0.5">▊</span>
    </span>
  );
};

/* ─── Semantic Graph (improved for dark theme) ─── */
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

  // Helper to render status badge without nested ternary
  const renderStatusBadge = () => {
    if (gPhase === "score") {
      return (
        <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-violet-500/10 border border-violet-500/25 backdrop-blur-md">
          <span className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" />
          <span className="text-xs font-medium text-violet-400 font-mono tracking-wide">
            94% alignment found
          </span>
        </div>
      );
    }
    if (gPhase === "edge") {
      return (
        <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-amber-500/10 border border-amber-500/25 backdrop-blur-md">
          <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
          <span className="text-xs font-medium text-amber-400 font-mono tracking-wide">
            issue #42 selected
          </span>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-violet-500/5 border border-violet-500/15 backdrop-blur-md">
        <span className="w-2 h-2 rounded-full bg-violet-400 animate-pulse" />
        <span className="text-xs font-medium text-violet-400 font-mono tracking-wide">
          semantic graph active
        </span>
      </div>
    );
  };

  return (
    <div className="relative w-full h-full min-h-105 flex items-center justify-center">
      {/* Ambient glow */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(139,92,246,0.15)_0%,transparent_70%)] rounded-2xl" />

      <div
        className={`absolute inset-0 flex items-center justify-center transition-opacity duration-700 ${gPhase === "score" ? "opacity-0 scale-95" : "opacity-100 scale-100"}`}
      >
        <svg viewBox="0 0 400 400" className="w-full h-full max-w-110">
          <defs>
            <filter id="glow2" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="6" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <radialGradient id="nodeGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="rgba(139,92,246,0.3)" />
              <stop offset="100%" stopColor="rgba(139,92,246,0)" />
            </radialGradient>
          </defs>

          {/* Orbit rings */}
          <circle
            cx="200"
            cy="200"
            r="150"
            fill="none"
            stroke="rgba(139,92,246,0.12)"
            strokeWidth="1"
            strokeDasharray="3 6"
          />
          <circle
            cx="200"
            cy="200"
            r="100"
            fill="none"
            stroke="rgba(139,92,246,0.08)"
            strokeWidth="1"
          />

          {/* Edges */}
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
                  stroke={isIssue ? "#f59e0b" : s.color}
                  strokeWidth={isIssue ? 2.5 : 1}
                  strokeDasharray={isIssue ? "none" : "4 6"}
                  opacity={(() => {
                    if (isIssue) return gPhase === "edge" ? 0.9 : 0.4;
                    return 0.3;
                  })()}
                  className="transition-all duration-500"
                />
                {isIssue && gPhase === "edge" && (
                  <circle r="3.5" fill="#f59e0b" filter="url(#glow2)">
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

          {/* Skill nodes */}
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
                  r="20"
                  fill="rgba(15,10,30,0.85)"
                  stroke={s.color}
                  strokeWidth="2"
                />
                <text
                  x={x}
                  y={y + 4}
                  textAnchor="middle"
                  fill="rgba(255,255,255,0.9)"
                  style={{
                    fontSize: "9px",
                    fontFamily: "JetBrains Mono, monospace",
                    fontWeight: 700,
                  }}
                >
                  {s.name}
                </text>
              </g>
            );
          })}

          {/* Center hub */}
          <g>
            <circle
              cx="200"
              cy="200"
              r="55"
              fill="none"
              stroke="rgba(139,92,246,0.15)"
              strokeWidth="1"
            >
              <animate
                attributeName="r"
                values="55;68;55"
                dur="4s"
                repeatCount="indefinite"
              />
              <animate
                attributeName="opacity"
                values="1;0;1"
                dur="4s"
                repeatCount="indefinite"
              />
            </circle>
            <circle
              cx="200"
              cy="200"
              r="28"
              fill="rgba(15,10,30,0.9)"
              stroke="#8b5cf6"
              strokeWidth="2.5"
              filter="url(#glow2)"
            />
            <text
              x="200"
              y="205"
              textAnchor="middle"
              fill="#a78bfa"
              style={{
                fontSize: "10px",
                fontFamily: "JetBrains Mono, monospace",
                fontWeight: 700,
                letterSpacing: "0.15em",
              }}
            >
              REPO
            </text>
          </g>
        </svg>
      </div>

      {/* Score view */}
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
              stroke="rgba(139,92,246,0.15)"
              strokeWidth="7"
            />
            <circle
              cx="60"
              cy="60"
              r="52"
              fill="none"
              stroke="#8b5cf6"
              strokeWidth="7"
              strokeLinecap="round"
              strokeDasharray={326.73}
              strokeDashoffset={gPhase === "score" ? 19.6 : 326.73}
              className="transition-all duration-1200 ease-out"
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-4xl font-bold text-white tracking-tight">
              94%
            </span>
            <span className="text-[10px] font-mono text-violet-400/70 uppercase tracking-[0.2em] mt-1">
              skill match
            </span>
          </div>
        </div>
        <p className="mt-6 text-sm text-gray-400 font-light max-w-50 text-center leading-relaxed">
          Your profile aligns strongly with{" "}
          <span className="text-violet-400 font-medium">http</span> and auth
          patterns.
        </p>
      </div>

      {/* Status badge – no nested ternary */}
      <div className="absolute bottom-5 left-1/2 -translate-x-1/2 transition-all duration-500">
        {renderStatusBadge()}
      </div>
    </div>
  );
};

/* ─── Feature cards data ─── */
const Features = [
  {
    title: "Skill",
    titleAccent: "Graph",
    lead: "Interactive proficiency mapping.",
    body: "See exactly how your skills connect to repository needs in real time. No guesswork.",
    tag: "visual onboarding",
    icon: (
      <svg
        width="20"
        height="20"
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
        width="20"
        height="20"
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
        width="20"
        height="20"
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
        width="20"
        height="20"
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
    titleAccent: "Matching",
    lead: "FAISS-powered precision.",
    body: "Issues, skills, and PRs are embedded into a living graph that adapts to your interests.",
    tag: "knowledge graph",
    icon: (
      <svg
        width="20"
        height="20"
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
        width="20"
        height="20"
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

/* ─── Main Component ─── */
const LandingPage = () => {
  const scrollContainerRef = useRef(null);

  // Smooth horizontal scroll on vertical wheel
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
    <div className="relative min-h-screen bg-[#09090f] text-white overflow-hidden selection:bg-violet-500/20">
      {/* ── Navigation ── */}
      <header className="fixed top-0 left-0 right-0 z-50 border-b border-white/6 bg-[#09090f]/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-linear-to-br from-violet-500 to-purple-700 flex items-center justify-center text-white font-bold text-sm font-mono shadow-lg shadow-violet-500/30">
              R
            </div>
            <span className="font-bold text-base tracking-tight text-white font-display">
              RepoInsight
            </span>
          </div>

          <nav className="hidden md:flex items-center gap-8">
            <a
              href="/"
              className="text-sm font-semibold text-white hover:text-violet-300 transition-colors"
            >
              Home
            </a>
            <a
              href="/chat"
              className="text-sm font-medium text-white/50 hover:text-white transition-colors"
            >
              Contribute
            </a>
            <a
              href="/profile"
              className="text-sm font-medium text-white/50 hover:text-white transition-colors"
            >
              Profile
            </a>
          </nav>

          <div className="flex items-center gap-3">
            <Link
              to="/login"
              className="hidden sm:inline-flex items-center justify-center rounded-md px-3 py-1.5 text-sm font-medium text-white/60 hover:text-white transition-colors"
            >
              Log in
            </Link>
            <Link
              to="/signup"
              className="inline-flex items-center justify-center rounded-full px-4 py-1.5 text-sm font-semibold bg-violet-600 hover:bg-violet-500 text-white shadow-lg shadow-violet-500/25 transition-all hover:shadow-violet-500/40 hover:scale-[1.03]"
            >
              Sign Up Now
            </Link>
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* ── Hero Section ── */}
      <section className="relative min-h-screen flex flex-col items-center justify-center text-center px-4 overflow-hidden">
        {/* Particle canvas background */}
        <HeroCanvas />

        {/* Radial purple glow */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_60%_at_50%_60%,rgba(99,60,180,0.22)_0%,transparent_70%)] pointer-events-none" />

        {/* Vertical stripe pattern overlay */}
        <div
          className="absolute inset-0 pointer-events-none opacity-[0.035]"
          style={{
            backgroundImage:
              "repeating-linear-gradient(90deg,rgba(255,255,255,1) 0px,rgba(255,255,255,1) 1px,transparent 1px,transparent 32px)",
          }}
        />

        {/* Content */}
        <div className="relative z-10 max-w-4xl mx-auto space-y-7 pt-20">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-violet-500/30 bg-violet-500/10 backdrop-blur-sm">
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-violet-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-violet-400" />
            </span>
            <span className="text-[11px] font-mono text-violet-300 font-medium tracking-[0.18em] uppercase">
              Open Source AI Tutor · v1.0
            </span>
          </div>

          {/* Headline */}
          <h1 className="space-y-1">
            <span className="block text-5xl sm:text-6xl md:text-7xl lg:text-[82px] font-bold tracking-tight leading-[1.04] font-display text-white/90">
              Learn by doing.
            </span>
            <span className="block text-5xl sm:text-6xl md:text-7xl lg:text-[82px] font-bold tracking-tight leading-[1.04] font-display text-transparent bg-clip-text bg-linear-to-r from-violet-400 via-purple-400 to-indigo-400">
              Not by copying.
            </span>
          </h1>

          {/* Sub-headline */}
          <p className="text-lg md:text-xl text-white/45 max-w-2xl mx-auto leading-relaxed font-light font-sans">
            RepoInsight guides you through real open-source issues with Socratic
            questions, guarded code assists, and a skill graph that grows with
            you — never handing you the answer.
          </p>

          {/* CTAs */}
          <div className="flex flex-wrap items-center justify-center gap-4 pt-2">
            <Button
              size="lg"
              className="h-12 px-8 text-base font-semibold font-display rounded-full bg-violet-600 hover:bg-violet-500 text-white shadow-xl shadow-violet-500/30 hover:shadow-violet-500/50 hover:scale-[1.04] transition-all duration-200"
              onClick={() => {
                window.location.href = "/chat";
              }}
            >
              <svg
                className="mr-2 w-4 h-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
              Start Contributing
            </Button>
            <Button
              size="lg"
              variant="ghost"
              className="h-12 px-8 text-base font-medium font-display text-white/60 hover:text-white hover:bg-white/5 rounded-full border border-white/10 hover:border-white/20 transition-all duration-200"
              onClick={() =>
                document
                  .getElementById("features")
                  ?.scrollIntoView({ behavior: "smooth" })
              }
            >
              See how it works
            </Button>
          </div>

          {/* Terminal pill */}
          <div className="pt-4 flex justify-center">
            <div className="inline-flex items-center font-mono text-[13px] text-white/30 border border-white/8 rounded-xl px-5 py-3.5 bg-white/3 backdrop-blur-sm shadow-sm">
              <span className="text-violet-400 font-bold mr-2.5 select-none">
                $
              </span>
              <TerminalTypewriter text={terminalCommand} delay={65} />
            </div>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 animate-bounce opacity-40">
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="text-white/60"
          >
            <path d="M12 5v14M5 12l7 7 7-7" />
          </svg>
        </div>
      </section>

      {/* ── Divider glow line ── */}
      <div className="w-full h-px bg-linear-to-r from-transparent via-violet-500/30 to-transparent" />

      {/* ── How it works + Semantic Graph ── */}
      <section className="relative py-28 px-4">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_50%_at_50%_50%,rgba(99,60,180,0.10)_0%,transparent_70%)] pointer-events-none" />

        <div className="max-w-7xl mx-auto">
          {/* Section header */}
          <div className="text-center max-w-2xl mx-auto mb-20 space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-violet-500/25 bg-violet-500/8 text-[11px] font-mono text-violet-400 font-medium tracking-[0.15em] uppercase">
              How it works
            </div>
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold tracking-tight font-display text-white">
              Your journey to{" "}
              <span className="text-transparent bg-clip-text bg-linear-to-r from-violet-400 to-purple-400">
                real contribution
              </span>
            </h2>
            <p className="text-white/40 text-lg font-light font-sans">
              Three steps. No shortcuts. Real growth.
            </p>
          </div>

          {/* 3-step flow */}
          <div className="grid md:grid-cols-3 gap-6 mb-24">
            {[
              {
                step: "01",
                title: "Map your skills",
                body: "Tell us what you know. We build a live semantic graph of your strengths and find the perfect issues.",
                color: "from-violet-500/20 to-purple-600/10",
                border: "border-violet-500/20",
                dot: "bg-violet-500",
              },
              {
                step: "02",
                title: "Pick an issue",
                body: "Browse AI-ranked issues matched to your skill level. No more drowning in unrelated bugs.",
                color: "from-indigo-500/20 to-blue-600/10",
                border: "border-indigo-500/20",
                dot: "bg-indigo-400",
              },
              {
                step: "03",
                title: "Solve it yourself",
                body: "The agent guides with Socratic questions. You arrive at the answer. Your understanding, your code.",
                color: "from-purple-500/20 to-pink-600/10",
                border: "border-purple-500/20",
                dot: "bg-purple-400",
              },
            ].map((item) => (
              <div
                key={item.step}
                className={`relative rounded-2xl border ${item.border} bg-linear-to-br ${item.color} p-8 backdrop-blur-sm group hover:-translate-y-1 transition-all duration-300 overflow-hidden`}
              >
                <div className="absolute top-0 left-0 right-0 h-px bg-linear-to-r from-transparent via-white/10 to-transparent" />
                <span className="text-6xl font-bold text-white/5 font-display leading-none absolute top-4 right-6 select-none">
                  {item.step}
                </span>
                <div
                  className={`w-2.5 h-2.5 rounded-full ${item.dot} mb-6 shadow-lg`}
                />
                <h3 className="text-xl font-bold text-white font-display mb-3">
                  {item.title}
                </h3>
                <p className="text-white/45 text-sm leading-relaxed font-light font-sans">
                  {item.body}
                </p>
              </div>
            ))}
          </div>

          {/* Semantic graph visual */}
          <div className="relative rounded-3xl border border-white/[0.07] bg-[#0d0b1a]/80 backdrop-blur-2xl shadow-2xl shadow-violet-500/10 overflow-hidden p-8 md:p-12">
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_70%_70%_at_50%_50%,rgba(99,60,180,0.12)_0%,transparent_70%)]" />
            <div className="absolute top-0 left-0 right-0 h-px bg-linear-to-r from-transparent via-violet-500/30 to-transparent" />

            <div className="relative z-10 grid lg:grid-cols-2 gap-12 items-center">
              <div className="space-y-6">
                <div className="space-y-3">
                  <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-violet-500/25 bg-violet-500/10 text-[11px] font-mono text-violet-400 tracking-[0.15em] uppercase">
                    Live demo
                  </div>
                  <h3 className="text-2xl md:text-3xl font-bold font-display text-white">
                    Semantic skill matching in action
                  </h3>
                  <p className="text-white/40 leading-relaxed font-light font-sans">
                    Watch how RepoInsight builds a graph of your expertise,
                    identifies issue dependencies, and surfaces the best match —
                    before you write a single line.
                  </p>
                </div>

                <div className="space-y-3">
                  {[
                    {
                      label: "Skill graph built",
                      desc: "Python, HTTP, Auth profiled",
                      color: "text-violet-400",
                    },
                    {
                      label: "Issue matched",
                      desc: "#42 — HTTP client timeout bug",
                      color: "text-amber-400",
                    },
                    {
                      label: "Alignment scored",
                      desc: "94% — strong match",
                      color: "text-emerald-400",
                    },
                  ].map((item, i) => (
                    <div key={i} className="flex items-start gap-3">
                      <span
                        className={`mt-1 w-1.5 h-1.5 rounded-full shrink-0 ${item.color.replace("text-", "bg-")}`}
                      />
                      <div>
                        <span className={`text-sm font-semibold ${item.color}`}>
                          {item.label}
                        </span>
                        <span className="text-sm text-white/30 font-mono ml-2">
                          {item.desc}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="relative h-105]">
                <SemanticGraph />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Features horizontal scroll ── */}
      <section id="features" className="relative py-20 px-4">
        <div className="max-w-7xl mx-auto">
          {/* Section header */}
          <div className="text-center max-w-3xl mx-auto mb-14 space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-violet-500/25 bg-violet-500/8 text-[11px] font-mono text-violet-400 font-medium tracking-[0.15em] uppercase">
              Core Philosophy
            </div>
            <h2 className="text-3xl md:text-4xl lg:text-5xl font-bold tracking-tight font-display text-white">
              Designed for{" "}
              <span className="text-transparent bg-clip-text bg-linear-to-r from-violet-400 to-purple-400">
                real contribution
              </span>
            </h2>
            <p className="text-lg text-white/35 font-light font-sans max-w-xl mx-auto">
              Every feature exists to help you grow — no shortcuts, no
              spoon-feeding.
            </p>
          </div>

          {/* Scroll container */}
          <div className="relative -mx-4 px-4">
            {/* Right fade */}
            <div className="absolute right-0 top-0 bottom-0 w-28 bg-linear-to-l from-[#09090f] via-[#09090f]/90 to-transparent z-10 pointer-events-none" />

            <div
              ref={scrollContainerRef}
              className="flex overflow-x-auto gap-5 pb-10 snap-x snap-mandatory scrollbar-hide -mx-4 px-4"
            >
              {Features.map((feature, idx) => (
                <div
                  key={idx}
                  className="min-w-70 md:min-w-[320px] max-w-85 snap-start shrink-0 group relative rounded-2xl border border-white/[0.07] bg-white/2 hover:bg-white/4 backdrop-blur-md hover:border-violet-500/25 transition-all duration-300 hover:-translate-y-2 hover:shadow-2xl hover:shadow-violet-500/10 overflow-hidden"
                >
                  <div className="absolute top-0 left-0 right-0 h-px bg-linear-to-r from-transparent via-violet-500/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

                  <div className="p-7 space-y-5">
                    <div className="flex justify-between items-start">
                      <div className="w-11 h-11 rounded-xl bg-violet-500/10 flex items-center justify-center text-violet-400 group-hover:scale-110 transition-transform duration-300 border border-violet-500/20">
                        {feature.icon}
                      </div>
                      <span className="text-5xl font-bold text-white/4 font-display select-none leading-none">
                        {String(idx + 1).padStart(2, "0")}
                      </span>
                    </div>

                    <div>
                      <h3 className="font-display text-lg tracking-tight mb-2">
                        <span className="font-light text-white/40">
                          {feature.title}{" "}
                        </span>
                        <span className="font-bold text-white">
                          {feature.titleAccent}
                        </span>
                      </h3>
                      <p className="text-sm font-medium text-white/70 font-sans leading-relaxed mb-1.5">
                        {feature.lead}
                      </p>
                      <p className="text-sm leading-relaxed text-white/35 font-light font-sans">
                        {feature.body}
                      </p>
                    </div>

                    <div className="text-[10px] font-mono text-white/20 border-t border-white/6 pt-4 flex justify-between items-center uppercase tracking-widest">
                      <span>{feature.tag}</span>
                      <span className="opacity-0 group-hover:opacity-100 transition-all duration-300 text-violet-400 transform group-hover:translate-x-1">
                        →
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <p className="text-center mt-2 text-[11px] text-white/20 font-mono tracking-widest uppercase animate-pulse">
              ← Scroll to explore →
            </p>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-white/6 py-10 text-center space-y-3 px-4">
        <div className="flex items-center justify-center gap-2.5 mb-4">
          <div className="w-7 h-7 rounded-lg bg-linear-to-br from-violet-500 to-purple-700 flex items-center justify-center text-white font-bold text-xs font-mono">
            R
          </div>
          <span className="font-bold text-sm tracking-tight text-white/60 font-display">
            RepoInsight
          </span>
        </div>
        <p className="text-sm text-white/25 font-mono">
          Built with Django · Celery · FAISS · LangGraph · shadcn/ui
        </p>
        <p className="text-xs text-white/15 font-mono tracking-wide">
          No code spoon-feeding · Socratic guidance · Ethical guardrails
        </p>
      </footer>
    </div>
  );
};

export default LandingPage;
