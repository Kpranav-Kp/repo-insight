import { motion } from "framer-motion";
import {
  ArrowRight,
  Compass,
  HelpCircle,
  Shield,
  UserCircle,
  BarChart3,
  ChevronRight,
} from "lucide-react";
import { useRef, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/button";

export default function LandingPage() {
  const canvasRef = useRef(null);
  const scrollContainerRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragStart = useRef({ x: 0, scrollLeft: 0 });

  // ── 1. Full-Page Aurora Canvas ──
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let w = (canvas.width = window.innerWidth);
    let h = (canvas.height = window.innerHeight);
    let animationFrameId;
    let time = 0;

    const handleResize = () => {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", handleResize);

    const darkColumns = [
      {
        color: "rgba(37, 65, 178, 0.50)",
        baseLeft: 0.2,
        widthFactor: 0.35,
        speed: 0.0025,
      },
      {
        color: "rgba(16, 152, 247, 0.45)",
        baseLeft: 0.4,
        widthFactor: 0.3,
        speed: 0.003,
      },
      {
        color: "rgba(86, 227, 159, 0.38)",
        baseLeft: 0.55,
        widthFactor: 0.35,
        speed: 0.002,
      },
      {
        color: "rgba(37, 65, 178, 0.40)",
        baseLeft: 0.75,
        widthFactor: 0.28,
        speed: 0.0035,
      },
      {
        color: "rgba(16, 152, 247, 0.35)",
        baseLeft: 0.9,
        widthFactor: 0.25,
        speed: 0.0015,
      },
    ];

    const render = () => {
      ctx.clearRect(0, 0, w, h);
      time += 0.4;
      const activeColumns = darkColumns;

      activeColumns.forEach((col, idx) => {
        const centerOffset = Math.sin(time * col.speed + idx * 3) * (w * 0.08);
        const colCenterX = w * col.baseLeft + centerOffset;
        const colWidth = w * col.widthFactor;
        const sliceHeight = 6;

        for (let y = 0; y < h; y += sliceHeight) {
          const ripple =
            Math.sin(y * 0.004 + time * 0.008 + idx) * 50 +
            Math.cos(y * 0.012 - time * 0.004) * 25;
          const currentX = colCenterX + ripple;

          const gradient = ctx.createLinearGradient(
            currentX - colWidth / 2,
            0,
            currentX + colWidth / 2,
            0,
          );
          gradient.addColorStop(0, "transparent");
          gradient.addColorStop(0.5, col.color);
          gradient.addColorStop(1, "transparent");

          ctx.fillStyle = gradient;
          ctx.globalCompositeOperation = "screen";
          ctx.fillRect(currentX - colWidth / 2, y, colWidth, sliceHeight);
        }
      });

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  // ── 2. Horizontal Scroll Tracking + MouseWheel → Horizontal + Drag ──
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;

    const handleWheel = (e) => {
      const maxScrollLeft = container.scrollWidth - container.clientWidth;
      const hasVerticalIntent = Math.abs(e.deltaY) > Math.abs(e.deltaX);
      if (!hasVerticalIntent) return;
      const nextLeft = container.scrollLeft + e.deltaY;
      const nextClamped = Math.max(0, Math.min(maxScrollLeft, nextLeft));
      if (nextClamped === container.scrollLeft) return;
      e.preventDefault();
      container.scrollLeft = nextClamped;
    };

    container.addEventListener("wheel", handleWheel, { passive: false });

    return () => {
      container.removeEventListener("wheel", handleWheel);
    };
  }, []);

  const handleMouseDown = (e) => {
    const container = scrollContainerRef.current;
    if (!container) return;
    setIsDragging(true);
    dragStart.current = {
      x: e.pageX - container.getBoundingClientRect().left,
      scrollLeft: container.scrollLeft,
    };
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    const container = scrollContainerRef.current;
    if (!container) return;
    e.preventDefault();
    const x = e.pageX - container.getBoundingClientRect().left;
    const walk = (x - dragStart.current.x) * 1.5;
    container.scrollLeft = dragStart.current.scrollLeft - walk;
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const howItWorksCards = [
    {
      num: "01",
      title: "Skill-Based Issue Matching",
      subtitle: "FAISS Semantic Graph",
      icon: <Compass className="w-5 h-5" />,
      description:
        "Your developer profile is embedded into a low-dimensional vector space. Our FAISS-powered semantic graph maps your actual skills against live GitHub issue trees in real time.",
      accent: "#2541B2",
      stat: "95%",
      statLabel: "Match Accuracy",
    },
    {
      num: "02",
      title: "Socratic Guidance",
      subtitle: "LangGraph Multi-Agent",
      icon: <HelpCircle className="w-5 h-5" />,
      description:
        "The AI never hands you code. Independent verification agents prompt your approach, forcing you to demonstrate genuine architectural understanding before advancing.",
      accent: "#1098F7",
      stat: "0",
      statLabel: "Direct Solutions",
    },
    {
      num: "03",
      title: "Ethical Guardrails",
      subtitle: "Controlled Assistance",
      icon: <Shield className="w-5 h-5" />,
      description:
        "Maximum three boilerplate assists per session, each injected with explicit TODO comments. The system enforces learning integrity at every checkpoint.",
      accent: "#56E39F",
      stat: "3",
      statLabel: "Max Assists",
    },
    {
      num: "04",
      title: "Learner Profile",
      subtitle: "Persistent Progress Matrix",
      icon: <UserCircle className="w-5 h-5" />,
      description:
        "An immutable conceptual profile catalogs your verified codebase masteries across sessions. Skills decay and grow based on demonstrated understanding, not completion.",
      accent: "#2541B2",
      stat: "∞",
      statLabel: "Sessions Tracked",
    },
    {
      num: "05",
      title: "Built-in Evaluation",
      subtitle: "Research-Ready Metrics",
      icon: <BarChart3 className="w-5 h-5" />,
      description:
        "Precision@5, NDCG@5, Novelty and Freshness scores are computed automatically. The synthetic dataset is ready for publication-grade benchmarking.",
      accent: "#1098F7",
      stat: "4",
      statLabel: "Core Metrics",
    },
  ];

  return (
    <div className="relative min-h-screen overflow-x-hidden transition-colors duration-500 font-sans bg-[#000000] text-white">
      {/* ── Full-Page Aurora Background ── */}
      <canvas
        ref={canvasRef}
        className="fixed inset-0 pointer-events-none z-0 mix-blend-normal"
      />

      {/* ── Navigation ── */}
      <nav className="fixed top-4 left-4 right-4 md:left-8 md:right-8 z-50 flex items-center justify-between px-6 md:px-10 py-4 bg-black/50 border border-white/10 backdrop-blur-xl rounded-2xl">
        <div className="flex items-center gap-2.5 font-mono text-sm font-black tracking-widest uppercase">
          <span className="text-white">RepoInsight</span>
        </div>

        <div className="hidden md:flex items-center gap-10 font-mono text-[11px] tracking-widest uppercase text-white/70">
          <a
            href="#how-it-works"
            className="hover:text-white transition-colors"
          >
            How It Works
          </a>
          <a href="#journey" className="hover:text-white transition-colors">
            Journey
          </a>
          <a
            href="#architecture"
            className="hover:text-white transition-colors"
          >
            Architecture
          </a>
        </div>

        <div className="flex items-center gap-3">
          <Link to="/login">
            <span className="font-mono text-[11px] uppercase tracking-widest hidden sm:inline-block hover:text-white transition-colors text-white/70">
              Log In
            </span>
          </Link>
          <Link to="/signup">
            <Button
              size="sm"
              className="font-mono text-[10px] uppercase tracking-widest rounded-full px-5 py-2 transition-all duration-300 bg-[#2541B2] text-white hover:bg-[#1098F7]"
            >
              Get Started
            </Button>
          </Link>
        </div>
      </nav>

      {/* ── Hero Section ── */}
      <main className="relative z-10 max-w-5xl mx-auto px-6 pt-40 pb-32 text-center flex flex-col items-center justify-center min-h-screen">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
          className="flex flex-col items-center"
        >
          <h1 className="text-5xl sm:text-6xl md:text-7xl font-serif font-black tracking-tight leading-[1.08] mb-6 text-white">
            Learn by doing.
            <br />
            <span className="text-[#1098F7] font-serif italic">
              Not by copying.
            </span>
          </h1>

          <p className="max-w-xl text-sm md:text-base opacity-60 leading-relaxed font-sans mb-12">
            RepoInsight maps your developer taxonomy against real, unassigned
            GitHub issues — then guides you through understanding with Socratic
            questioning until you genuinely know the code.
          </p>

          <div className="flex items-center gap-4">
            <Link to="/signup">
              <Button
                size="lg"
                className="group font-mono font-bold text-xs tracking-widest uppercase rounded-full px-8 py-6 transition-all duration-300 shadow-xl hover:scale-105 bg-white text-black hover:bg-white/90"
              >
                Start Contributing
                <ArrowRight className="w-4 h-4 ml-2 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
            <a href="#how-it-works">
              <Button
                variant="outline"
                size="lg"
                className="font-mono text-xs tracking-widest uppercase rounded-full px-6 py-6 border border-white/20 text-white hover:bg-white/5"
              >
                See How It Works
              </Button>
            </a>
          </div>
        </motion.div>
      </main>

      {/* ── How It Works: Horizontal Scroll Section ── */}
      <section
        id="how-it-works"
        className="relative z-10 border-t border-white/5"
      >
        <div className="max-w-7xl mx-auto px-6 py-24">
          <div className="flex items-end justify-between mb-12">
            <div>
              <span className="text-[10px] font-mono tracking-[0.15em] uppercase text-[#1098F7] block mb-3">
                The Pipeline
              </span>
              <h2 className="text-4xl md:text-5xl font-serif font-black tracking-tight leading-[1.1] text-white">
                How It Works
              </h2>
            </div>
          </div>

          {/* eslint-disable-next-line jsx-a11y/no-static-element-interactions */}
          <div
            ref={scrollContainerRef}
            className="flex gap-6 overflow-x-auto pb-8 scrollbar-hide cursor-grab active:cursor-grabbing select-none"
            style={{
              scrollbarWidth: "none",
              msOverflowStyle: "none",
              scrollBehavior: "smooth",
            }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            {howItWorksCards.map((card, idx) => (
              <div
                key={idx}
                className="shrink-0 w-85 md:w-100 snap-start group perspective:[1000px]"
              >
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-50px" }}
                  transition={{ duration: 0.5, delay: idx * 0.1 }}
                  className="transform--3d transition-transform duration-700 ease-in-out group-hover:transform-[rotateY(180deg)] rounded-2xl border border-white/10 grid [grid-template-areas:'stack']"
                >
                  {/* Front Face */}
                  <div className="[grid-area:stack] backface-hidden rounded-2xl bg-black p-8">
                    <div className="flex items-start justify-between mb-8">
                      <div
                        className="w-10 h-10 rounded-xl flex items-center justify-center"
                        style={{
                          backgroundColor: `${card.accent}18`,
                          color: card.accent,
                        }}
                      >
                        {card.icon}
                      </div>
                      <span
                        className="text-5xl font-black font-mono opacity-40"
                        style={{ color: card.accent }}
                      >
                        {card.num}
                      </span>
                    </div>
                    <div className="mb-2">
                      <span
                        className="text-[10px] font-mono tracking-widest uppercase opacity-60"
                        style={{ color: card.accent }}
                      >
                        {card.subtitle}
                      </span>
                    </div>
                    <h3 className="text-xl font-serif font-bold mb-4 leading-tight text-white">
                      {card.title}
                    </h3>
                    <p className="text-sm text-white/50 leading-relaxed font-sans">
                      Hover to explore &rarr;
                    </p>
                  </div>
                  {/* Back Face */}
                  <div className="[grid-area:stack] backface-hidden transform-[rotateY(180deg)] rounded-2xl bg-white p-8">
                    <div className="flex items-start justify-between mb-6">
                      <div
                        className="w-10 h-10 rounded-xl flex items-center justify-center"
                        style={{
                          backgroundColor: `${card.accent}15`,
                          color: card.accent,
                        }}
                      >
                        {card.icon}
                      </div>
                      <span
                        className="text-3xl font-black font-mono opacity-20"
                        style={{ color: card.accent }}
                      >
                        {card.num}
                      </span>
                    </div>
                    <p className="text-sm text-black/80 leading-relaxed font-sans mb-6">
                      {card.description}
                    </p>
                    <div className="flex items-baseline gap-2">
                      <span
                        className="text-3xl font-bold font-serif"
                        style={{ color: card.accent }}
                      >
                        {card.stat}
                      </span>
                      <span className="text-[10px] font-mono uppercase tracking-widest text-black/40">
                        {card.statLabel}
                      </span>
                    </div>
                  </div>
                </motion.div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Your Learning Journey ── */}
      <section id="journey" className="relative z-10 border-t border-white/5">
        <div className="max-w-6xl mx-auto px-6 py-24">
          <div className="mb-16">
            <span className="text-[10px] font-mono tracking-[0.15em] uppercase text-[#1098F7] block mb-3">
              The Path
            </span>
            <h2 className="text-3xl md:text-4xl font-serif font-bold tracking-tight leading-[1.15] text-white">
              Your Learning Journey
            </h2>
            <p className="text-sm mt-3 font-sans max-w-lg leading-relaxed text-white/50">
              From discovering your first issue to building a portfolio of
              meaningful contributions — every stage is guided.
            </p>
          </div>

          <div className="space-y-16">
            {[
              {
                num: "01",
                title: "Discover",
                desc: "Your developer profile is embedded into a vector space and matched against live GitHub issues. You see only what fits your current skills — no noise, no irrelevant suggestions.",
                highlight: "Skill-issue alignment",
                color: "#2541B2",
              },
              {
                num: "02",
                title: "Understand",
                desc: "The AI never writes code for you. Instead, it questions your approach, forces you to reason about the codebase, and only advances you once genuine understanding is demonstrated.",
                highlight: "Socratic verification",
                color: "#1098F7",
              },
              {
                num: "03",
                title: "Contribute",
                desc: "Submit your first pull request with confidence. The system provides structured PR outlines and readiness reviews based on your session history and verified understanding.",
                highlight: "Guided PR submission",
                color: "#56E39F",
              },
              {
                num: "04",
                title: "Grow",
                desc: "Skills decay and grow based on demonstrated understanding, not completion. Your profile evolves as you contribute more, opening doors to increasingly complex challenges.",
                highlight: "Persistent skill graph",
                color: "#2541B2",
              },
            ].map((step, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: i * 0.1 }}
                className={`flex flex-col md:flex-row ${
                  i % 2 === 1 ? "md:flex-row-reverse" : ""
                } items-start gap-8 md:gap-16`}
              >
                <div className="flex-1">
                  <span
                    className="text-[10px] font-mono tracking-[0.15em] uppercase block mb-3"
                    style={{ color: step.color }}
                  >
                    {step.highlight}
                  </span>
                  <h3 className="text-2xl md:text-3xl font-serif font-bold mb-4 leading-tight text-white">
                    {step.title}
                  </h3>
                  <p className="text-sm leading-relaxed font-sans text-white/60">
                    {step.desc}
                  </p>
                </div>
                <div
                  className="flex-none w-16 h-16 rounded-2xl flex items-center justify-center"
                  style={{ backgroundColor: `${step.color}18` }}
                >
                  <span
                    className="text-2xl font-serif font-black"
                    style={{ color: step.color }}
                  >
                    {step.num}
                  </span>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Architecture CTA ── */}
      <section
        id="architecture"
        className="relative z-10 border-t border-white/5"
      >
        <div className="max-w-4xl mx-auto px-6 py-24 text-center">
          <span className="text-[10px] font-mono tracking-[0.15em] uppercase text-[#56E39F] block mb-3">
            System Architecture
          </span>
          <h2 className="text-3xl md:text-5xl font-display font-bold tracking-tight mb-6 text-white">
            React · Django · FAISS · Groq
          </h2>
          <p className="text-sm leading-relaxed font-sans max-w-2xl mx-auto mb-10 opacity-70">
            A modern stack powering semantic search, async Celery workers,
            LangGraph agent orchestration, and real-time Socratic tutoring — all
            wrapped in a polished Vite + Tailwind frontend.
          </p>
          <Link to="/signup">
            <Button
              size="lg"
              className="group font-mono font-bold text-xs tracking-widest uppercase rounded-full px-8 py-6 transition-all duration-300 shadow-xl hover:scale-105 bg-[#56E39F] text-black hover:bg-[#56E39F]/90"
            >
              Initialize Your Profile
              <ChevronRight className="w-4 h-4 ml-2 transition-transform group-hover:translate-x-1" />
            </Button>
          </Link>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="relative z-10 border-t border-white/5 py-12">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2 font-mono text-xs tracking-widest uppercase text-white/40">
            <span>RepoInsight</span>
          </div>
          <div className="flex items-center gap-6 text-[10px] font-mono tracking-widest uppercase text-white/30">
            <span>Django</span>
            <span>Celery</span>
            <span>FAISS</span>
            <span>LangGraph</span>
            <span>Tailwind</span>
          </div>
          <div className="flex items-center gap-4 text-[11px] font-mono tracking-widest uppercase text-white/30">
            <span className="hover:text-white transition-colors cursor-pointer">
              Portfolio
            </span>
            <span className="text-white/10">/</span>
            <span className="hover:text-white transition-colors cursor-pointer">
              LinkedIn
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
