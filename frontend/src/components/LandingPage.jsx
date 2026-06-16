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

let introPlayed = false;

export default function LandingPage() {
  const canvasRef = useRef(null);
  const scrollContainerRef = useRef(null);
  const navTextRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [flippedIdx, setFlippedIdx] = useState(null);
  const [introPhase, setIntroPhase] = useState(
    introPlayed ? "done" : "initial",
  );
  const [isMobile, setIsMobile] = useState(false);
  const [navTextCenter, setNavTextCenter] = useState(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const dragStart = useRef({ x: 0, scrollLeft: 0 });

  useEffect(() => {
    const measure = () => {
      if (navTextRef.current) {
        const rect = navTextRef.current.getBoundingClientRect();
        setNavTextCenter({
          x: rect.left + rect.width / 2,
          y: rect.top + rect.height / 2,
        });
      }
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  useEffect(() => {
    if (introPhase !== "initial") return;
    const t1 = setTimeout(() => setIntroPhase("left"), 1200);
    const t2 = setTimeout(() => setIntroPhase("right"), 3800);
    const t3 = setTimeout(() => setIntroPhase("reveal"), 7500);
    const t4 = setTimeout(() => {
      setIntroPhase("done");
      introPlayed = true;
      window.scrollTo({ top: 0, behavior: "instant" });
    }, 9200);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      clearTimeout(t4);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 768);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  const toggleFlip = (idx) => {
    setFlippedIdx(flippedIdx === idx ? null : idx);
  };

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
        color: "rgba(37, 65, 178, 0.65)",
        baseLeft: 0.2,
        widthFactor: 0.35,
        speed: 0.0025,
      },
      {
        color: "rgba(16, 152, 247, 0.58)",
        baseLeft: 0.4,
        widthFactor: 0.3,
        speed: 0.003,
      },
      {
        color: "rgba(86, 227, 159, 0.50)",
        baseLeft: 0.55,
        widthFactor: 0.35,
        speed: 0.002,
      },
      {
        color: "rgba(37, 65, 178, 0.52)",
        baseLeft: 0.75,
        widthFactor: 0.28,
        speed: 0.0035,
      },
      {
        color: "rgba(16, 152, 247, 0.45)",
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
        "Your developer profile maps your skills against live GitHub issues using a FAISS-powered semantic graph for precise, real-time matching.",
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
      title: "Knowledge Graph",
      subtitle: "Skill Visualization",
      icon: <BarChart3 className="w-5 h-5" />,
      description:
        "Your verified skills form an interconnected graph that grows with every contribution. Watch conceptual relationships strengthen as you demonstrate deeper understanding across repositories.",
      accent: "#56E39F",
      stat: "∞",
      statLabel: "Skill Connections",
    },
  ];

  return (
    <div className="relative min-h-screen overflow-x-hidden transition-colors duration-500 font-sans bg-[#000000] text-white">
      {/* ── Full-Page Aurora Background ── */}
      <canvas
        ref={canvasRef}
        className="fixed inset-0 pointer-events-none z-0 mix-blend-normal"
        style={{ opacity: 0.8 }}
      />

      {/* ── Intro Overlay ── */}
      {introPhase !== "done" && (
        <div className="fixed inset-0 z-50 pointer-events-none">
          {/* Black background — slides up (slow-fast-slow bezier) */}
          <div
            className="absolute inset-0 bg-black"
            style={{
              transform:
                introPhase === "reveal" ? "translateY(-100%)" : "translateY(0)",
              transition: "transform 1.5s cubic-bezier(0.8, 0, 0.2, 1)",
            }}
          />

          {/* Left side text — above title on mobile, left on desktop */}
          <div
            className={
              isMobile
                ? "absolute top-[15%] left-1/2 -translate-x-1/2 w-[85vw] max-w-[320px] text-center"
                : "absolute left-8 md:left-16 top-1/2 -translate-y-1/2 max-w-[240px] text-right"
            }
          >
            <p
              className="text-[12px] md:text-sm tracking-wider text-white/90 leading-relaxed uppercase text-center md:text-right"
              style={{
                fontFamily: "Mileast, serif",
                opacity:
                  introPhase === "reveal"
                    ? 0
                    : introPhase === "initial"
                      ? 0
                      : 1,
                transition: "opacity 1.5s ease-in-out",
              }}
            >
              &ldquo;Open source is not just about code — it is about community,
              collaboration, and building something bigger than yourself.&rdquo;
            </p>
          </div>

          {/* Centered title — shrinks to navbar text position on reveal */}
          <div
            className="absolute inset-0 flex items-center justify-center"
            style={{
              transform:
                introPhase === "reveal" && navTextCenter
                  ? `translateX(${navTextCenter.x - window.innerWidth / 2}px) translateY(${navTextCenter.y - window.innerHeight / 2}px) scale(0.15)`
                  : "translateX(0) translateY(0) scale(1)",
              transition: "transform 1.5s cubic-bezier(0.8, 0, 0.2, 1)",
            }}
          >
            <h1
              className="text-7xl sm:text-8xl md:text-9xl font-black tracking-tight text-white leading-none"
              style={{ fontFamily: "Orange Vintage, serif" }}
            >
              RepoInsight
            </h1>
          </div>

          {/* Right side text — below title on mobile, right on desktop */}
          <div
            className={
              isMobile
                ? "absolute bottom-[15%] left-1/2 -translate-x-1/2 w-[85vw] max-w-[320px] text-center"
                : "absolute right-8 md:right-16 top-1/2 -translate-y-1/2 max-w-[240px] text-left"
            }
          >
            <p
              className="text-[12px] md:text-sm tracking-wider text-white/90 leading-relaxed uppercase text-center md:text-left"
              style={{
                fontFamily: "Mileast, serif",
                opacity:
                  introPhase === "reveal"
                    ? 0
                    : introPhase === "initial" || introPhase === "left"
                      ? 0
                      : 1,
                transition: "opacity 1.5s ease-in-out",
              }}
            >
              &ldquo;Every contribution starts with a single issue. Every expert
              was once a beginner who refused to give up.&rdquo;
            </p>
          </div>
        </div>
      )}

      {/* ── Navigation (always visible, behind intro initially) ── */}
      <nav className="fixed top-4 left-4 right-4 md:left-8 md:right-8 z-40 flex items-center justify-between px-6 md:px-10 py-4 bg-black/50 border border-white/10 backdrop-blur-xl rounded-2xl">
        <div className="flex items-center gap-2.5 font-mono text-sm font-black tracking-widest uppercase">
          <span
            ref={navTextRef}
            className="text-white"
            style={{ opacity: introPhase === "done" ? 1 : 0 }}
          >
            RepoInsight
          </span>
        </div>

        {/* Desktop nav links */}
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

          {/* Hamburger — visible on mobile only */}
          <button
            className="md:hidden focus:outline-none flex items-center justify-center w-9 h-9 rounded-xl bg-white/10 border border-white/20 hover:bg-white/20 transition-colors text-white text-xl leading-none"
            onClick={() => setMobileMenuOpen((prev) => !prev)}
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? "✕" : "☰"}
          </button>
        </div>
      </nav>

      {/* Mobile menu overlay + panel */}
      {isMobile && mobileMenuOpen && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/70 backdrop-blur-sm"
            onClick={() => setMobileMenuOpen(false)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Escape" && setMobileMenuOpen(false)}
          />
          <div className="relative bg-black/95 border border-white/10 backdrop-blur-xl rounded-2xl p-6 mx-4 w-full max-w-sm">
            {/* Close button inside the panel */}
            <button
              className="absolute top-3 right-3 w-8 h-8 flex items-center justify-center text-white/60 hover:text-white rounded-xl hover:bg-white/5 transition-colors"
              onClick={() => setMobileMenuOpen(false)}
              aria-label="Close menu"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path
                  d="M3 3L13 13M13 3L3 13"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              </svg>
            </button>
            <div className="flex flex-col gap-2 pt-4">
              {[
                { label: "How It Works", href: "#how-it-works" },
                { label: "Journey", href: "#journey" },
                { label: "Architecture", href: "#architecture" },
              ].map((item) => (
                <a
                  key={item.href}
                  href={item.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className="font-mono text-sm tracking-widest uppercase text-white/70 hover:text-white transition-colors py-3 px-4 rounded-xl hover:bg-white/5"
                >
                  {item.label}
                </a>
              ))}
              <hr className="border-white/10 my-1" />
              <Link
                to="/login"
                onClick={() => setMobileMenuOpen(false)}
                className="font-mono text-sm tracking-widest uppercase text-white/70 hover:text-white transition-colors py-3 px-4 rounded-xl hover:bg-white/5"
              >
                Log In
              </Link>
              <Link
                to="/signup"
                onClick={() => setMobileMenuOpen(false)}
                className="font-mono text-sm tracking-widest uppercase text-white/90 bg-white/10 hover:bg-white/20 transition-colors py-3 px-4 rounded-xl"
              >
                Get Started
              </Link>
            </div>
          </div>
        </div>
      )}

      <div className="relative">
        {/* ── Hero Section ── */}
        <main className="relative z-10 max-w-5xl mx-auto px-6 text-center flex flex-col items-center justify-center min-h-screen">
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
              GitHub issues — then guides you through understanding with
              Socratic questioning until you genuinely know the code.
            </p>

            <div className="flex flex-col sm:flex-row items-center gap-4">
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

            {isMobile ? (
              <div className="flex flex-col gap-4">
                {howItWorksCards.map((card, idx) => (
                  <div
                    key={idx}
                    className="w-full"
                    style={{ perspective: "1000px" }}
                  >
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.4, delay: idx * 0.05 }}
                    >
                      <div
                        className="transition-transform duration-500 ease-in-out rounded-2xl border border-white/10 grid [grid-template-areas:'stack'] min-h-[280px] cursor-pointer"
                        style={{
                          transformStyle: "preserve-3d",
                          transform:
                            flippedIdx === idx
                              ? "rotateY(180deg)"
                              : "rotateY(0deg)",
                        }}
                        onClick={() => toggleFlip(idx)}
                        role="button"
                        tabIndex={0}
                        onKeyDown={(e) =>
                          (e.key === "Enter" || e.key === " ") &&
                          toggleFlip(idx)
                        }
                      >
                        <div className="[grid-area:stack] backface-hidden rounded-2xl bg-black p-6">
                          <div className="flex items-start justify-between mb-6">
                            <div
                              className="w-9 h-9 rounded-xl flex items-center justify-center"
                              style={{
                                backgroundColor: `${card.accent}18`,
                                color: card.accent,
                              }}
                            >
                              {card.icon}
                            </div>
                            <span
                              className="text-4xl font-black font-mono opacity-40"
                              style={{ color: card.accent }}
                            >
                              {card.num}
                            </span>
                          </div>
                          <div className="mb-1">
                            <span
                              className="text-[10px] font-mono tracking-widest uppercase opacity-60"
                              style={{ color: card.accent }}
                            >
                              {card.subtitle}
                            </span>
                          </div>
                          <h3 className="text-lg font-serif font-bold mb-3 leading-tight text-white">
                            {card.title}
                          </h3>
                          <p className="text-sm text-white/50 leading-relaxed font-sans">
                            Tap to explore &rarr;
                          </p>
                        </div>
                        <div
                          className="[grid-area:stack] backface-hidden rounded-2xl bg-white p-6"
                          style={{ transform: "rotateY(180deg)" }}
                        >
                          <div className="flex items-start justify-between mb-5">
                            <div
                              className="w-9 h-9 rounded-xl flex items-center justify-center"
                              style={{
                                backgroundColor: `${card.accent}15`,
                                color: card.accent,
                              }}
                            >
                              {card.icon}
                            </div>
                            <span
                              className="text-2xl font-black font-mono opacity-20"
                              style={{ color: card.accent }}
                            >
                              {card.num}
                            </span>
                          </div>
                          <p className="text-sm text-black/80 leading-relaxed font-sans mb-5">
                            {card.description}
                          </p>
                          <div className="flex items-baseline gap-2">
                            <span
                              className="text-2xl font-bold font-serif"
                              style={{ color: card.accent }}
                            >
                              {card.stat}
                            </span>
                            <span className="text-[10px] font-mono uppercase tracking-widest text-black/40">
                              {card.statLabel}
                            </span>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  </div>
                ))}
              </div>
            ) : (
              // eslint-disable-next-line jsx-a11y/no-static-element-interactions
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
                // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "ArrowRight")
                    scrollContainerRef.current?.scrollBy({
                      left: 300,
                      behavior: "smooth",
                    });
                  if (e.key === "ArrowLeft")
                    scrollContainerRef.current?.scrollBy({
                      left: -300,
                      behavior: "smooth",
                    });
                }}
              >
                {howItWorksCards.map((card, idx) => (
                  <div
                    key={idx}
                    className="shrink-0 w-85 md:w-100 snap-start"
                    style={{ perspective: "1000px" }}
                    onMouseEnter={() => setFlippedIdx(idx)}
                    onMouseLeave={() => setFlippedIdx(null)}
                  >
                    <motion.div
                      initial={{ opacity: 0, y: 30 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true, margin: "-50px" }}
                      transition={{ duration: 0.5, delay: idx * 0.1 }}
                    >
                      <div
                        className="transition-transform duration-700 ease-in-out rounded-2xl border border-white/10 grid [grid-template-areas:'stack'] h-[420px]"
                        style={{
                          transformStyle: "preserve-3d",
                          transform:
                            flippedIdx === idx
                              ? "rotateY(180deg)"
                              : "rotateY(0deg)",
                        }}
                      >
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
                        <div
                          className="[grid-area:stack] backface-hidden rounded-2xl bg-white p-8"
                          style={{ transform: "rotateY(180deg)" }}
                        >
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
                      </div>
                    </motion.div>
                  </div>
                ))}
              </div>
            )}
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
              React · Django · PostgreSQL · Redis
            </h2>
            <p className="text-sm leading-relaxed font-sans max-w-2xl mx-auto mb-10 opacity-70">
              A modern stack powering FAISS-powered semantic search, async
              Celery workers, LangGraph agent orchestration, and real-time
              Socratic tutoring with Groq / Cloudflare inference — all wrapped
              in a polished Vite + Tailwind frontend.
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
    </div>
  );
}
