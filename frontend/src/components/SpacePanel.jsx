import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

const quotes = [
  {
    text: "Talk is cheap. Show me the code.",
    author: "Linus Torvalds",
    role: "Creator of Linux & Git",
  },
  {
    text: "Programs must be written for people to read, and only incidentally for machines to execute.",
    author: "Harold Abelson",
    role: "Co-Author of SICP",
  },
  {
    text: "The best way to predict the future is to invent it.",
    author: "Alan Kay",
    role: "Pioneer of Object-Oriented Programming",
  },
  {
    text: "Debugging is twice as hard as writing the code in the first place. Therefore, if you write the code as cleverly as possible, you are by definition not smart enough to debug it.",
    author: "Brian Kernighan",
    role: "Co-Creator of Unix",
  },
  {
    text: "Always code as if the guy who ends up maintaining your code will be a violent psychopath who knows where you live.",
    author: "John Woods",
    role: "Software Engineer & Author",
  },
  {
    text: "First, solve the problem. Then, write the code.",
    author: "John Johnson",
    role: "Computer Scientist",
  },
  {
    text: "The most dangerous phrase in the language is: 'We've always done it this way.'",
    author: "Grace Hopper",
    role: "Computer Science Pioneer",
  },
  {
    text: "Simplicity is a prerequisite for reliability.",
    author: "Edsger W. Dijkstra",
    role: "Turing Award Laureate",
  },
  {
    text: "Any fool can write code that a computer can understand. Good programmers write code that humans can understand.",
    author: "Martin Fowler",
    role: "Author of Refactoring",
  },
  {
    text: "The function of good software is to make the complex appear to be simple.",
    author: "Grady Booch",
    role: "Co-Creator of UML",
  },
  {
    text: "Software is a great combination between artistry and engineering.",
    author: "Bill Gates",
    role: "Co-Founder of Microsoft",
  },
  {
    text: "Walking on water and developing software from a specification are easy if both are frozen.",
    author: "Edward V. Berard",
    role: "Software Engineering Author",
  },
];

export default function SpacePanel() {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [currentQuote, setCurrentQuote] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentQuote((prev) => (prev + 1) % quotes.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let animationFrameId;
    let bgStars = [];
    let orbitalStars = [];
    const TARGET_COUNT = 10;
    const MIN_RADIUS = 80;
    const MAX_RADIUS = 280;

    const resize = () => {
      canvas.width = containerRef.current.clientWidth;
      canvas.height = containerRef.current.clientHeight;
      generateBgStars();
    };
    window.addEventListener("resize", resize);
    resize();

    function generateBgStars() {
      bgStars = [];
      for (let i = 0; i < 80; i++) {
        bgStars.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          size: 0.3 + Math.random() * 0.8,
          baseOpacity: 0.06 + Math.random() * 0.18,
          phase: Math.random() * Math.PI * 2,
          speed: 0.001 + Math.random() * 0.003,
        });
      }
    }

    class ShootingStar {
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height * 0.4;
        const angle = Math.PI / 3 + (Math.random() * Math.PI) / 3;
        const speed = 4 + Math.random() * 5;
        this.vx = Math.cos(angle) * speed;
        this.vy = Math.sin(angle) * speed;
        this.life = 40 + Math.floor(Math.random() * 30);
        this.maxLife = this.life;
        this.size = 1.0 + Math.random() * 1.5;
        this.history = [];
        this.trailLength = 12 + Math.floor(Math.random() * 8);
      }

      update() {
        this.life--;
        this.history.push({ x: this.x, y: this.y });
        if (this.history.length > this.trailLength) {
          this.history.shift();
        }
        this.x += this.vx;
        this.y += this.vy;
      }

      draw() {
        const lifeRatio = this.life / this.maxLife;
        for (let i = 1; i < this.history.length; i++) {
          const t = i / this.history.length;
          const alpha = t * lifeRatio * 0.5;
          if (alpha <= 0) continue;
          ctx.beginPath();
          ctx.arc(
            this.history[i].x,
            this.history[i].y,
            this.size * t * 0.7,
            0,
            Math.PI * 2,
          );
          ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
          ctx.fill();
        }
        const headAlpha = lifeRatio * 0.9;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size * 1.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${headAlpha})`;
        ctx.fill();
      }

      isDead() {
        return this.life <= 0;
      }
    }

    class OrbitalStar {
      constructor(cx, cy) {
        this.angle = Math.random() * Math.PI * 2;
        this.radius = MIN_RADIUS + Math.random() * (MAX_RADIUS - MIN_RADIUS);
        this.speed =
          (0.002 + Math.random() * 0.006) * (this.radius > 200 ? 0.5 : 1.0);
        this.size = 1.5 + Math.random() * 2.0;
        this.maxOpacity = 0.4 + Math.random() * 0.5;
        this.opacity = 0;
        this.cx = cx;
        this.cy = cy;
        this.x = cx + Math.cos(this.angle) * this.radius;
        this.y = cy + Math.sin(this.angle) * this.radius;
        this.maxLife = 360 + Math.random() * 360;
        this.life = 0;
        this.history = [];
        this.trailLength = 4 + Math.floor(Math.random() * 4);
      }

      update(cx, cy) {
        this.life++;
        this.cx = cx;
        this.cy = cy;
        this.angle += this.speed;
        this.x = this.cx + Math.cos(this.angle) * this.radius;
        this.y = this.cy + Math.sin(this.angle) * this.radius;
        this.history.push({ x: this.x, y: this.y });
        if (this.history.length > this.trailLength) {
          this.history.shift();
        }
        const fadeIn = 60;
        const fadeOut = 90;
        if (this.life < fadeIn) {
          this.opacity = (this.life / fadeIn) * this.maxOpacity;
        } else if (this.life > this.maxLife - fadeOut) {
          this.opacity =
            ((this.maxLife - this.life) / fadeOut) * this.maxOpacity;
        } else {
          this.opacity = this.maxOpacity;
        }
      }

      draw() {
        if (this.opacity <= 0) return;
        for (let i = 1; i < this.history.length; i++) {
          const t = i / this.history.length;
          const trailAlpha = t * this.opacity * 0.2;
          if (trailAlpha <= 0) continue;
          ctx.beginPath();
          ctx.arc(
            this.history[i].x,
            this.history[i].y,
            this.size * 0.4 * t,
            0,
            Math.PI * 2,
          );
          ctx.fillStyle = `rgba(255, 255, 255, ${trailAlpha})`;
          ctx.fill();
        }
        const glowAlpha = this.opacity * 0.1;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size * 3, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${glowAlpha})`;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${this.opacity})`;
        ctx.fill();
      }

      isDead() {
        return this.life >= this.maxLife;
      }
    }

    let shootingStars = [];

    for (let i = 0; i < 5; i++) {
      orbitalStars.push(new OrbitalStar(canvas.width / 2, canvas.height / 2));
    }

    const loop = () => {
      ctx.fillStyle = "#060B15";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      bgStars.forEach((star) => {
        const twinkle =
          Math.sin(Date.now() * star.speed + star.phase) * 0.35 + 0.65;
        const alpha = star.baseOpacity * twinkle;
        if (alpha <= 0) return;
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
        ctx.fill();
      });

      if (orbitalStars.length < TARGET_COUNT && Math.random() < 0.04) {
        orbitalStars.push(new OrbitalStar(canvas.width / 2, canvas.height / 2));
      }

      orbitalStars = orbitalStars.filter((s) => !s.isDead());
      orbitalStars.forEach((s) => {
        s.update(canvas.width / 2, canvas.height / 2);
        s.draw();
      });

      if (Math.random() < 0.008) {
        shootingStars.push(new ShootingStar());
      }
      shootingStars = shootingStars.filter((s) => !s.isDead());
      shootingStars.forEach((s) => {
        s.update();
        s.draw();
      });

      animationFrameId = requestAnimationFrame(loop);
    };
    loop();

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full bg-[#060B15] flex flex-col justify-between p-12 overflow-hidden select-none"
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 z-0 pointer-events-none"
      />

      <div className="relative z-10 flex flex-col items-start justify-center min-h-0 mt-8 md:mt-12">
        <Link
          to="/"
          className="text-2xl md:text-3xl font-orangevoyage tracking-widest text-white/90"
        >
          RepoInsight
        </Link>
      </div>

      <div className="relative z-10 max-w-md my-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentQuote}
            initial={{ opacity: 0, x: -10, filter: "blur(4px)" }}
            animate={{ opacity: 1, x: 0, filter: "blur(0px)" }}
            exit={{ opacity: 0, x: 10, filter: "blur(4px)" }}
            transition={{ duration: 0.6, ease: "easeInOut" }}
          >
            <p className="text-xl md:text-2xl font-medium tracking-tight text-white leading-relaxed font-sans">
              &ldquo;{quotes[currentQuote].text}&rdquo;
            </p>
            <div className="mt-4 flex items-center gap-2">
              <span className="text-xs font-mono tracking-wider text-[#1098F7] uppercase">
                {quotes[currentQuote].author}
              </span>
              <span className="text-white/20 font-mono text-xs">{"//"}</span>
              <span className="text-xs font-mono text-white/40">
                {quotes[currentQuote].role}
              </span>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="relative z-10 text-[9px] font-mono tracking-wider text-white/20 uppercase">
        &copy; Repo Insight // Autonomous Core
      </div>
    </div>
  );
}
