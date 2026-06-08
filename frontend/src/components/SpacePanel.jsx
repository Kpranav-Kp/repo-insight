import { useEffect, useRef, useState } from "react";

const QUOTES = [
  {
    text: "Talk is cheap. Show me the code.",
    author: "Linus Torvalds",
    role: "Creator of Linux & Git",
  },
  {
    text: "Most good programmers do programming not because they expect to get paid, but because it is fun to program.",
    author: "Linus Torvalds",
    role: "Creator of Linux & Git",
  },
  {
    text: "The best way to predict the future is to invent it.",
    author: "Alan Kay",
    role: "Pioneer of OOP & GUI",
  },
  {
    text: "Open source is not about being nice. It is about being direct and getting things done.",
    author: "Guido van Rossum",
    role: "Creator of Python",
  },
  {
    text: "Contributing to open source is one of the most rewarding things you can do as a developer.",
    author: "Brendan Eich",
    role: "Creator of JavaScript",
  },
  {
    text: "The more you share, the more you have.",
    author: "Wil Wheaton",
    role: "Open source advocate",
  },
  {
    text: "Every contribution matters, no matter how small. Start somewhere.",
    author: "Nat Friedman",
    role: "Former CEO of GitHub",
  },
  {
    text: "When you write open source code, you are writing for the future.",
    author: "Mitchell Hashimoto",
    role: "Co-founder of HashiCorp",
  },
  {
    text: "First, solve the problem. Then, write the code.",
    author: "John Johnson",
    role: "Computer Scientist",
  },
  {
    text: "The strength of open source is that it allows anyone to contribute and improve the code.",
    author: "Tim O'Reilly",
    role: "Founder of O'Reilly Media",
  },
  {
    text: "If you want to go fast, go alone. If you want to go far, go together.",
    author: "African Proverb",
    role: "Adopted by the open source community",
  },
  {
    text: "Release early. Release often. And listen to your customers.",
    author: "Eric S. Raymond",
    role: "Author of The Cathedral & the Bazaar",
  },
  {
    text: "The best projects are not born from one person's vision, but from many people building on each other's work.",
    author: "Sarah Novotny",
    role: "Open source strategist, Google",
  },
];

class ShootingStar {
  constructor(width, height, cx, cy) {
    this.width = width;
    this.height = height;
    this.cx = cx ?? width / 2;
    this.cy = cy ?? height / 2;
    this.reset();
  }

  reset() {
    // Spawn from a position near the cursor (spiral center)
    const spawnRadius = 30 + Math.random() * 80;
    const spawnAngle = Math.random() * Math.PI * 2;
    this.x0 = this.cx + spawnRadius * Math.cos(spawnAngle);
    this.y0 = this.cy + spawnRadius * Math.sin(spawnAngle);

    // Fly outward in a curved arc
    const direction = Math.random() > 0.5 ? 1 : -1;
    const distance = 150 + Math.random() * 200;
    const angle = spawnAngle + (Math.random() - 0.5) * (Math.PI / 3);

    this.x1 = this.x0 + direction * distance * Math.cos(angle);
    this.y1 = this.y0 + distance * Math.sin(angle);

    // Control point for quadratic Bézier
    const midX = (this.x0 + this.x1) / 2;
    const midY = (this.y0 + this.y1) / 2;
    const offset = 40 + Math.random() * 60;
    this.ctrlX = midX - direction * offset * Math.sin(angle);
    this.ctrlY = midY + offset * Math.cos(angle);

    this.progress = 0;
    this.duration = 50 + Math.random() * 20;
    this.trail = [];
    this.trailMaxLength = 16;
  }

  update() {
    this.progress += 1 / this.duration;
    if (this.progress > 1) {
      this.progress = 1;
      return false;
    }

    const t = this.progress;
    const x =
      (1 - t) * (1 - t) * this.x0 +
      2 * (1 - t) * t * this.ctrlX +
      t * t * this.x1;
    const y =
      (1 - t) * (1 - t) * this.y0 +
      2 * (1 - t) * t * this.ctrlY +
      t * t * this.y1;

    this.trail.push({ x, y });
    if (this.trail.length > this.trailMaxLength) this.trail.shift();
    return true;
  }

  draw(ctx) {
    if (this.trail.length < 2) return;

    ctx.beginPath();
    ctx.moveTo(this.trail[0].x, this.trail[0].y);
    for (let i = 1; i < this.trail.length; i++) {
      ctx.lineTo(this.trail[i].x, this.trail[i].y);
    }

    const head = this.trail[this.trail.length - 1];
    const tail = this.trail[0];
    const fade = Math.max(0, 1 - this.progress);

    const gradient = ctx.createLinearGradient(tail.x, tail.y, head.x, head.y);
    gradient.addColorStop(0, "rgba(99,102,241,0)");
    gradient.addColorStop(0.4, `rgba(139,92,246,${0.25 * fade})`);
    gradient.addColorStop(0.85, `rgba(167,139,250,${0.55 * fade})`);
    gradient.addColorStop(1, `rgba(255,255,255,${0.85 * fade})`);

    ctx.strokeStyle = gradient;
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.stroke();

    // Head glow
    ctx.beginPath();
    ctx.arc(head.x, head.y, 1.8, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(255,255,255,${fade * 0.9})`;
    ctx.shadowBlur = 10;
    ctx.shadowColor = "rgba(139,92,246,0.9)";
    ctx.fill();
    ctx.shadowBlur = 0;
  }
}

export default function SpacePanel() {
  const canvasRef = useRef(null);
  const [currentQuote, setCurrentQuote] = useState(0);
  const [fade, setFade] = useState(true);

  // Quote rotation
  useEffect(() => {
    const interval = setInterval(() => {
      setFade(false);
      setTimeout(() => {
        setCurrentQuote((prev) => (prev + 1) % QUOTES.length);
        setFade(true);
      }, 500);
    }, 7000);
    return () => clearInterval(interval);
  }, []);

  // Canvas: starfield + cursor-tracked shooting stars
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf;
    let w = (canvas.width = canvas.offsetWidth);
    let h = (canvas.height = canvas.offsetHeight);

    // Cursor position (starts at center)
    let cursorX = w / 2;
    let cursorY = h / 2;

    const onMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      cursorX = e.clientX - rect.left;
      cursorY = e.clientY - rect.top;
    };
    canvas.parentElement?.addEventListener("mousemove", onMouseMove);

    // Fewer static stars
    const staticStars = Array.from({ length: 25 }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      size: 0.4 + Math.random() * 1.2,
      alpha: 0.15 + Math.random() * 0.5,
      speed: 0.008 + Math.random() * 0.02,
    }));

    const activeShootingStars = [];

    const handleResize = () => {
      w = canvas.width = canvas.offsetWidth;
      h = canvas.height = canvas.offsetHeight;
      staticStars.forEach((s) => {
        s.x = Math.random() * w;
        s.y = Math.random() * h;
      });
    };
    window.addEventListener("resize", handleResize);

    const render = () => {
      ctx.clearRect(0, 0, w, h);

      // Static stars
      staticStars.forEach((star) => {
        star.alpha += star.speed;
        if (star.alpha > 0.65 || star.alpha < 0.1) star.speed = -star.speed;
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(200,200,255,${Math.max(0.1, star.alpha)})`;
        ctx.fill();
      });

      // Shooting stars (max 2 at once)
      for (let i = activeShootingStars.length - 1; i >= 0; i--) {
        const star = activeShootingStars[i];
        star.cx = cursorX;
        star.cy = cursorY;
        const alive = star.update();
        if (alive) {
          star.draw(ctx);
        } else {
          activeShootingStars.splice(i, 1);
        }
      }

      // Spawn chance: low probability, max 2 simultaneous
      if (Math.random() < 0.004 && activeShootingStars.length < 2) {
        activeShootingStars.push(new ShootingStar(w, h, cursorX, cursorY));
      }

      raf = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", handleResize);
      canvas.parentElement?.removeEventListener("mousemove", onMouseMove);
    };
  }, []);

  return (
    <div
      className="absolute inset-0 w-full h-full flex flex-col justify-center items-center px-8 text-white select-none"
      style={{
        background:
          "radial-gradient(ellipse at 40% 35%, #0b1225 0%, #060b18 45%, #030609 80%, #010204 100%)",
      }}
    >
      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />

      {/* Quote */}
      <div className="relative z-10 max-w-md text-center space-y-5 px-4">
        <span className="block font-serif text-7xl text-indigo-500/15 leading-none select-none">
          &quot;;
        </span>
        <p
          className={`font-display text-xl md:text-2xl font-medium tracking-tight leading-relaxed transition-all duration-500 transform ${
            fade ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-3"
          }`}
        >
          {QUOTES[currentQuote].text}
        </p>
        <div
          className={`transition-all duration-500 delay-100 transform ${
            fade ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-3"
          }`}
        >
          <p className="text-violet-400 font-semibold tracking-wider text-xs uppercase">
            — {QUOTES[currentQuote].author}
          </p>
          <p className="text-xs text-gray-500 mt-1 font-mono">
            {QUOTES[currentQuote].role}
          </p>
        </div>

        {/* Quote dots */}
        <div className="flex justify-center gap-1.5 pt-2">
          {QUOTES.map((_, i) => (
            <button
              key={i}
              onClick={() => {
                setFade(false);
                setTimeout(() => {
                  setCurrentQuote(i);
                  setFade(true);
                }, 250);
              }}
              className={`w-1 h-1 rounded-full transition-all duration-300 ${
                i === currentQuote ? "bg-violet-400 w-4" : "bg-white/20"
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
