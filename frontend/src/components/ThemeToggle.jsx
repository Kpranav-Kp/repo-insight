import { Moon, Sun } from "lucide-react";
import { useState } from "react";

const getInitialTheme = () => {
  if (typeof window === "undefined") return "light";
  const stored = localStorage.getItem("theme");
  if (stored) return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
};


export const ThemeToggle = () => {
  const [theme, setTheme] = useState(getInitialTheme);

  const toggleTheme = () => {
    const next = theme === "light" ? "dark" : "light";
    const root = document.documentElement;
    if (next === "dark") {
      root.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      root.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
    setTheme(next);
  };

  return (
    <button
      onClick={toggleTheme}
      className="relative p-2.5 rounded-full border border-border/60 bg-background/60 hover:bg-accent transition-colors backdrop-blur-md"
      aria-label="Toggle Dark Mode"
    >
      <Sun
        className={`w-4.5 h-4.5 transition-all duration-300 ${theme === "dark" ? "rotate-90 scale-0 opacity-0 absolute" : "rotate-0 scale-100 opacity-100"}`}
      />
      <Moon
        className={`w-4.5 h-4.5 transition-all duration-300 ${theme === "light" ? "-rotate-90 scale-0 opacity-0 absolute" : "rotate-0 scale-100 opacity-100"}`}
      />
    </button>
  );
};
