import { Moon, Sun } from "lucide-react";
import { createContext, useContext, useState, useEffect } from "react";

const ThemeContext = createContext(undefined);

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("repo-insight-theme") || "dark";
    }
    return "dark";
  });

  useEffect(() => {
    const root = document.documentElement;
    localStorage.setItem("repo-insight-theme", theme);
    if (theme === "dark") {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}

export const ThemeToggle = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      className={`relative p-2 rounded-full border transition-colors backdrop-blur-md cursor-pointer ${
        theme === "dark"
          ? "border-white/10 bg-white/5 hover:bg-white/10"
          : "border-black/10 bg-black/5 hover:bg-black/10"
      }`}
      aria-label="Toggle Dark Mode"
    >
      <Sun
        className={`w-4 h-4 transition-all duration-300 ${
          theme === "dark"
            ? "rotate-90 scale-0 opacity-0 absolute"
            : "rotate-0 scale-100 opacity-100 text-[#2541B2]"
        }`}
      />
      <Moon
        className={`w-4 h-4 transition-all duration-300 ${
          theme === "dark"
            ? "rotate-0 scale-100 opacity-100 text-[#1098F7]"
            : "-rotate-90 scale-0 opacity-0 absolute"
        }`}
      />
    </button>
  );
};
