// frontend/src/components/chat/SettingsModal.jsx
import { Sun, Moon, Check } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";

function getInitialTheme() {
  const stored = localStorage.getItem("theme");
  if (stored) return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export function SettingsModal({ open, onOpenChange, user, onUsernameChange }) {
  const [theme, setTheme] = useState(getInitialTheme);
  const [newUsername, setNewUsername] = useState(user?.name || "");
  const [saved, setSaved] = useState(false);

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

  const handleSaveUsername = () => {
    if (newUsername.trim()) {
      localStorage.setItem("username", newUsername.trim());
      onUsernameChange?.(newUsername.trim());
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-sm border-border bg-white dark:bg-card p-6 shadow-xl backdrop-blur-md">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>
            Customise your profile and appearance.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Theme toggle */}
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-foreground">Theme</p>
              <p className="text-xs text-muted-foreground">
                Switch between light and dark.
              </p>
            </div>
            <button
              onClick={toggleTheme}
              className="relative inline-flex h-8 w-14 items-center rounded-full border border-border bg-muted transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              aria-label={
                theme === "dark" ? "Switch to light" : "Switch to dark"
              }
            >
              <span
                className={`inline-flex h-6 w-6 items-center justify-center rounded-full transform transition-transform ${
                  theme === "dark"
                    ? "translate-x-6 bg-primary text-white"
                    : "translate-x-1 bg-white text-gray-900 shadow-sm"
                }`}
              >
                {theme === "dark" ? (
                  <Moon className="h-3.5 w-3.5" />
                ) : (
                  <Sun className="h-3.5 w-3.5 text-gray-900" />
                )}
              </span>
            </button>
          </div>

          {/* Change username */}
          <div className="space-y-3">
            <div>
              <p className="text-sm font-medium text-foreground">Username</p>
              <p className="text-xs text-muted-foreground">
                Change your display name.
              </p>
            </div>
            <div className="flex gap-2">
              <Input
                value={newUsername}
                onChange={(e) => setNewUsername(e.target.value)}
                placeholder="Enter new username"
                className="flex-1"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSaveUsername();
                }}
              />
              <Button
                size="sm"
                onClick={handleSaveUsername}
                disabled={!newUsername.trim()}
              >
                {saved ? <Check className="h-4 w-4" /> : "Save"}
              </Button>
            </div>
            {saved && (
              <p className="text-xs text-emerald-500">Username updated!</p>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
