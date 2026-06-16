import { Sun, Moon, Check, AlertCircle } from "lucide-react";
import { useState } from "react";

import { useTheme } from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { request } from "@/lib/api";

export function SettingsModal({ open, onOpenChange, user, onUsernameChange }) {
  const { theme, toggleTheme } = useTheme();
  const isDark = theme === "dark";
  const [newUsername, setNewUsername] = useState(user?.name || "");
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const handleSaveUsername = async () => {
    if (!newUsername.trim() || saving) return;
    setSaving(true);
    setError(null);
    try {
      await request("/auth/username/", {
        method: "PATCH",
        body: JSON.stringify({ username: newUsername.trim() }),
      });
      localStorage.setItem("username", newUsername.trim());
      onUsernameChange?.(newUsername.trim());
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to update username",
      );
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className={`sm:max-w-sm p-6 shadow-xl backdrop-blur-md rounded-2xl border ${
          isDark ? "border-white/8 bg-[#1a1a1a]" : "border-black/8 bg-white"
        }`}
      >
        <DialogHeader>
          <DialogTitle className={isDark ? "text-white" : "text-black"}>
            Settings
          </DialogTitle>
          <DialogDescription
            className={isDark ? "text-white/50" : "text-black/50"}
          >
            Customise your profile and appearance.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 mt-4">
          {/* Theme toggle */}
          <div className="flex items-center justify-between">
            <div>
              <p
                className={`text-sm font-medium ${isDark ? "text-white" : "text-black"}`}
              >
                Theme
              </p>
              <p
                className={`text-xs mt-0.5 ${isDark ? "text-white/40" : "text-black/40"}`}
              >
                Switch between light and dark.
              </p>
            </div>
            <button
              onClick={toggleTheme}
              className={`relative inline-flex h-8 w-14 items-center rounded-full border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#2541B2]/50 ${
                isDark
                  ? "border-white/10 bg-white/5"
                  : "border-black/10 bg-black/5"
              }`}
              aria-label={isDark ? "Switch to light" : "Switch to dark"}
            >
              <span
                className={`inline-flex h-6 w-6 items-center justify-center rounded-full transform transition-transform ${
                  isDark
                    ? "translate-x-7 bg-[#2541B2] text-white"
                    : "translate-x-1 bg-white text-[#2541B2] shadow-sm"
                }`}
              >
                {isDark ? (
                  <Moon className="h-3.5 w-3.5" />
                ) : (
                  <Sun className="h-3.5 w-3.5" />
                )}
              </span>
            </button>
          </div>

          {/* Change username */}
          <div className="space-y-3">
            <div>
              <p
                className={`text-sm font-medium ${isDark ? "text-white" : "text-black"}`}
              >
                Username
              </p>
              <p
                className={`text-xs mt-0.5 ${isDark ? "text-white/40" : "text-black/40"}`}
              >
                Change your display name.
              </p>
            </div>
            <div className="flex gap-2">
              <Input
                value={newUsername}
                onChange={(e) => setNewUsername(e.target.value)}
                placeholder="Enter new username"
                className={`flex-1 rounded-xl ${
                  isDark
                    ? "bg-white/5 border-white/10 text-white"
                    : "bg-black/5 border-black/10 text-black"
                }`}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSaveUsername();
                }}
              />
              <Button
                size="sm"
                onClick={handleSaveUsername}
                disabled={!newUsername.trim() || saving}
                className={`rounded-xl ${
                  isDark
                    ? "bg-white text-black hover:bg-white/90"
                    : "bg-[#000000] text-white hover:bg-[#2541B2]"
                }`}
              >
                {saving ? (
                  <span className="flex items-center gap-1.5">
                    <span className="animate-spin">⟳</span>
                    Saving...
                  </span>
                ) : saved ? (
                  <Check className="h-4 w-4" />
                ) : (
                  "Save"
                )}
              </Button>
            </div>
            {error && (
              <p className="flex items-center gap-1.5 text-xs text-red-500">
                <AlertCircle className="h-3.5 w-3.5" />
                {error}
              </p>
            )}
            {saved && !error && (
              <p className="text-xs text-emerald-500">Username updated!</p>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
