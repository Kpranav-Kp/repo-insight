import { Popover } from "@base-ui/react/popover";
import {
  ChevronLeft,
  ChevronRight,
  LogOut,
  Settings,
  Plus,
} from "lucide-react";
import { useEffect, useState } from "react";

import { useTheme } from "@/components/ThemeToggle";
import { loadSessions, deleteSession } from "@/lib/sessionStore";
import { cn } from "@/lib/utils";

import { SettingsModal } from "./SettingsModal";

export function ChatSidebar({
  onLogout,
  user,
  onSelectSession,
  activeSessionId,
  onNewChat,
  onUsernameChange,
}) {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const [expanded, setExpanded] = useState(true);
  const [sessions, setSessions] = useState([]);
  const [settingsOpen, setSettingsOpen] = useState(false);

  useEffect(() => {
    const refresh = () => setSessions(loadSessions());
    refresh();
    window.addEventListener("sessions-updated", refresh);
    window.addEventListener("storage", refresh);
    return () => {
      window.removeEventListener("sessions-updated", refresh);
      window.removeEventListener("storage", refresh);
    };
  }, []);

  const initials = (user?.name || user?.email || "U")
    .split(" ")
    .map((s) => s[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();

  const handleDelete = (e, id) => {
    e.stopPropagation();
    deleteSession(id);
  };

  return (
    <>
      <SettingsModal
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        user={user}
        onUsernameChange={onUsernameChange}
      />

      <aside
        className={cn(
          "flex h-full flex-col transition-all duration-300",
          expanded ? "w-64" : "w-14",
        )}
      >
        {/* Top: expand/collapse + New Chat */}
        <div className="flex shrink-0 items-center gap-2 p-3">
          <button
            onClick={() => setExpanded(!expanded)}
            className={`p-2 rounded-xl transition-colors ${
              isDark
                ? "text-white/40 hover:bg-white/5 hover:text-white"
                : "text-black/40 hover:bg-black/5 hover:text-black"
            }`}
            aria-label={expanded ? "Collapse sidebar" : "Expand sidebar"}
          >
            {expanded ? (
              <ChevronLeft className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </button>

          {expanded && (
            <button
              onClick={onNewChat}
              className={`flex-1 flex items-center justify-center gap-1.5 text-xs font-medium py-2 rounded-xl transition-all ${
                isDark
                  ? "bg-white text-black hover:bg-white/90"
                  : "bg-[#000000] text-white hover:bg-[#2541B2]"
              }`}
            >
              <Plus className="h-3.5 w-3.5" />
              New Chat
            </button>
          )}
        </div>

        {/* Session list */}
        {expanded && (
          <div className="flex-1 overflow-y-auto px-3 py-2 space-y-0.5">
            {sessions.length === 0 && (
              <p
                className={`text-xs text-center py-8 ${isDark ? "text-white/30" : "text-black/30"}`}
              >
                No saved sessions
              </p>
            )}

            {sessions.map((s) => (
              <div key={s.localId} className="group relative">
                <button
                  onClick={() => onSelectSession?.(s)}
                  className={cn(
                    "w-full text-left truncate text-xs p-2.5 rounded-xl transition-all pr-8",
                    activeSessionId === s.localId
                      ? isDark
                        ? "bg-white/10 text-white"
                        : "bg-black/5 text-black"
                      : isDark
                        ? "text-white/40 hover:bg-white/5 hover:text-white/70"
                        : "text-black/40 hover:bg-black/5 hover:text-black/70",
                  )}
                >
                  <div className="truncate font-medium">
                    {s.repoLabel || "Untitled"}
                  </div>
                  <div
                    className={`text-[10px] mt-0.5 ${isDark ? "text-white/25" : "text-black/25"}`}
                  >
                    {s.phase || ""}
                  </div>
                </button>

                <button
                  onClick={(e) => handleDelete(e, s.localId)}
                  className={`absolute right-1.5 top-1/2 -translate-y-1/2 p-1 rounded-lg opacity-0 group-hover:opacity-100 transition-all ${
                    isDark
                      ? "text-white/30 hover:text-red-400 hover:bg-red-500/10"
                      : "text-black/30 hover:text-red-500 hover:bg-red-500/10"
                  }`}
                  aria-label="Delete session"
                >
                  <span className="text-xs">×</span>
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Bottom: avatar + settings */}
        <div className="mt-auto shrink-0 p-3">
          <Popover.Root>
            <Popover.Trigger
              className={cn(
                "w-full flex items-center gap-2 p-2 rounded-xl transition-colors",
                isDark
                  ? "text-white/40 hover:bg-white/5 hover:text-white"
                  : "text-black/40 hover:bg-black/5 hover:text-black",
                expanded ? "justify-start" : "justify-center",
              )}
              aria-label="User menu"
            >
              <div
                className={`h-8 w-8 rounded-full flex items-center justify-center text-xs font-semibold shrink-0 transition-colors ${
                  isDark ? "bg-[#2541B2] text-white" : "bg-[#1098F7] text-white"
                }`}
              >
                {initials}
              </div>
              {expanded && (
                <span className="text-xs font-medium truncate">
                  {user?.name || "User"}
                </span>
              )}
            </Popover.Trigger>

            <Popover.Portal>
              <Popover.Positioner>
                <Popover.Popup
                  className={`z-[60] w-52 rounded-2xl border p-2 shadow-2xl ${
                    isDark
                      ? "border-white/10 bg-[#1a1a1a]"
                      : "border-black/10 bg-white"
                  }`}
                >
                  <div
                    className={`mb-1 truncate border-b px-2 py-2 text-xs ${
                      isDark
                        ? "border-white/10 text-white/40"
                        : "border-black/10 text-black/40"
                    }`}
                  >
                    {user?.email || "user@example.com"}
                  </div>

                  <button
                    onClick={() => setSettingsOpen(true)}
                    className={`w-full flex items-center gap-2 px-2 py-2 rounded-xl text-sm transition-colors ${
                      isDark
                        ? "hover:bg-white/5 text-white/70"
                        : "hover:bg-black/5 text-black/70"
                    }`}
                  >
                    <Settings className="h-4 w-4" />
                    Settings
                  </button>

                  <button
                    onClick={() => onLogout?.()}
                    className="w-full flex items-center gap-2 px-2 py-2 rounded-xl text-sm text-red-400 hover:bg-red-500/10 transition-colors"
                  >
                    <LogOut className="h-4 w-4" />
                    Log out
                  </button>
                </Popover.Popup>
              </Popover.Positioner>
            </Popover.Portal>
          </Popover.Root>
        </div>
      </aside>
    </>
  );
}
