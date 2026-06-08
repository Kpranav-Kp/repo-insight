// frontend/src/components/chat/ChatSidebar.jsx
import { Popover } from "@base-ui/react/popover";
import { ChevronLeft, ChevronRight, LogOut, Settings } from "lucide-react";
import { useEffect, useState } from "react";

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
  const [expanded, setExpanded] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Refresh session list
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
      {/* Settings modal (outside sidebar, rendered at document root) */}
      <SettingsModal
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        user={user}
        onUsernameChange={onUsernameChange}
      />

      <aside
        className={cn(
          "flex h-full flex-col border-r border-border bg-card dark:bg-card transition-all duration-300",
          expanded ? "w-64" : "w-14",
        )}
      >
        {/* Top: expand/collapse + New Chat button */}
        <div className="flex shrink-0 items-center gap-2 border-b border-border p-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="p-2 rounded-lg hover:bg-violet-600/15 text-muted-foreground transition-colors"
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
              className="flex-1 text-xs font-medium bg-primary text-white py-1.5 rounded-lg transition-opacity hover:opacity-90"
            >
              + New Chat
            </button>
          )}
        </div>

        {/* Session list (visible when expanded) */}
        {expanded && (
          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {sessions.length === 0 && (
              <p className="text-xs text-muted-foreground text-center py-8">
                No saved sessions
              </p>
            )}

            {sessions.map((s) => (
              <div key={s.localId} className="group relative">
                <button
                  onClick={() => onSelectSession?.(s)}
                  className={cn(
                    "w-full text-left truncate text-xs p-2 rounded-lg transition-colors pr-8",
                    activeSessionId === s.localId
                      ? "bg-violet-600/20 text-foreground dark:text-foreground"
                      : "text-muted-foreground hover:bg-violet-600/15 hover:text-foreground",
                  )}
                >
                  <div className="truncate">{s.repoLabel || "Untitled"}</div>
                  <div className="text-[10px] text-muted-foreground/60 mt-0.5">
                    {s.phase || ""}
                  </div>
                </button>

                <button
                  onClick={(e) => handleDelete(e, s.localId)}
                  className="absolute right-1 top-1/2 -translate-y-1/2 p-1 rounded text-muted-foreground opacity-0 group-hover:opacity-100 hover:text-red-400 hover:bg-red-500/10 transition-all"
                  aria-label="Delete session"
                >
                  <span className="text-xs">×</span>
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Bottom: avatar + settings popover — always pinned */}
        <div className="mt-auto shrink-0 border-t border-border p-2">
          <Popover.Root>
            <Popover.Trigger
              className={cn(
                "w-full flex items-center gap-2 p-2 rounded-lg transition-colors",
                "text-muted-foreground hover:bg-violet-600/15 hover:text-foreground dark:hover:text-foreground",
                expanded ? "justify-start" : "justify-center",
              )}
              aria-label="User menu"
            >
              <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center text-white text-xs font-semibold shrink-0">
                {initials}
              </div>
              {expanded && (
                <span className="text-xs font-medium truncate text-foreground dark:text-foreground">
                  {user?.name || "User"}
                </span>
              )}
            </Popover.Trigger>

            <Popover.Portal>
              <Popover.Positioner>
                <Popover.Popup className="z-50 w-52 rounded-xl border border-border bg-card p-2 shadow-xl">
                  <div className="mb-1 truncate border-b border-border px-2 py-1.5 text-xs text-muted-foreground">
                    {user?.email || "user@example.com"}
                  </div>

                  <button
                    onClick={() => {
                      setSettingsOpen(true);
                    }}
                    className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm hover:bg-violet-600/15 transition-colors"
                  >
                    <Settings className="h-4 w-4" />
                    Settings
                  </button>

                  <button
                    onClick={() => onLogout?.()}
                    className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm text-red-400 hover:bg-red-500/10 transition-colors"
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
