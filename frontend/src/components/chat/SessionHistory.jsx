// src/components/chat/SessionHistory.jsx
import { Search, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";

import { loadSessions, deleteSession } from "@/lib/sessionStore";
import { cn } from "@/lib/utils";

export function SessionHistory({ onResume, activeLocalId }) {
  const [query, setQuery] = useState("");
  const [sessions, setSessions] = useState([]);

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

  const filtered = sessions.filter((s) => {
    const q = query.toLowerCase();
    return (
      (s.repoLabel || "").toLowerCase().includes(q) ||
      (s.messages?.[1]?.content || "").toLowerCase().includes(q)
    );
  });

  const handleDelete = (e, id) => {
    e.stopPropagation();
    deleteSession(id);
  };

  return (
    <div className="flex h-full flex-col">
      <div className="px-6 pb-4 pt-5">
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search your sessions..."
            className="w-full rounded-xl border border-border bg-card/60 py-2.5 pl-10 pr-4 text-sm text-foreground placeholder:text-muted-foreground backdrop-blur-sm focus:border-violet-500/60 focus:outline-none focus:ring-2 focus:ring-violet-500/20"
          />
        </div>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto px-6 pb-6">
        {filtered.length === 0 && (
          <p className="py-12 text-center text-sm text-muted-foreground">
            No saved sessions yet. Start a new chat to begin.
          </p>
        )}
        {filtered.map((s) => (
          <div
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onResume?.(s);
              }
            }}
            key={s.localId}
            onClick={() => onResume?.(s)}
            className={cn(
              "group relative cursor-pointer overflow-hidden rounded-xl border bg-card/60 p-4 backdrop-blur-sm transition-all hover:-translate-y-0.5 hover:shadow-lg hover:shadow-violet-600/10",
              activeLocalId === s.localId
                ? "border-violet-500/60 ring-1 ring-violet-500/40"
                : "border-border/60",
            )}
          >
            <header className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold text-foreground">
                  {s.repoLabel || "New chat"}
                </p>
                <p className="mt-0.5 truncate text-xs text-muted-foreground">
                  {s.messages?.length || 0} messages · {s.phase || "—"}
                </p>
              </div>
              <button
                onClick={(e) => handleDelete(e, s.localId)}
                className="rounded-md p-1 text-muted-foreground opacity-0 transition hover:bg-red-500/15 hover:text-red-400 group-hover:opacity-100"
                aria-label="Delete session"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </header>
            <p className="mt-2 line-clamp-2 text-xs text-muted-foreground">
              {s.messages?.[s.messages.length - 1]?.content || "—"}
            </p>
            <footer className="mt-3 text-[10px] uppercase tracking-wider text-muted-foreground">
              {new Date(s.updatedAt).toLocaleString()}
            </footer>
          </div>
        ))}
      </div>
    </div>
  );
}
