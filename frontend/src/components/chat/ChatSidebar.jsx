import { Compass, MessageSquare, History, Bookmark, LogOut } from "lucide-react";
import { cn } from "@/lib/utils";

const items = [
  { key: "explore", icon: Compass, label: "Explore" },
  { key: "chat", icon: MessageSquare, label: "Chat" },
  { key: "history", icon: History, label: "History" },
  { key: "saved", icon: Bookmark, label: "Saved" },
];

export function ChatSidebar({ active, onChange }) {
  return (
    <aside className="flex w-14 flex-col items-center justify-between border-r border-border/50 bg-card/40 py-4 backdrop-blur-sm">
      <div className="flex flex-col items-center gap-2">
        {items.map(({ key, icon: Icon, label }) => {
          const isActive = active === key;
          return (
            <button
              key={key}
              onClick={() => onChange(key)}
              aria-label={label}
              className={cn(
                "group relative flex h-10 w-10 items-center justify-center rounded-xl transition-all",
                isActive
                  ? "bg-gradient-to-br from-indigo-600 via-violet-600 to-purple-600 text-white shadow-lg shadow-violet-600/30 dark:from-indigo-400 dark:via-violet-400 dark:to-purple-400"
                  : "text-muted-foreground hover:bg-violet-600/15 hover:text-foreground"
              )}
            >
              <Icon className="h-5 w-5" />
            </button>
          );
        })}
      </div>
      <div className="flex flex-col items-center gap-3">
        <button
          aria-label="Log out"
          className="flex h-10 w-10 items-center justify-center rounded-xl text-muted-foreground transition-colors hover:bg-violet-600/15 hover:text-foreground"
        >
          <LogOut className="h-5 w-5" />
        </button>
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br from-indigo-600 via-violet-600 to-purple-600 text-xs font-semibold text-white dark:from-indigo-400 dark:via-violet-400 dark:to-purple-400">
          AK
        </div>
      </div>
    </aside>
  );
}
