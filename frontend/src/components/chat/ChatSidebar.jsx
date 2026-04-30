import { MessageSquare, History, LogOut, User } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { cn } from "@/lib/utils";

const items = [
  { key: "chat", icon: MessageSquare, label: "Chat" },
  { key: "history", icon: History, label: "History" },
];

export function ChatSidebar({ active, onChange, onLogout, user }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);

  // close profile menu when clicking outside
  useEffect(() => {
    function handleClick(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const initials = (user?.name || user?.email || "U")
    .split(" ")
    .map((s) => s[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();

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
              title={label}
              className={cn(
                "group relative flex h-10 w-10 items-center justify-center rounded-xl transition-all",
                isActive
                  ? "bg-gradient-to-br from-indigo-600 via-violet-600 to-purple-600 text-white shadow-lg shadow-violet-600/30"
                  : "text-muted-foreground hover:bg-violet-600/15 hover:text-foreground",
              )}
            >
              <Icon className="h-5 w-5" />
            </button>
          );
        })}
      </div>

      <div className="flex flex-col items-center gap-3">
        {/* Logout */}
        <button
          onClick={onLogout}
          aria-label="Log out"
          title="Log out"
          className="flex h-10 w-10 items-center justify-center rounded-xl text-muted-foreground transition-colors hover:bg-red-600/15 hover:text-red-400"
        >
          <LogOut className="h-5 w-5" />
        </button>

        {/* Profile avatar + popover menu */}
        <div className="relative" ref={menuRef}>
          <button
            onClick={() => setMenuOpen((v) => !v)}
            aria-label="Profile"
            title={user?.email || "Profile"}
            className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br from-indigo-600 via-violet-600 to-purple-600 text-xs font-semibold text-white"
          >
            {initials}
          </button>

          {menuOpen && (
            <div className="absolute bottom-0 left-12 z-50 w-56 rounded-xl border border-border/60 bg-card/95 p-3 shadow-2xl backdrop-blur-xl">
              <div className="flex items-center gap-2 border-b border-border/50 pb-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-indigo-600 via-violet-600 to-purple-600 text-xs font-semibold text-white">
                  {initials}
                </div>
                <div className="min-w-0">
                  <div className="truncate text-sm font-semibold text-foreground">
                    {user?.name || "Signed in"}
                  </div>
                  <div className="truncate text-xs text-muted-foreground">
                    {user?.email || "user@example.com"}
                  </div>
                </div>
              </div>
              <button
                onClick={() => {
                  setMenuOpen(false);
                  // hook up later: open profile page / modal
                  alert("Profile page coming soon");
                }}
                className="mt-2 flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-muted-foreground hover:bg-violet-600/15 hover:text-foreground"
              >
                <User className="h-4 w-4" />
                View profile
              </button>
              <button
                onClick={() => {
                  setMenuOpen(false);
                  onLogout?.();
                }}
                className="mt-1 flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-red-400 hover:bg-red-600/15"
              >
                <LogOut className="h-4 w-4" />
                Log out
              </button>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
