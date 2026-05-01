import { MessageSquare, History, LogOut } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

const items = [
  { key: "chat", icon: MessageSquare, label: "Chat" },
  { key: "history", icon: History, label: "History" },
];

export function ChatSidebar({ active, onChange, onLogout, user }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);
  const avatarRef = useRef(null);
  const [menuPos, setMenuPos] = useState({ top: 0, left: 0 });

  useEffect(() => {
    function handleClick(e) {
      if (
        menuRef.current &&
        !menuRef.current.contains(e.target) &&
        avatarRef.current &&
        !avatarRef.current.contains(e.target)
      ) {
        setMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const openMenu = () => {
    if (avatarRef.current) {
      const rect = avatarRef.current.getBoundingClientRect();

      setMenuPos({
        top: rect.top - 180,   // 👈 above avatar (adjust if needed)
        left: rect.left - 400, // 👈 shift into green area
      });
    }

    setMenuOpen((v) => !v);
  };

  const initials = (user?.name || user?.email || "U")
    .split(" ")
    .map((s) => s[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();

  return (
    <aside className="flex w-14 flex-col items-center justify-between border-r border-border/50 bg-card/40 py-4 backdrop-blur-sm">
      
      {/* Top icons */}
      <div className="flex flex-col items-center gap-2">
        {items.map(({ key, icon: Icon, label }) => {
          const isActive = active === key;
          return (
            <button
              key={key}
              onClick={() => onChange(key)}
              title={label}
              className={cn(
                "flex h-10 w-10 items-center justify-center rounded-xl transition-all",
                isActive
                  ? "bg-gradient-to-br from-indigo-600 via-violet-600 to-purple-600 text-white shadow-lg"
                  : "text-muted-foreground hover:bg-violet-600/15 hover:text-foreground"
              )}
            >
              <Icon className="h-5 w-5" />
            </button>
          );
        })}
      </div>

      {/* Avatar */}
      <div className="flex flex-col items-center gap-3">
        <div className="relative">
          <div
            ref={avatarRef}
            onClick={openMenu}
            title={user?.email || "Profile"}
            className="flex h-9 w-9 cursor-pointer items-center justify-center rounded-full bg-gradient-to-br from-indigo-600 via-violet-600 to-purple-600 text-xs font-semibold text-white"
          >
            {initials}
          </div>
        </div>
      </div>

      {/* Popup */}
      {menuOpen && (
        <div
          ref={menuRef}
          style={{ top: menuPos.top, left: menuPos.left }}
          className="fixed z-[99999] w-60 rounded-xl border border-violet-400/30 bg-gradient-to-br from-indigo-600 via-violet-600 to-purple-600 p-4 shadow-2xl text-white"
        >
          {/* User info */}
          <div className="flex items-center gap-2 border-b border-white/20 pb-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-white/10 text-xs font-semibold text-white">
              {initials}
            </div>

            <div className="min-w-0">
              <div className="text-sm font-semibold truncate">
                {user?.name || "User"}
              </div>
              <div className="text-xs text-white/70 truncate">
                {user?.email || "user@example.com"}
              </div>
            </div>
          </div>

          {/* Logout */}
          <button
            onClick={() => {
              setMenuOpen(false);
              onLogout?.();
            }}
            className="mt-2 flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-red-300 hover:bg-white/10"
          >
            <LogOut className="h-4 w-4" />
            Log out
          </button>
        </div>
      )}
    </aside>
  );
}