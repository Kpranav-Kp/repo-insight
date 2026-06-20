import { PanelRight } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

import { ChatSidebar } from "@/components/chat/ChatSidebar";
import { CurrentSession } from "@/components/chat/CurrentSession";
import { useTheme } from "@/components/ThemeToggle";

export default function ChatPage({ onLogout }) {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const [activeSession, setActiveSession] = useState(null);
  const [chatKey, setChatKey] = useState(0);
  const [username, setUsername] = useState(
    localStorage.getItem("username") || "",
  );
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const user = {
    email: localStorage.getItem("email"),
    name: username,
  };

  const handleNewChat = () => {
    setActiveSession(null);
    setChatKey((k) => k + 1);
    setSidebarOpen(false);
  };

  const handleSelectSession = (session) => {
    setActiveSession(session);
    setChatKey((k) => k + 1);
    setSidebarOpen(false);
  };

  const handleLogout = () => {
    localStorage.removeItem("username");
    localStorage.removeItem("email");
    onLogout?.();
  };

  return (
    <div
      className={`relative h-screen overflow-hidden transition-colors duration-300 ${
        isDark ? "bg-[#0A0A0A] text-white" : "bg-[#FAFAFA] text-[#1a1a1a]"
      }`}
    >
      <div className="relative flex h-full p-3 gap-3">
        {/* Sidebar — hidden on mobile, inline on desktop */}
        <div
          className={`hidden md:block shrink-0 rounded-2xl overflow-hidden transition-all duration-300 ${
            isDark
              ? "bg-[#171717] border border-white/6"
              : "bg-white border border-black/6 shadow-sm"
          }`}
        >
          <ChatSidebar
            user={user}
            onLogout={handleLogout}
            onSelectSession={handleSelectSession}
            activeSessionId={activeSession?.localId}
            onNewChat={handleNewChat}
            onUsernameChange={(newName) => {
              setUsername(newName);
            }}
          />
        </div>

        {/* Mobile sidebar overlay */}
        {sidebarOpen && (
          <div className="fixed inset-0 z-50 md:hidden">
            <div
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
              onClick={() => setSidebarOpen(false)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === "Escape" && setSidebarOpen(false)}
            />
            <div className="absolute left-0 top-0 bottom-0 w-72 max-w-[80vw]">
              <div
                className={`h-full rounded-r-2xl overflow-hidden ${
                  isDark
                    ? "bg-[#171717] border-r border-white/6"
                    : "bg-white border-r border-black/6 shadow-sm"
                }`}
              >
                <ChatSidebar
                  user={user}
                  onLogout={handleLogout}
                  onSelectSession={handleSelectSession}
                  activeSessionId={activeSession?.localId}
                  onNewChat={handleNewChat}
                  onUsernameChange={(newName) => {
                    setUsername(newName);
                  }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Main chat area with rounded corners */}
        <div
          className={`flex-1 min-w-0 rounded-2xl overflow-hidden transition-colors duration-300 ${
            isDark
              ? "bg-[#171717] border border-white/6"
              : "bg-white border border-black/6 shadow-sm"
          }`}
        >
          {/* Mobile header with sidebar toggle */}
          <div className="md:hidden flex items-center gap-2 px-4 py-2 border-b border-white/10">
            <button
              onClick={() => setSidebarOpen(true)}
              className={`p-2 rounded-xl transition-colors ${
                isDark
                  ? "text-white/40 hover:bg-white/5 hover:text-white"
                  : "text-black/40 hover:bg-black/5 hover:text-black"
              }`}
              aria-label="Open sidebar"
            >
              <PanelRight className="h-4 w-4" />
            </button>
            <Link
              to="/"
              className="text-sm font-orangevoyage tracking-widest text-white/50 hover:text-white/80 transition-colors"
            >
              RepoInsight
            </Link>
          </div>

          <CurrentSession key={chatKey} activeSession={activeSession} />
        </div>
      </div>
    </div>
  );
}
