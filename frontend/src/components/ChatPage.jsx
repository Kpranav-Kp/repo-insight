import { useState } from "react";

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

  const user = {
    email: localStorage.getItem("email"),
    name: username,
  };

  const handleNewChat = () => {
    setActiveSession(null);
    setChatKey((k) => k + 1);
  };

  const handleSelectSession = (session) => {
    setActiveSession(session);
    setChatKey((k) => k + 1);
  };

  const handleLogout = async () => {
    await fetch("/api/auth/logout/", {
      method: "POST",
      credentials: "include",
    });
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
        {/* Sidebar with rounded corners and subtle separation */}
        <div
          className={`shrink-0 rounded-2xl overflow-hidden transition-all duration-300 ${
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

        {/* Main chat area with rounded corners */}
        <div
          className={`flex-1 min-w-0 rounded-2xl overflow-hidden transition-colors duration-300 ${
            isDark
              ? "bg-[#171717] border border-white/6"
              : "bg-white border border-black/6 shadow-sm"
          }`}
        >
          <CurrentSession key={chatKey} activeSession={activeSession} />
        </div>
      </div>
    </div>
  );
}
