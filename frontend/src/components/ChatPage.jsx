// frontend/src/components/ChatPage.jsx
import { useState } from "react";

import { ChatSidebar } from "@/components/chat/ChatSidebar";
import { CurrentSession } from "@/components/chat/CurrentSession";

export default function ChatPage({ onLogout }) {
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

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("username");
    localStorage.removeItem("email");
    onLogout?.();
  };

  return (
    <div className="relative h-screen bg-background text-foreground overflow-hidden">
      {/* Background blobs */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 overflow-hidden"
      >
        <div className="absolute -left-24 top-0 h-105 w-105 rounded-full bg-violet-600/15 blur-3xl" />
        <div className="absolute -right-24 bottom-0 h-105 w-105 rounded-full bg-indigo-600/15 blur-3xl" />
      </div>

      <div className="relative flex h-full">
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

        <main className="flex-1 min-w-0">
          <CurrentSession key={chatKey} activeSession={activeSession} />
        </main>
      </div>
    </div>
  );
}
