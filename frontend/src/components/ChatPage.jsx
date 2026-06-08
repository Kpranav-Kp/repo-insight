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
    <div className="relative h-screen bg-white dark:bg-background text-foreground overflow-hidden">
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

        <main className="flex-1 min-w-0 bg-white dark:bg-background">
          <CurrentSession key={chatKey} activeSession={activeSession} />
        </main>
      </div>
    </div>
  );
}
