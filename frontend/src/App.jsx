import { useState } from "react";

import ChatPage from "./components/ChatPage";
import LandingPage from "./components/LandingPage";

export default function App() {
  const [view, setView] = useState("landing");

  const user = localStorage.getItem("username");

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("username");
    localStorage.removeItem("email");
    setView("landing");
  };

  if (view === "chat") {
    return <ChatPage user={user} onLogout={handleLogout} />;
  }

  return (
    <LandingPage
      onLoginSuccess={() => setView("chat")}
    />
  );
}