import { useState } from "react";

import ChatPage from "./components/ChatPage";
import LandingPage from "./components/LandingPage";
import Login from "./components/Login";

export default function App() {
  const [view, setView] = useState("landing"); // "landing" | "login" | "chat"

  // read user from localStorage (saved at login)
  const raw = localStorage.getItem("user");
  const user = raw ? JSON.parse(raw) : null;
  

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("user");
    setView("login");
  };

  if (view === "chat") {
    return <ChatPage user={user} onLogout={handleLogout} />;
  }

  if (view === "login") {
    return <Login onLoginSuccess={() => setView("chat")} />;
  }

  return (
    <LandingPage
      onGetStarted={() => setView("login")}
      onLoginSuccess={() => setView("chat")}
    />
  );
}
