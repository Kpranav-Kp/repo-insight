
import { useState } from "react";
import LandingPage from "./components/LandingPage";
import Login from "./components/Login";
import ChatPage from "./components/ChatPage";

export default function App() {
  const [view, setView] = useState("landing"); // "landing" | "login" | "chat"

  if (view === "chat") return <ChatPage />;
  if (view === "login") return <Login onLoginSuccess={() => setView("chat")} />;
  return <LandingPage onGetStarted={() => setView("login")} onLoginSuccess={() => setView("chat")} />;
}
