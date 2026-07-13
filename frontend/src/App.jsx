import { useState, useEffect } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  useNavigate,
} from "react-router-dom";

import { ThemeProvider } from "@/components/ThemeToggle";
import { api } from "@/lib/api";
import { backendLogout, checkSession } from "@/lib/auth";

import AuthCallback from "./components/AuthCallback";
import ChatPage from "./components/ChatPage";
import LandingPage from "./components/LandingPage";
import Login from "./components/Login";
import Signup from "./components/Signup";
import VerifyEmail from "./components/VerifyEmail";

function ProtectedRoute({ children, sessionChecked }) {
  const [isAuthenticated, setIsAuthenticated] = useState(
    !!localStorage.getItem("username"),
  );

  useEffect(() => {
    const checkAuth = () => {
      setIsAuthenticated(!!localStorage.getItem("username"));
    };
    window.addEventListener("storage", checkAuth);
    return () => window.removeEventListener("storage", checkAuth);
  }, []);

  if (!sessionChecked) return null;
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  return children;
}

function AppRoutes() {
  const navigate = useNavigate();
  const [sessionChecked, setSessionChecked] = useState(false);

  useEffect(() => {
    checkSession().then((data) => {
      if (data && data.username) {
        localStorage.setItem("username", data.username);
        if (data.email) localStorage.setItem("email", data.email);
      } else {
        localStorage.removeItem("username");
        localStorage.removeItem("email");
      }
      setSessionChecked(true);
    });
  }, []);

  const handleLogout = async () => {
    await api.flushFeedback().catch(() => {});
    await backendLogout();
    localStorage.removeItem("username");
    localStorage.removeItem("email");
    navigate("/");
  };

  if (!sessionChecked) return null;

  return (
    <Routes>
      <Route
        path="/"
        element={
          localStorage.getItem("username") ? (
            <Navigate to="/chat" replace />
          ) : (
            <LandingPage />
          )
        }
      />
      <Route
        path="/login"
        element={
          localStorage.getItem("username") ? (
            <Navigate to="/chat" replace />
          ) : (
            <Login />
          )
        }
      />
      <Route path="/signup" element={<Signup />} />
      <Route path="/auth/callback" element={<AuthCallback />} />
      <Route path="/verify-email" element={<VerifyEmail />} />
      <Route
        path="/chat"
        element={
          <ProtectedRoute sessionChecked={sessionChecked}>
            <ChatPage onLogout={handleLogout} />
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <Router>
        <AppRoutes />
      </Router>
    </ThemeProvider>
  );
}
