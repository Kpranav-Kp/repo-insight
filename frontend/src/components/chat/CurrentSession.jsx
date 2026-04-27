import { Send, Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { api, poll } from "@/lib/api";
import { cn } from "@/lib/utils";

export function CurrentSession() {
  // stage: "idle" | "analyzing" | "ready" | "thinking"
  // Add this at the top of the component
  const user = localStorage.getItem("username") || "default";

  // Then change all localStorage keys like this:
  const [stage, setStage] = useState(
    () => localStorage.getItem(`${user}_stage`) || "idle",
  );
  const [repoLabel, setRepoLabel] = useState(
    () => localStorage.getItem(`${user}_repoLabel`) || "",
  );
  const [sessionId, setSessionId] = useState(
    () => localStorage.getItem(`${user}_sessionId`) || null,
  );
  const [phase, setPhase] = useState(
    () => localStorage.getItem(`${user}_phase`) || "onboarding",
  );
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem(`${user}_messages`);
    return saved
      ? JSON.parse(saved)
      : [
          {
            role: "ai",
            content:
              "Hey! I'm here to help you find a meaningful open source issue to work on — and guide you through it without handing you the solution.\n\nWhat's the GitHub repo URL you'd like to contribute to?",
          },
        ];
  });
  useEffect(() => {
    localStorage.setItem(`${user}_stage`, stage);
    localStorage.setItem(`${user}_repoLabel`, repoLabel);
    localStorage.setItem(`${user}_sessionId`, sessionId);
    localStorage.setItem(`${user}_phase`, phase);
    localStorage.setItem(`${user}_messages`, JSON.stringify(messages));
  }, [stage, repoLabel, sessionId, phase, messages]);
  {
    /*}  const [stage, setStage] = useState("idle");
  const [repoLabel, setRepoLabel] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [phase, setPhase] = useState("onboarding");
  const [messages, setMessages] = useState([
    {
      role: "ai",
      content:
        "Hey! I'm here to help you find a meaningful open source issue to work on — and guide you through it without handing you the solution.\n\nWhat's the GitHub repo URL you'd like to contribute to?",
    },
  ]);*/
  }
  const [input, setInput] = useState("");
  const [error, setError] = useState(null);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const repoNameFromUrl = (url) =>
    url.replace(/\.git$/, "").replace(/^https?:\/\/github\.com\//, "");

  const startRepo = async (url) => {
    setError(null);
    setStage("analyzing");
    setRepoLabel(repoNameFromUrl(url));
    setMessages((m) => [
      ...m,
      { role: "user", content: url },
      {
        role: "ai",
        content: `Fetching issues and building the skill map for ${repoNameFromUrl(url)}...`,
        pending: true,
      },
    ]);

    try {
      const analyze = await api.analyzeRepository(url);
      const repoId = analyze.repository_id ?? analyze.id;
      if (!repoId) throw new Error("Backend did not return a repository id.");

      const repo = await poll(
        () => api.repositoryStatus(repoId),
        (r) => r.status === "completed" || r.status === "failed",
        { intervalMs: 2000, timeoutMs: 5 * 60000 },
      );
      if (repo.status === "failed") {
        throw new Error(repo.error_message || "Repository analysis failed.");
      }

      const session = await api.createSession(repoId);
      setSessionId(session.id);
      setPhase(session.phase || "onboarding");

      setMessages((m) => {
        const copy = [...m];
        if (copy[copy.length - 1]?.pending) copy.pop();
        copy.push({
          role: "ai",
          content: `All set! I've indexed ${repoNameFromUrl(url)}. Tell me a bit about your skills or what you'd like to work on.`,
        });
        return copy;
      });
      setStage("ready");
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Something went wrong.";
      setError(msg);
      setMessages((m) => {
        const copy = [...m];
        if (copy[copy.length - 1]?.pending) copy.pop();
        copy.push({ role: "ai", content: `⚠️ ${msg}` });
        return copy;
      });
      setStage("idle");
    }
  };

  const sendChat = async (text) => {
    if (!sessionId) return;
    setError(null);
    setMessages((m) => [
      ...m,
      { role: "user", content: text },
      { role: "ai", content: "", pending: true },
    ]);
    setStage("thinking");

    try {
      const accepted = await api.sendMessage(sessionId, text);
      const result = await poll(
        () => api.chatResult(accepted.task_id),
        (r) => r.status === "done",
        { intervalMs: 1500, timeoutMs: 2 * 60000 },
      );

      if (result.status === "done") {
        setPhase(result.phase || phase);
        setMessages((m) => {
          const copy = [...m];
          if (copy[copy.length - 1]?.pending) copy.pop();
          copy.push({ role: "ai", content: result.message });
          return copy;
        });
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Something went wrong.";
      setError(msg);
      setMessages((m) => {
        const copy = [...m];
        if (copy[copy.length - 1]?.pending) copy.pop();
        copy.push({ role: "ai", content: `⚠️ ${msg}` });
        return copy;
      });
    } finally {
      setStage("ready");
    }
  };

  const handleSend = () => {
    const text = input.trim();
    if (!text) return;
    setInput("");
    if (stage === "idle") startRepo(text);
    else if (stage === "ready") sendChat(text);
  };

  const busy = stage === "analyzing" || stage === "thinking";
  const placeholder =
    stage === "idle"
      ? "Paste a GitHub repo URL..."
      : busy
        ? "Working..."
        : "Type a message...";

  return (
    <div className="flex h-full flex-col">
      {/* Repo header */}
      <div className="flex items-center justify-between border-b border-border/50 px-6 py-4">
        <div className="flex items-center gap-3">
          <span className="font-semibold text-foreground">
            {repoLabel || "No repo selected"}
          </span>
          {phase && repoLabel && (
            <span className="rounded-md bg-violet-600/15 px-2 py-0.5 text-xs font-medium text-violet-300 dark:text-violet-300">
              {phase}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="relative flex h-2 w-2">
            <span
              className={cn(
                "absolute inline-flex h-full w-full rounded-full opacity-75",
                busy ? "animate-ping bg-amber-400" : "bg-emerald-400",
              )}
            />
            <span
              className={cn(
                "relative inline-flex h-2 w-2 rounded-full",
                busy ? "bg-amber-500" : "bg-emerald-500",
              )}
            />
          </span>
          {busy ? (stage === "analyzing" ? "analyzing" : "thinking") : "live"}
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 space-y-5 overflow-y-auto px-6 py-6"
      >
        {messages.map((msg, i) => (
          <MessageBubble key={i} msg={msg} />
        ))}
      </div>

      {/* Input */}
      <div className="border-t border-border/50 p-4">
        {error && <p className="mb-2 text-xs text-red-400">{error}</p>}
        <div className="flex items-end gap-2 rounded-xl border border-border bg-card/60 p-2 backdrop-blur-sm transition-all focus-within:border-violet-500/60">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={busy}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder={placeholder}
            rows={1}
            className="max-h-32 flex-1 resize-none bg-transparent px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none disabled:opacity-60"
          />
          <button
            onClick={handleSend}
            disabled={busy || !input.trim()}
            aria-label="Send"
            className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-600 via-violet-600 to-purple-600 text-white shadow-md shadow-violet-600/30 transition-transform hover:scale-105 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 dark:from-indigo-400 dark:via-violet-400 dark:to-purple-400"
          >
            {busy ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={cn("flex flex-col", isUser ? "items-end" : "items-start")}>
      <span className="mb-1 px-1 text-xs font-medium text-muted-foreground">
        {isUser ? "You" : "RepoInsight"}
      </span>
      <div
        className={cn(
          "max-w-[78%] whitespace-pre-line rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "rounded-tr-sm bg-gradient-to-br from-indigo-600 via-violet-600 to-purple-600 text-white shadow-md shadow-violet-600/20 dark:from-indigo-400 dark:via-violet-400 dark:to-purple-400"
            : "rounded-tl-sm bg-card/70 text-foreground ring-1 ring-border/60 backdrop-blur-sm",
        )}
      >
        {msg.content}
        {msg.pending && (
          <span className="ml-1 inline-flex gap-1 align-middle">
            <Dot delay="0ms" />
            <Dot delay="150ms" />
            <Dot delay="300ms" />
          </span>
        )}
      </div>
    </div>
  );
}

function Dot({ delay }) {
  return (
    <span
      className="inline-block h-1.5 w-1.5 animate-bounce rounded-full bg-violet-400"
      style={{ animationDelay: delay }}
    />
  );
}
