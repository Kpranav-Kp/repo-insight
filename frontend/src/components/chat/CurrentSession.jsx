// frontend/src/components/chat/CurrentSession.jsx
import { Send, Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

import { api, poll } from "@/lib/api";
import { upsertSession, newLocalId } from "@/lib/sessionStore";
import { cn } from "@/lib/utils";

const WELCOME = {
  role: "ai",
  content:
    "Hey! I'm here to help you find a meaningful open source issue to work on — and guide you through it without handing you the solution.\n\nWhat's the GitHub repo URL you'd like to contribute to?",
};

export function CurrentSession({ activeSession }) {
  const [stage, setStage] = useState(
    activeSession?.sessionId ? "ready" : "idle",
  );
  const [repoUrl, setRepoUrl] = useState(activeSession?.repoUrl || "");
  const [repoLabel, setRepoLabel] = useState(activeSession?.repoLabel || "");
  const [sessionId, setSessionId] = useState(activeSession?.sessionId || null);
  const [phase, setPhase] = useState(activeSession?.phase || "onboarding");
  const [messages, setMessages] = useState(
    activeSession?.messages?.length ? activeSession.messages : [WELCOME],
  );
  const [input, setInput] = useState("");
  const [error, setError] = useState(null);
  const localIdRef = useRef(activeSession?.localId ?? newLocalId());
  const scrollRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  useEffect(() => {
    if (!repoLabel && messages.length <= 1) return;
    upsertSession({
      localId: localIdRef.current,
      repoUrl,
      repoLabel,
      sessionId,
      phase,
      messages,
      updatedAt: Date.now(),
    });
  }, [repoUrl, repoLabel, sessionId, phase, messages]);

  const repoNameFromUrl = (url) =>
    url.replace(/\.git$/, "").replace(/^https?:\/\/github\.com\//, "");

  const startRepo = async (url) => {
    setError(null);
    setStage("analyzing");
    setRepoUrl(url);
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
        { intervalMs: 2000, timeoutMs: 5 * 60_000 },
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
        { intervalMs: 1500, timeoutMs: 2 * 60_000 },
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

  const adjustTextareaHeight = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const newHeight = Math.min(el.scrollHeight, 160); // max 160px
    el.style.height = `${newHeight}px`;
  };

  const handleInputChange = (e) => {
    setInput(e.target.value);
    adjustTextareaHeight();
  };

  const handleSend = () => {
    const text = input.trim();
    if (!text) return;
    setInput("");
    // reset height
    if (textareaRef.current) {
      textareaRef.current.style.height = "40px";
    }
    if (stage === "idle") startRepo(text);
    else if (stage === "ready") sendChat(text);
  };

  const busy = stage === "analyzing" || stage === "thinking";
  const isInitial = stage === "idle" && messages.length <= 1;
  let placeholderText = "Type a message...";
  if (isInitial) placeholderText = "Paste a GitHub repo URL...";
  else if (busy) placeholderText = "Working...";

  let statusText = "live";
  if (busy) {
    statusText = stage === "analyzing" ? "analyzing" : "thinking";
  }
  const InputArea = (
    <div className="border-t border-border/50 bg-background/80 backdrop-blur-sm p-4">
      {error && <p className="mb-2 text-xs text-red-400">{error}</p>}
      <div className="flex items-end gap-2 rounded-xl border border-border bg-card/60 backdrop-blur-sm transition-colors focus-within:border-violet-500/60 p-2">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleInputChange}
          disabled={busy}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          placeholder={placeholderText}
          rows={1}
          className="flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-foreground placeholder:text-muted-foreground outline-none disabled:opacity-60"
          style={{ height: 40, maxHeight: 160 }}
        />
        <button
          onClick={handleSend}
          disabled={busy || !input.trim()}
          aria-label="Send"
          className="flex h-8 w-8 items-center justify-center rounded-lg bg-linear-to-br from-indigo-600 via-violet-600 to-purple-600 text-white shadow-md shadow-violet-600/30 transition-transform hover:scale-105 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {busy ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </button>
      </div>
    </div>
  );

  if (isInitial) {
    return (
      <div className="h-full flex flex-col">
        <div className="flex-1 flex items-center justify-center px-6">
          <div className="w-full max-w-2xl">
            <h2 className="text-2xl font-display font-bold text-center mb-8">
              Let&rsquo;s find your next open-source contribution
            </h2>
            {InputArea}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-border/50">
        <div className="flex items-center gap-3">
          <span className="font-semibold text-foreground">
            {repoLabel || "No repo selected"}
          </span>
          {phase && repoLabel && (
            <span className="rounded-md bg-violet-600/15 px-2 py-0.5 text-xs font-medium text-violet-300">
              {phase}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span
            className={cn(
              "w-2 h-2 rounded-full",
              busy ? "bg-amber-400 animate-pulse" : "bg-emerald-400",
            )}
          />
          {statusText}
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-6 py-6 space-y-5"
      >
        {messages.map((msg, i) => (
          <MessageBubble key={i} msg={msg} />
        ))}
      </div>

      {/* Input (bottom) */}
      {InputArea}
    </div>
  );
}

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[78%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-linear-to-br from-indigo-600 via-violet-600 to-purple-600 text-white shadow-md shadow-violet-600/20"
            : "bg-card/70 text-foreground ring-1 ring-border/60 backdrop-blur-sm",
        )}
      >
        {isUser ? (
          <span>{msg.content}</span>
        ) : (
          <div className="prose prose-sm dark:prose-invert max-w-none break-words">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
            >
              {msg.content}
            </ReactMarkdown>
          </div>
        )}
        {msg.pending && (
          <span className="ml-1 inline-flex gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce" />
            <span
              className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce"
              style={{ animationDelay: "150ms" }}
            />
            <span
              className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce"
              style={{ animationDelay: "300ms" }}
            />
          </span>
        )}
      </div>
    </div>
  );
}
