// frontend/src/components/chat/CurrentSession.jsx
import { Send, Loader2, Check, Copy } from "lucide-react";
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
  const [repoSkills, setRepoSkills] = useState([]);
  const [selectedSkills, setSelectedSkills] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
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
      setRepoSkills(repo.skills_found || []);
      setStage("skills"); // ✅ move to skill selection (not overwritten later)
      setPhase(session.phase || "onboarding");

      // Remove the pending analysis message and add a neutral "analysis complete" message
      setMessages((m) => {
        const copy = [...m];
        if (copy[copy.length - 1]?.pending) copy.pop();
        copy.push({
          role: "ai",
          content: `Repository analysis complete. Please select your skills below.`,
        });
        return copy;
      });
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

  // Submit structured skills and fetch recommendations
  const submitSkillsAndGetRecommendations = async () => {
    if (selectedSkills.length === 0) return;
    setStage("thinking");
    setError(null);
    try {
      // 1. Save skills to session
      await api.submitStructuredSkills(sessionId, selectedSkills);
      // 2. Fetch recommendations from backend (deterministic ranking)
      const recs = await api.getRecommendations(sessionId);

      if (recs.recommendations && recs.recommendations.length > 0) {
        setRecommendations(recs.recommendations);
        setMessages((prev) => [
          ...prev,
          {
            role: "ai",
            content: `Great! I found ${recs.recommendations.length} issues that match your skills. Select one to start working.`,
          },
        ]);
      } else if (recs.learning_path) {
        // No matching issues - show learning path
        setRecommendations([]);
        setMessages((prev) => [
          ...prev,
          { role: "ai", content: recs.learning_path },
        ]);
      }

      setStage("recommendations");
    } catch (err) {
      console.error("Error fetching recommendations:", err);
      setError("Failed to get recommendations. Please try again.");
      setStage("skills");
    }
  };

  // User selects an issue from the card
  const selectIssue = async (issue) => {
    setError(null);
    try {
      // Store selected issue in backend
      await api.selectIssue(sessionId, issue);
      setStage("thinking");

      // Add user message and get agent response
      const userMessage = `I'll work on issue #${issue.id}: ${issue.title}`;
      setMessages((prev) => [...prev, { role: "user", content: userMessage }]);

      // Send message to trigger guidance phase
      const accepted = await api.sendMessage(
        sessionId,
        "Let's start working on this issue.",
      );

      // Poll for agent response
      const result = await poll(
        () => api.chatResult(accepted.task_id),
        (r) => r.status === "done",
        { intervalMs: 1500, timeoutMs: 2 * 60_000 },
      );

      if (result.status === "done") {
        setPhase(result.phase || "guidance");
        setMessages((prev) => [
          ...prev,
          { role: "ai", content: result.message },
        ]);
      }

      setStage("ready");
    } catch (err) {
      const errMsg =
        err instanceof Error ? err.message : "Failed to select issue";
      setError(errMsg);
      setStage("recommendations");
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
    const newHeight = Math.min(el.scrollHeight, 160);
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
    if (textareaRef.current) textareaRef.current.style.height = "40px";
    if (stage === "idle") startRepo(text);
    else if (stage === "ready") sendChat(text);
    // Other stages (skills, recommendations) have their own buttons, not the main input.
  };
  const getDifficultyClass = (difficulty) => {
    if (difficulty === "beginner") {
      return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300";
    }
    if (difficulty === "advanced") {
      return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300";
    }
    return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300";
  };
  const busy = stage === "analyzing" || stage === "thinking";
  const isInitial = stage === "idle" && messages.length <= 1;
  let placeholderText = "Type a message...";
  if (isInitial) placeholderText = "Paste a GitHub repo URL...";
  else if (busy) placeholderText = "Working...";
  else if (stage === "skills") placeholderText = "Select skills above...";
  else if (stage === "recommendations")
    placeholderText = "Select an issue above...";
  else placeholderText = "Type a message...";

  let statusText = "live";
  if (busy) statusText = stage === "analyzing" ? "analyzing" : "thinking";
  else if (stage === "skills") statusText = "skills";
  else if (stage === "recommendations") statusText = "recommendations";

  const InputArea = (
    <div className="border-t border-gray-200 dark:border-gray-800 bg-background p-4">
      {error && <p className="mb-2 text-xs text-red-500">{error}</p>}
      <div className="flex items-end gap-2 rounded-xl border border-gray-200 dark:border-gray-800 bg-card transition-colors focus-within:border-primary p-2">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleInputChange}
          disabled={busy || (stage !== "ready" && stage !== "idle")}
          onKeyDown={(e) => {
            if (
              e.key === "Enter" &&
              !e.shiftKey &&
              (stage === "ready" || stage === "idle")
            ) {
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
          disabled={
            busy || (stage !== "ready" && stage !== "idle") || !input.trim()
          }
          aria-label="Send"
          className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-white shadow-md transition-transform hover:scale-105 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
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

  // Render skill selection card
  if (stage === "skills") {
    return (
      <div className="flex h-full flex-col">
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-2xl mx-auto bg-card rounded-xl border border-gray-200 dark:border-gray-800 p-6 space-y-6">
            <h3 className="text-xl font-display font-bold">Your Skills</h3>
            <p className="text-sm text-muted-foreground">
              Select the skills you have and your proficiency level.
            </p>
            <div className="space-y-4">
              {repoSkills.slice(0, 8).map((skill) => (
                <div
                  key={skill}
                  className="flex items-center justify-between gap-4"
                >
                  <span className="text-sm font-medium w-32">{skill}</span>
                  <div className="flex gap-2">
                    {["beginner", "intermediate", "advanced"].map((band) => (
                      <button
                        key={band}
                        onClick={() => {
                          const existing = selectedSkills.find(
                            (s) => s.skill === skill,
                          );
                          if (existing) {
                            setSelectedSkills((prev) =>
                              prev.map((s) =>
                                s.skill === skill ? { skill, band } : s,
                              ),
                            );
                          } else {
                            setSelectedSkills((prev) => [
                              ...prev,
                              { skill, band },
                            ]);
                          }
                        }}
                        className={`px-3 py-1 text-xs rounded-full border transition-colors ${
                          selectedSkills.some(
                            (s) => s.skill === skill && s.band === band,
                          )
                            ? "bg-primary text-primary-foreground border-primary"
                            : "bg-background text-foreground border-gray-300 dark:border-gray-700 hover:bg-secondary"
                        }`}
                      >
                        {band.charAt(0).toUpperCase() + band.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            <button
              onClick={submitSkillsAndGetRecommendations}
              disabled={selectedSkills.length === 0}
              className="w-full py-2 bg-primary text-primary-foreground rounded-lg disabled:opacity-50"
            >
              Continue
            </button>
          </div>
        </div>
        {InputArea}
      </div>
    );
  }

  // Render recommendation cards
  if (stage === "recommendations") {
    return (
      <div className="flex h-full flex-col">
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5">
          {/* Messages area showing learning path if no issues match */}
          {messages.map((msg, i) => (
            <MessageBubble key={i} msg={msg} />
          ))}

          {/* Issues grid - only shown if recommendations exist */}
          {recommendations.length > 0 && (
            <div>
              <h3 className="text-xl font-display font-bold mb-4">
                Recommended Issues
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {recommendations.map((issue) => (
                  <div
                    key={issue.id}
                    className="bg-card border border-gray-200 dark:border-gray-800 rounded-xl p-4 space-y-3"
                  >
                    <div className="flex justify-between items-start">
                      <h4 className="font-semibold text-base">{issue.title}</h4>
                      <span
                        className={`text-xs px-2 py-0.5 rounded-full ${getDifficultyClass(issue.difficulty)}`}
                      >
                        {issue.difficulty}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {issue.skills?.slice(0, 3).map((skill) => (
                        <span
                          key={skill}
                          className="text-xs bg-secondary px-2 py-0.5 rounded-full"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Match: {Math.round((issue.combined_score || 0) * 100)}%
                    </div>
                    {issue.labels && issue.labels.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {issue.labels.slice(0, 2).map((label) => (
                          <span
                            key={label}
                            className="text-xs text-primary border border-primary/30 px-2 py-0.5 rounded-full"
                          >
                            {label}
                          </span>
                        ))}
                      </div>
                    )}
                    <button
                      onClick={() => selectIssue(issue)}
                      className="w-full py-2 bg-primary text-primary-foreground rounded-lg"
                    >
                      Select This Issue
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        {InputArea}
      </div>
    );
  }

  // Initial state (only URL input)
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

  // Normal chat mode (after issue selected, regular conversation)
  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-gray-200 dark:border-gray-800 bg-background">
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

      {/* Input area (normal chat) */}
      {InputArea}
    </div>
  );
}

// Copy-button enabled message bubble (unchanged from original but moved outside)
function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  const [copied, setCopied] = useState(false);
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[78%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-primary text-white shadow-md"
            : "bg-gray-100 dark:bg-gray-800 text-foreground ring-1 ring-gray-200 dark:ring-gray-700",
        )}
      >
        {isUser ? (
          <span>{msg.content}</span>
        ) : (
          <div className="prose prose-sm dark:prose-invert max-w-none break-words">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              components={{
                code({ _, inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || "");
                  if (!inline && match) {
                    const codeString = String(children).replace(/\n$/, "");
                    return (
                      <div className="relative group">
                        <pre className={className} {...props}>
                          <code className={match[1]}>{children}</code>
                        </pre>
                        <button
                          onClick={() => {
                            navigator.clipboard.writeText(codeString);
                            setCopied(true);
                            setTimeout(() => setCopied(false), 2000);
                          }}
                          className="absolute top-2 right-2 p-1 rounded bg-gray-800 opacity-0 group-hover:opacity-100 transition"
                        >
                          {copied ? <Check size={14} /> : <Copy size={14} />}
                        </button>
                      </div>
                    );
                  }
                  return (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                },
              }}
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
