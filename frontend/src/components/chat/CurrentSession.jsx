// frontend/src/components/chat/CurrentSession.jsx
import { Send, Loader2, Check, Copy, ThumbsUp, ThumbsDown } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

import { useTheme } from "@/components/ThemeToggle";
import { api, poll } from "@/lib/api";
import { upsertSession, newLocalId } from "@/lib/sessionStore";
import { cn } from "@/lib/utils";

const WELCOME = {
  role: "ai",
  content:
    "Hey! I'm here to help you find a meaningful open source issue to work on — and guide you through it without handing you the solution.\n\nWhat's the GitHub repo URL you'd like to contribute to?",
};

export function CurrentSession({ activeSession }) {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const mountedRef = useRef(true);
  const abortControllerRef = useRef(null);

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

  useEffect(() => {
    mountedRef.current = true;
    abortControllerRef.current = new AbortController();
    return () => {
      mountedRef.current = false;
      abortControllerRef.current?.abort();
    };
  }, []);

  const [input, setInput] = useState("");
  const [error, setError] = useState(null);
  const [repoSkills, setRepoSkills] = useState([]);
  const [selectedSkills, setSelectedSkills] = useState([]);
  const [selectedIssueId, setSelectedIssueId] = useState(
    activeSession?.selectedIssueId || null,
  );
  const [_, setRecommendations] = useState([]);
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
      selectedIssueId,
      updatedAt: Date.now(),
    });
  }, [repoUrl, repoLabel, sessionId, phase, messages, selectedIssueId]);

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
        {
          intervalMs: 2000,
          timeoutMs: 5 * 60_000,
          signal: abortControllerRef.current?.signal,
        },
      );
      if (repo.status === "failed") {
        throw new Error(repo.error_message || "Repository analysis failed.");
      }

      const session = await api.createSession(repoId);
      setSessionId(session.id);
      setRepoSkills(repo.skills_found || []);
      setStage("skills");
      setPhase(session.phase || "onboarding");

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

  const submitSkillsAndGetRecommendations = async () => {
    if (selectedSkills.length === 0) return;
    setStage("thinking");
    setError(null);
    try {
      await api.submitStructuredSkills(sessionId, selectedSkills);
      const recs = await api.getRecommendations(sessionId);

      if (recs.recommendations && recs.recommendations.length > 0) {
        setRecommendations(recs.recommendations);
        setMessages((prev) => [
          ...prev,
          {
            role: "ai",
            content: `Great! I found ${recs.recommendations.length} issues that match your skills. Select one to start working.`,
            recommendations: recs.recommendations,
          },
        ]);
      } else if (recs.learning_path) {
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

  const selectIssue = async (issue) => {
    setError(null);
    try {
      await api.selectIssue(sessionId, issue);
      setSelectedIssueId(issue.id);
      setStage("thinking");

      const userMessage = `I'll work on issue #${issue.id}: ${issue.title}`;
      setMessages((prev) => [...prev, { role: "user", content: userMessage }]);

      const accepted = await api.sendMessage(
        sessionId,
        "Let's start working on this issue.",
      );

      const result = await poll(
        () => api.chatResult(accepted.task_id),
        (r) => r.status === "done",
        {
          intervalMs: 1500,
          timeoutMs: 2 * 60_000,
          signal: abortControllerRef.current?.signal,
        },
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
        {
          intervalMs: 1500,
          timeoutMs: 2 * 60_000,
          signal: abortControllerRef.current?.signal,
        },
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
  };

  const getDifficultyClass = (difficulty) => {
    if (difficulty === "beginner") {
      return isDark
        ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/20"
        : "bg-emerald-50 text-emerald-700 border border-emerald-200";
    }
    if (difficulty === "advanced") {
      return isDark
        ? "bg-red-500/15 text-red-400 border border-red-500/20"
        : "bg-red-50 text-red-700 border border-red-200";
    }
    return isDark
      ? "bg-amber-500/15 text-amber-400 border border-amber-500/20"
      : "bg-amber-50 text-amber-700 border border-amber-200";
  };

  const getButtonDisabledState = (isThisSelected, hasAnySelected) => {
    if (isThisSelected) return true;
    if (hasAnySelected) return true;
    return false;
  };

  const getButtonClassName = (isThisSelected, hasAnySelected) => {
    if (isThisSelected) {
      return isDark
        ? "w-full py-2 text-xs font-semibold rounded-xl transition-all bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 cursor-default"
        : "w-full py-2 text-xs font-semibold rounded-xl transition-all bg-emerald-50 text-emerald-700 border border-emerald-200 cursor-default";
    }
    if (hasAnySelected) {
      return isDark
        ? "w-full py-2 text-xs font-semibold rounded-xl transition-all bg-white/5 text-white/20 border border-white/10 cursor-not-allowed"
        : "w-full py-2 text-xs font-semibold rounded-xl transition-all bg-black/5 text-black/20 border border-black/10 cursor-not-allowed";
    }
    return isDark
      ? "w-full py-2 text-xs font-semibold rounded-xl transition-all bg-[#2541B2] text-white hover:bg-[#1098F7] cursor-pointer"
      : "w-full py-2 text-xs font-semibold rounded-xl transition-all bg-[#2541B2] text-white hover:bg-[#1098F7] cursor-pointer";
  };

  const getButtonText = (isThisSelected, hasAnySelected) => {
    if (isThisSelected) return "Selected";
    if (hasAnySelected) return "Unavailable";
    return "Select This Issue";
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
    <div
      className={`border-t p-4 ${isDark ? "border-white/6" : "border-black/6"}`}
    >
      {error && <p className="mb-2 text-xs text-red-500">{error}</p>}
      <div
        className={`flex items-end gap-2 rounded-2xl border p-2 transition-colors focus-within:border-[#2541B2] ${
          isDark ? "border-white/8 bg-white/3" : "border-black/8 bg-black/2"
        }`}
      >
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
          className={`flex-1 resize-none bg-transparent px-3 py-2.5 text-sm outline-none disabled:opacity-50 ${
            isDark
              ? "text-white placeholder:text-white/25"
              : "text-black placeholder:text-black/25"
          }`}
          style={{ height: 40, maxHeight: 160 }}
        />
        <button
          onClick={handleSend}
          disabled={
            busy || (stage !== "ready" && stage !== "idle") || !input.trim()
          }
          aria-label="Send"
          className={`flex h-8 w-8 items-center justify-center rounded-xl transition-all hover:scale-105 active:scale-95 disabled:cursor-not-allowed disabled:opacity-40 ${
            isDark ? "bg-white text-black" : "bg-[#000000] text-white"
          }`}
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
          <div
            className={`max-w-2xl mx-auto rounded-2xl border p-6 space-y-6 ${
              isDark ? "bg-white/2 border-white/6" : "bg-white border-black/6"
            }`}
          >
            <h3
              className={`text-xl font-bold ${isDark ? "text-white" : "text-black"}`}
            >
              Your Skills
            </h3>
            <p
              className={`text-sm ${isDark ? "text-white/50" : "text-black/50"}`}
            >
              Select the skills you have and your proficiency level.
            </p>
            <div className="space-y-4">
              {repoSkills.slice(0, 8).map((skill) => (
                <div
                  key={skill}
                  className="flex items-center justify-between gap-4"
                >
                  <span
                    className={`text-sm font-medium w-32 ${isDark ? "text-white" : "text-black"}`}
                  >
                    {skill}
                  </span>
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
                        className={`px-3 py-1 text-xs rounded-full border font-medium transition-all cursor-pointer ${
                          selectedSkills.some(
                            (s) => s.skill === skill && s.band === band,
                          )
                            ? isDark
                              ? "bg-[#2541B2] text-white border-[#2541B2]"
                              : "bg-[#2541B2] text-white border-[#2541B2]"
                            : isDark
                              ? "bg-white/5 text-white/60 border-white/10 hover:bg-white/10"
                              : "bg-black/5 text-black/60 border-black/10 hover:bg-black/10"
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
              className={`w-full py-2.5 font-medium rounded-xl disabled:opacity-40 transition-all cursor-pointer ${
                isDark
                  ? "bg-white text-black hover:bg-white/90"
                  : "bg-[#000000] text-white hover:bg-[#2541B2]"
              }`}
            >
              Continue
            </button>
          </div>
        </div>
        {InputArea}
      </div>
    );
  }

  // Initial state
  if (isInitial) {
    return (
      <div className="h-full flex flex-col">
        <div className="flex-1 flex items-center justify-center px-6">
          <div className="w-full max-w-2xl text-center">
            <h2
              className={`text-2xl font-bold mb-2 ${isDark ? "text-white" : "text-black"}`}
            >
              Let&apos;s find your next open-source contribution
            </h2>
            <p
              className={`text-sm mb-8 ${isDark ? "text-white/40" : "text-black/40"}`}
            >
              Paste a GitHub repository URL to get started
            </p>
            {InputArea}
          </div>
        </div>
      </div>
    );
  }

  // Normal chat mode
  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div
        className={`flex items-center justify-between px-6 py-3 border-b ${
          isDark ? "border-white/6" : "border-black/6"
        }`}
      >
        <div className="flex items-center gap-3">
          <span
            className={`font-semibold text-sm ${isDark ? "text-white" : "text-black"}`}
          >
            {repoLabel || "No repo selected"}
          </span>
          {phase && repoLabel && (
            <span
              className={`rounded-lg px-2.5 py-0.5 text-[10px] font-medium ${
                isDark
                  ? "bg-[#2541B2]/20 text-[#1098F7]"
                  : "bg-[#2541B2]/10 text-[#2541B2]"
              }`}
            >
              {phase}
            </span>
          )}
        </div>
        <div
          className={`flex items-center gap-2 text-[10px] ${isDark ? "text-white/30" : "text-black/30"}`}
        >
          <span
            className={cn(
              "w-1.5 h-1.5 rounded-full",
              busy ? "bg-amber-400 animate-pulse" : "bg-emerald-400",
            )}
          />
          {statusText}
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-6 py-6 space-y-6"
      >
        {messages.map((msg, i) => (
          <MessageBubble
            key={i}
            msg={msg}
            selectedIssueId={selectedIssueId}
            onSelectIssue={selectIssue}
            getButtonDisabledState={getButtonDisabledState}
            getButtonClassName={getButtonClassName}
            getButtonText={getButtonText}
            getDifficultyClass={getDifficultyClass}
            isDark={isDark}
          />
        ))}
      </div>

      {InputArea}
    </div>
  );
}

function MessageBubble({
  msg,
  selectedIssueId,
  onSelectIssue,
  getButtonDisabledState,
  getButtonClassName,
  getButtonText,
  getDifficultyClass,
  isDark,
}) {
  const isUser = msg.role === "user";
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState(null);

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleFeedback = (type) => {
    setFeedback(type);
  };

  return (
    <div
      className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}
    >
      <div
        className={cn(
          "max-w-[85%] leading-relaxed",
          isUser
            ? `rounded-2xl px-4 py-3 ${isDark ? "bg-[#2541B2] text-white" : "bg-[#2541B2] text-white"}`
            : "w-full py-1",
        )}
      >
        {isUser ? (
          <span className="whitespace-pre-wrap text-sm">{msg.content}</span>
        ) : (
          <div className="w-full">
            <div className="prose prose-sm max-w-none break-words">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeHighlight]}
                components={{
                  code({ _, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || "");
                    if (!inline && match) {
                      const codeString = String(children).replace(/\n$/, "");
                      return (
                        <div className="relative group/my-2">
                          <pre
                            className={`${className} rounded-xl p-4 overflow-x-auto text-xs`}
                            {...props}
                          >
                            <code className={match[1]}>{children}</code>
                          </pre>
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(codeString);
                              setCopied(true);
                              setTimeout(() => setCopied(false), 2000);
                            }}
                            className={`absolute top-2 right-2 p-1.5 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity ${
                              isDark
                                ? "bg-white/10 text-white/60 hover:text-white"
                                : "bg-black/10 text-black/60 hover:text-black"
                            }`}
                          >
                            {copied ? <Check size={14} /> : <Copy size={14} />}
                          </button>
                        </div>
                      );
                    }
                    return (
                      <code
                        className={`${className} px-1 py-0.5 rounded text-xs ${
                          isDark
                            ? "bg-white/10 text-white/80"
                            : "bg-black/10 text-black/80"
                        }`}
                        {...props}
                      >
                        {children}
                      </code>
                    );
                  },
                }}
              >
                {msg.content}
              </ReactMarkdown>
            </div>

            {/* Inline recommendations */}
            {msg.recommendations && msg.recommendations.length > 0 && (
              <div
                className={`mt-4 pt-4 ${isDark ? "border-t border-white/6" : "border-t border-black/6"}`}
              >
                <h4
                  className={`font-semibold text-sm mb-3 ${isDark ? "text-white" : "text-black"}`}
                >
                  Recommended Issues
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {msg.recommendations.map((issue) => {
                    const isThisSelected = selectedIssueId === issue.id;
                    const hasAnySelected = selectedIssueId !== null;
                    return (
                      <div
                        key={issue.id}
                        className={cn(
                          "border rounded-xl p-4 space-y-3 transition-all",
                          isThisSelected
                            ? isDark
                              ? "border-[#2541B2] ring-1 ring-[#2541B2]/20 bg-[#2541B2]/5"
                              : "border-[#2541B2] ring-1 ring-[#2541B2]/20 bg-[#2541B2]/5"
                            : isDark
                              ? "border-white/6 bg-white/2"
                              : "border-black/6 bg-white",
                        )}
                      >
                        <div className="flex justify-between items-start gap-2">
                          <h5
                            className={`font-semibold text-sm leading-snug line-clamp-2 ${isDark ? "text-white" : "text-black"}`}
                          >
                            {issue.title}
                          </h5>
                          <span
                            className={cn(
                              "text-[10px] px-2 py-0.5 rounded-full font-medium shrink-0 uppercase tracking-wider",
                              getDifficultyClass(issue.difficulty),
                            )}
                          >
                            {issue.difficulty}
                          </span>
                        </div>

                        {issue.summary && (
                          <p
                            className={`text-xs line-clamp-3 leading-relaxed ${isDark ? "text-white/40" : "text-black/40"}`}
                          >
                            {issue.summary}
                          </p>
                        )}

                        <div className="flex flex-wrap gap-1">
                          {issue.skills?.slice(0, 3).map((skill) => (
                            <span
                              key={skill}
                              className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                                isDark
                                  ? "bg-white/5 text-white/60"
                                  : "bg-black/5 text-black/60"
                              }`}
                            >
                              {skill}
                            </span>
                          ))}
                        </div>

                        <button
                          disabled={getButtonDisabledState(
                            isThisSelected,
                            hasAnySelected,
                          )}
                          onClick={() => onSelectIssue(issue)}
                          className={getButtonClassName(
                            isThisSelected,
                            hasAnySelected,
                          )}
                        >
                          {getButtonText(isThisSelected, hasAnySelected)}
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Action buttons */}
            {!msg.pending && msg.content && (
              <div
                className={`flex items-center gap-2 mt-3 text-xs ${isDark ? "text-white/30" : "text-black/30"}`}
              >
                <button
                  onClick={handleCopy}
                  className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border transition-colors ${
                    isDark
                      ? "border-white/8 hover:bg-white/5 hover:text-white/60"
                      : "border-black/8 hover:bg-black/5 hover:text-black/60"
                  }`}
                >
                  {copied ? (
                    <Check className="h-3.5 w-3.5" />
                  ) : (
                    <Copy className="h-3.5 w-3.5" />
                  )}
                  <span>{copied ? "Copied" : "Copy"}</span>
                </button>
                <button
                  onClick={() => handleFeedback("good")}
                  className={cn(
                    "flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border transition-colors",
                    feedback === "good"
                      ? isDark
                        ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                        : "bg-emerald-50 border-emerald-200 text-emerald-600"
                      : isDark
                        ? "border-white/8 hover:bg-white/5 hover:text-white/60"
                        : "border-black/8 hover:bg-black/5 hover:text-black/60",
                  )}
                >
                  <ThumbsUp className="h-3.5 w-3.5" />
                  <span>Good</span>
                </button>
                <button
                  onClick={() => handleFeedback("bad")}
                  className={cn(
                    "flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border transition-colors",
                    feedback === "bad"
                      ? isDark
                        ? "bg-red-500/10 border-red-500/20 text-red-400"
                        : "bg-red-50 border-red-200 text-red-600"
                      : isDark
                        ? "border-white/8 hover:bg-white/5 hover:text-white/60"
                        : "border-black/8 hover:bg-black/5 hover:text-black/60",
                  )}
                >
                  <ThumbsDown className="h-3.5 w-3.5" />
                  <span>Bad</span>
                </button>
              </div>
            )}
          </div>
        )}
        {msg.pending && (
          <span className="ml-1 inline-flex gap-1 mt-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[#1098F7] animate-bounce" />
            <span
              className="w-1.5 h-1.5 rounded-full bg-[#1098F7] animate-bounce"
              style={{ animationDelay: "150ms" }}
            />
            <span
              className="w-1.5 h-1.5 rounded-full bg-[#1098F7] animate-bounce"
              style={{ animationDelay: "300ms" }}
            />
          </span>
        )}
      </div>
    </div>
  );
}
