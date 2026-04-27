import { useState } from "react";
import { Search } from "lucide-react";
import { cn } from "@/lib/utils";

// Replace this with data from your backend later.
const sessions = [
  {
    repo: "django/django",
    issueNumber: 15821,
    title: "bulk_create() ignores update_fields on conflict",
    status: "in_progress",
    stage: "Understand it",
    steps: 4,
    doneSteps: 2,
    updatedLabel: "2 hours ago",
  },
  {
    repo: "fastapi/fastapi",
    issueNumber: 2201,
    title: "Background tasks don't inherit request context",
    status: "in_progress",
    stage: "Plan your PR",
    steps: 4,
    doneSteps: 1,
    updatedLabel: "Yesterday",
  },
  {
    repo: "celery/celery",
    issueNumber: 8812,
    title: "ETA tasks ignore timezone offset in Windows",
    status: "completed",
    stage: "Completed",
    steps: 4,
    doneSteps: 4,
    updatedLabel: "3 days ago",
  },
  {
    repo: "langchain-ai/langchain",
    issueNumber: 12044,
    title: "Streaming output drops final chunk on retry",
    status: "completed",
    stage: "Completed",
    steps: 4,
    doneSteps: 4,
    updatedLabel: "Apr 20",
  },
];

export function SessionHistory() {
  const [query, setQuery] = useState("");
  const filtered = sessions.filter(
    (s) =>
      s.repo.toLowerCase().includes(query.toLowerCase()) ||
      s.title.toLowerCase().includes(query.toLowerCase())
  );

  const inProgress = filtered.filter((s) => s.status === "in_progress");
  const completed = filtered.filter((s) => s.status === "completed");

  return (
    <div className="flex h-full flex-col">
      <div className="px-6 pb-4 pt-5">
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search repos or issues..."
            className="w-full rounded-xl border border-border bg-card/60 py-2.5 pl-10 pr-4 text-sm text-foreground placeholder:text-muted-foreground backdrop-blur-sm transition-all focus:border-violet-500/60 focus:outline-none focus:ring-2 focus:ring-violet-500/20"
          />
        </div>
      </div>

      <div className="flex-1 space-y-6 overflow-y-auto px-6 pb-6">
        {inProgress.length > 0 && (
          <Section title="In progress">
            {inProgress.map((s, i) => (
              <SessionCard key={i} session={s} />
            ))}
          </Section>
        )}
        {completed.length > 0 && (
          <Section title="Completed">
            {completed.map((s, i) => (
              <SessionCard key={i} session={s} />
            ))}
          </Section>
        )}
        {filtered.length === 0 && (
          <p className="py-12 text-center text-sm text-muted-foreground">
            No sessions match your search.
          </p>
        )}
      </div>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <section>
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
        {title}
      </h3>
      <div className="space-y-3">{children}</div>
    </section>
  );
}

function SessionCard({ session }) {
  const inProgress = session.status === "in_progress";
  return (
    <article
      className={cn(
        "group relative overflow-hidden rounded-xl border bg-card/60 p-4 backdrop-blur-sm transition-all hover:-translate-y-0.5 hover:shadow-lg hover:shadow-violet-600/10",
        inProgress
          ? "border-violet-500/40 ring-1 ring-violet-500/30"
          : "border-border/60"
      )}
    >
      {inProgress && (
        <span className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-violet-400 to-transparent" />
      )}
      <header className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-foreground">
            {session.repo}
          </p>
          <p className="mt-0.5 truncate text-sm text-muted-foreground">
            #{session.issueNumber} — {session.title}
          </p>
        </div>
        <span className="shrink-0 text-xs text-muted-foreground">
          {session.updatedLabel}
        </span>
      </header>

      <footer className="mt-4 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <StepDots total={session.steps} done={session.doneSteps} />
          <span
            className={cn(
              "text-xs font-medium",
              session.status === "completed"
                ? "text-emerald-400"
                : "text-foreground"
            )}
          >
            {session.stage}
          </span>
        </div>
        <button
          className={cn(
            "rounded-md px-3 py-1.5 text-xs font-semibold transition-all",
            inProgress
              ? "bg-gradient-to-r from-indigo-600 via-violet-600 to-purple-600 text-white shadow-md shadow-violet-600/30 hover:shadow-violet-600/50 dark:from-indigo-400 dark:via-violet-400 dark:to-purple-400"
              : "border border-border text-muted-foreground hover:bg-violet-600/15 hover:text-foreground"
          )}
        >
          {inProgress ? "Resume →" : "Review"}
        </button>
      </footer>
    </article>
  );
}

function StepDots({ total, done }) {
  return (
    <div className="flex items-center gap-1">
      {Array.from({ length: total }).map((_, i) => (
        <span
          key={i}
          className={cn(
            "h-2 w-2 rounded-full transition-colors",
            i < done
              ? "bg-gradient-to-br from-indigo-500 via-violet-500 to-purple-500"
              : "bg-muted"
          )}
        />
      ))}
    </div>
  );
}
