// frontend/src/components/chat/IssueRecommendations.jsx
import { cn } from "@/lib/utils";

function difficultyClass(difficulty) {
  if (difficulty === "beginner") {
    return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300";
  }
  if (difficulty === "advanced") {
    return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300";
  }
  return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300";
}

export function IssueRecommendations({
  recommendations,
  onSelect,
  selectedId,
  compact = false,
  title = "Recommended Issues",
}) {
  if (!recommendations?.length) return null;

  return (
    <div className={cn("space-y-4", compact && "mt-2")}>
      <h3
        className={cn(
          "font-display font-bold text-foreground",
          compact ? "text-base" : "text-xl",
        )}
      >
        {title}
      </h3>
      <div
        className={cn(
          "grid gap-4",
          compact ? "grid-cols-1" : "grid-cols-1 md:grid-cols-2",
        )}
      >
        {recommendations.map((issue) => {
          const isSelected =
            selectedId !== null && String(selectedId) === String(issue.id);
          return (
            <div
              key={issue.id}
              className={cn(
                "rounded-xl border bg-card p-4 space-y-3 transition-all cursor-pointer",
                isSelected
                  ? "border-primary ring-2 ring-primary/40 shadow-lg shadow-primary/20 bg-primary/5"
                  : "border-border hover:border-primary/60 hover:shadow-md",
              )}
            >
              <div className="flex justify-between items-start gap-2">
                <h4 className="font-semibold text-sm text-foreground leading-snug">
                  #{issue.id} — {issue.title}
                </h4>
                <span
                  className={cn(
                    "shrink-0 text-xs px-2 py-0.5 rounded-full font-medium",
                    difficultyClass(issue.difficulty),
                  )}
                >
                  {issue.difficulty}
                </span>
              </div>

              {issue.about && (
                <p className="text-xs text-muted-foreground leading-relaxed line-clamp-3">
                  {issue.about}
                </p>
              )}

              {issue.action && (
                <p className="text-xs text-foreground/80 leading-relaxed">
                  <span className="font-medium">What to do:</span>{" "}
                  {issue.action}
                </p>
              )}

              <div className="flex flex-wrap gap-1">
                {issue.skills?.slice(0, 4).map((skill) => (
                  <span
                    key={skill}
                    className="text-xs bg-secondary text-secondary-foreground px-2 py-0.5 rounded-full"
                  >
                    {skill}
                  </span>
                ))}
              </div>

              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>
                  Match: {Math.round((issue.combined_score || 0) * 100)}%
                </span>
                {issue.labels?.slice(0, 2).map((label) => (
                  <span
                    key={label}
                    className="text-primary border border-primary/30 px-2 py-0.5 rounded-full"
                  >
                    {label}
                  </span>
                ))}
              </div>

              {onSelect && !isSelected && (
                <button
                  type="button"
                  onClick={() => onSelect(issue)}
                  className="w-full py-2 text-sm bg-primary text-primary-foreground rounded-lg hover:opacity-90 transition-opacity font-medium"
                >
                  Select This Issue
                </button>
              )}
              {isSelected && (
                <div className="text-xs text-center text-primary font-bold p-2 rounded-lg bg-primary/10">
                  ✓ Currently selected
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
