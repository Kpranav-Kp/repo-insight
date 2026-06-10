import { useTheme } from "@/components/ThemeToggle";
import { cn } from "@/lib/utils";

function difficultyClass(difficulty, isDark) {
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
}

export function IssueRecommendations({
  recommendations,
  onSelect,
  selectedId,
  compact = false,
  title = "Recommended Issues",
}) {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  if (!recommendations?.length) return null;

  return (
    <div className={cn("space-y-4", compact && "mt-2")}>
      <h3
        className={cn(
          "font-bold",
          compact ? "text-base" : "text-xl",
          isDark ? "text-white" : "text-black",
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
                "rounded-xl border p-4 space-y-3 transition-all cursor-pointer",
                isSelected
                  ? isDark
                    ? "border-[#2541B2] ring-1 ring-[#2541B2]/20 bg-[#2541B2]/5"
                    : "border-[#2541B2] ring-1 ring-[#2541B2]/20 bg-[#2541B2]/5"
                  : isDark
                    ? "border-white/6 bg-white/2 hover:border-white/12"
                    : "border-black/6 bg-white hover:border-black/12",
              )}
            >
              <div className="flex justify-between items-start gap-2">
                <h4
                  className={`font-semibold text-sm leading-snug ${isDark ? "text-white" : "text-black"}`}
                >
                  #{issue.id} — {issue.title}
                </h4>
                <span
                  className={cn(
                    "shrink-0 text-xs px-2 py-0.5 rounded-full font-medium",
                    difficultyClass(issue.difficulty, isDark),
                  )}
                >
                  {issue.difficulty}
                </span>
              </div>

              {issue.about && (
                <p
                  className={`text-xs leading-relaxed line-clamp-3 ${isDark ? "text-white/40" : "text-black/40"}`}
                >
                  {issue.about}
                </p>
              )}

              {issue.action && (
                <p
                  className={`text-xs leading-relaxed ${isDark ? "text-white/60" : "text-black/60"}`}
                >
                  <span className="font-medium">What to do:</span>{" "}
                  {issue.action}
                </p>
              )}

              <div className="flex flex-wrap gap-1">
                {issue.skills?.slice(0, 4).map((skill) => (
                  <span
                    key={skill}
                    className={`text-xs px-2 py-0.5 rounded-full ${
                      isDark
                        ? "bg-white/5 text-white/60"
                        : "bg-black/5 text-black/60"
                    }`}
                  >
                    {skill}
                  </span>
                ))}
              </div>

              <div
                className={`flex items-center justify-between text-xs ${isDark ? "text-white/30" : "text-black/30"}`}
              >
                <span>
                  Match: {Math.round((issue.combined_score || 0) * 100)}%
                </span>
                {issue.labels?.slice(0, 2).map((label) => (
                  <span
                    key={label}
                    className={`border px-2 py-0.5 rounded-full ${
                      isDark
                        ? "text-[#1098F7] border-[#1098F7]/30"
                        : "text-[#2541B2] border-[#2541B2]/30"
                    }`}
                  >
                    {label}
                  </span>
                ))}
              </div>

              {onSelect && !isSelected && (
                <button
                  type="button"
                  onClick={() => onSelect(issue)}
                  className={`w-full py-2 text-sm rounded-xl font-medium transition-all ${
                    isDark
                      ? "bg-[#2541B2] text-white hover:bg-[#1098F7]"
                      : "bg-[#2541B2] text-white hover:bg-[#1098F7]"
                  }`}
                >
                  Select This Issue
                </button>
              )}
              {isSelected && (
                <div
                  className={`text-xs text-center font-bold p-2 rounded-xl ${
                    isDark
                      ? "text-[#1098F7] bg-[#2541B2]/10"
                      : "text-[#2541B2] bg-[#2541B2]/10"
                  }`}
                >
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
