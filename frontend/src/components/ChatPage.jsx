import { useEffect, useState } from "react";
import { MoreHorizontal } from "lucide-react";
import { ChatSidebar } from "@/components/chat/ChatSidebar";
import { CurrentSession } from "@/components/chat/CurrentSession";
import { SessionHistory } from "@/components/chat/SessionHistory";
import { cn } from "@/lib/utils";

export default function ChatPage() {
  const [tab, setTab] = useState("current"); // "current" | "history"
  const [sidebar, setSidebar] = useState("chat"); // "explore" | "chat" | "history" | "saved"



  const handleSidebar = (key) => {
    setSidebar(key);
    if (key === "history") setTab("history");
    if (key === "chat") setTab("current");
  };

  return (
    <div className="relative min-h-screen bg-background text-foreground">
      <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -left-24 top-0 h-[420px] w-[420px] rounded-full bg-violet-600/15 blur-3xl" />
        <div className="absolute -right-24 bottom-0 h-[420px] w-[420px] rounded-full bg-indigo-600/15 blur-3xl" />
      </div>

      <div className="relative mx-auto flex h-screen max-w-6xl gap-0 p-4">
        <div className="flex w-full overflow-hidden rounded-2xl border border-border/60 bg-card/40 shadow-2xl shadow-violet-950/30 backdrop-blur-xl">
          <ChatSidebar active={sidebar} onChange={handleSidebar} />

          <main className="flex min-w-0 flex-1 flex-col">
            <div className="flex items-center justify-between border-b border-border/50 px-6 pt-4">
              <div className="flex gap-6">
                <TabButton active={tab === "current"} onClick={() => setTab("current")}>
                  Current session
                </TabButton>
                <TabButton active={tab === "history"} onClick={() => setTab("history")}>
                  Session history
                </TabButton>
              </div>
              <button
                aria-label="More"
                className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg border border-border/60 bg-card/60 text-muted-foreground transition-colors hover:bg-violet-600/15 hover:text-foreground"
              >
                <MoreHorizontal className="h-4 w-4" />
              </button>
            </div>

            <div className="min-h-0 flex-1">
              {tab === "current" ? <CurrentSession /> : <SessionHistory />}
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}

function TabButton({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "relative pb-3 text-sm font-semibold transition-colors",
        active ? "text-foreground" : "text-muted-foreground hover:text-foreground"
      )}
    >
      {children}
      {active && (
        <span className="absolute inset-x-0 -bottom-px h-0.5 rounded-full bg-gradient-to-r from-indigo-600 via-violet-600 to-purple-600 dark:from-indigo-400 dark:via-violet-400 dark:to-purple-400" />
      )}
    </button>
  );
}
