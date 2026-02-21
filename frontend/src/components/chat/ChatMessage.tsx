import { motion } from "framer-motion";
import { Scale, User, FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import { Source } from "@/lib/api";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
  sources?: Source[];
}

export const ChatMessage = ({ role, content, timestamp, sources }: ChatMessageProps) => {
  const isUser = role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "flex gap-4 p-4",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      <div
        className={cn(
          "flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center",
          isUser
            ? "bg-secondary text-secondary-foreground"
            : "bg-primary text-primary-foreground"
        )}
      >
        {isUser ? <User className="w-5 h-5" /> : <Scale className="w-5 h-5" />}
      </div>

      <div
        className={cn(
          "flex-1 max-w-[80%]",
          isUser ? "text-right" : "text-left"
        )}
      >
        <div
          className={cn(
            "inline-block p-4 rounded-2xl",
            isUser
              ? "bg-primary text-primary-foreground rounded-br-md"
              : "bg-card shadow-card border border-border rounded-bl-md"
          )}
        >
          <p className="text-sm leading-relaxed whitespace-pre-wrap font-body">
            {content}
          </p>
        </div>
        
        {sources && sources.length > 0 && (
          <div className="mt-3 space-y-2">
            <p className="text-xs font-medium text-muted-foreground flex items-center gap-1">
              <FileText className="w-3 h-3" />
              Источники:
            </p>
            {sources.map((source, index) => (
              <div
                key={index}
                className="text-xs bg-card border border-border rounded-lg p-3 shadow-sm"
              >
                <p className="font-medium text-foreground mb-1">
                  {source.codex} - Статья {source.article_id}
                </p>
                <p className="text-muted-foreground mb-1">{source.article_title}</p>
                <p className="text-xs text-muted-foreground">
                  Релевантность: {(source.score * 100).toFixed(1)}%
                </p>
              </div>
            ))}
          </div>
        )}
        
        {timestamp && (
          <p className="text-xs text-muted-foreground mt-2">{timestamp}</p>
        )}
      </div>
    </motion.div>
  );
};
