import { useState } from "react";
import { motion } from "framer-motion";
import { Send, Paperclip, Mic } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading?: boolean;
}

export const ChatInput = ({ onSend, isLoading }: ChatInputProps) => {
  const [message, setMessage] = useState("");

  const handleSend = () => {
    if (message.trim() && !isLoading) {
      onSend(message.trim());
      setMessage("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-4 border-t border-border bg-card/80 backdrop-blur-sm"
    >
      <div className="max-w-4xl mx-auto">
        <div className="flex items-end gap-3 bg-background rounded-2xl border border-border p-2 shadow-soft focus-within:ring-2 focus-within:ring-legal-gold/50 transition-all">
          <Button
            variant="ghost"
            size="icon"
            className="flex-shrink-0 text-muted-foreground hover:text-foreground"
          >
            <Paperclip className="w-5 h-5" />
          </Button>

          <Textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Задайте юридический вопрос..."
            className="flex-1 min-h-[44px] max-h-[200px] resize-none border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 text-foreground placeholder:text-muted-foreground"
            rows={1}
          />

          <Button
            variant="ghost"
            size="icon"
            className="flex-shrink-0 text-muted-foreground hover:text-foreground"
          >
            <Mic className="w-5 h-5" />
          </Button>

          <Button
            variant="gold"
            size="icon"
            onClick={handleSend}
            disabled={!message.trim() || isLoading}
            className="flex-shrink-0"
          >
            <Send className="w-5 h-5" />
          </Button>
        </div>

        <p className="text-xs text-muted-foreground text-center mt-3">
          ЮрАссистент предоставляет справочную информацию. Для принятия решений
          проконсультируйтесь с квалифицированным юристом.
        </p>
      </div>
    </motion.div>
  );
};
