import { motion } from "framer-motion";
import { Scale } from "lucide-react";

export const TypingIndicator = () => {
  return (
    <div className="flex gap-4 p-4">
      <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary text-primary-foreground flex items-center justify-center">
        <Scale className="w-5 h-5" />
      </div>

      <div className="bg-card shadow-card border border-border rounded-2xl rounded-bl-md p-4">
        <div className="flex gap-1.5">
          {[0, 1, 2].map((i) => (
            <motion.div
              key={i}
              className="w-2 h-2 rounded-full bg-legal-gold"
              animate={{
                y: [0, -6, 0],
              }}
              transition={{
                duration: 0.6,
                repeat: Infinity,
                delay: i * 0.1,
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
};
