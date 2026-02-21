import { motion } from "framer-motion";
import { Scale, Sparkles } from "lucide-react";

export const WelcomeHero = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="text-center py-12 px-4"
    >
      <motion.div
        initial={{ scale: 0.8 }}
        animate={{ scale: 1 }}
        transition={{ delay: 0.2, type: "spring" }}
        className="w-20 h-20 rounded-2xl bg-primary mx-auto mb-6 flex items-center justify-center shadow-card"
      >
        <Scale className="w-10 h-10 text-primary-foreground" />
      </motion.div>

      <h1 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-4">
        Добро пожаловать в{" "}
        <span className="text-gradient">Юридический Ассистент</span>
      </h1>

      <p className="text-muted-foreground text-lg max-w-xl mx-auto mb-8">
        Ваш персональный AI-помощник по юридическим вопросам. Получайте
        мгновенные консультации на основе актуального законодательства РФ.
      </p>

      <div className="flex items-center justify-center gap-2 text-sm text-legal-gold">
        <Sparkles className="w-4 h-4" />
        <span>Работает на базе современных AI технологий</span>
      </div>
    </motion.div>
  );
};
