import { motion } from "framer-motion";
import {
  Scale,
  MessageSquare,
  History,
  BookOpen,
  Settings,
  Plus,
  ChevronLeft,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  onHomeClick?: () => void;
}

const recentChats = [
  { id: 1, title: "Трудовой договор", time: "Сегодня" },
  { id: 2, title: "Права потребителя", time: "Вчера" },
  { id: 3, title: "Развод и алименты", time: "3 дня назад" },
  { id: 4, title: "Налоговый вычет", time: "Неделю назад" },
];

export const Sidebar = ({ isOpen, onToggle, onHomeClick }: SidebarProps) => {
  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-foreground/20 backdrop-blur-sm z-40 lg:hidden"
          onClick={onToggle}
        />
      )}

      <motion.aside
        initial={false}
        animate={{ x: isOpen ? 0 : -280 }}
        transition={{ type: "spring", damping: 25, stiffness: 200 }}
        className={cn(
          "fixed left-0 top-0 h-full w-[280px] bg-sidebar text-sidebar-foreground z-50 flex flex-col",
          "lg:relative lg:translate-x-0",
          !isOpen && "lg:w-0 lg:overflow-hidden"
        )}
      >
        {/* Header */}
        <div className="p-4 border-b border-sidebar-border flex items-center justify-between">
          <button
            type="button"
            onClick={onHomeClick}
            className="flex items-center gap-3 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring rounded-lg"
          >
            <div className="w-10 h-10 rounded-xl bg-sidebar-primary flex items-center justify-center">
              <Scale className="w-5 h-5 text-sidebar-primary-foreground" />
            </div>
            <div>
              <h1 className="font-display font-bold text-lg">ЮрАссистент</h1>
              <p className="text-xs text-sidebar-foreground/70">AI помощник</p>
            </div>
          </button>
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggle}
            className="text-sidebar-foreground hover:bg-sidebar-accent"
          >
            <ChevronLeft className="w-5 h-5" />
          </Button>
        </div>

        {/* New Chat Button */}
        <div className="p-4">
          <Button
            variant="gold"
            className="w-full justify-start gap-2"
          >
            <Plus className="w-4 h-4" />
            Новый диалог
          </Button>
        </div>

        {/* Recent Chats */}
        <div className="flex-1 overflow-y-auto px-4">
          <div className="flex items-center gap-2 text-xs text-sidebar-foreground/70 mb-3">
            <History className="w-4 h-4" />
            <span>Недавние диалоги</span>
          </div>
          <div className="space-y-1">
            {recentChats.map((chat) => (
              <button
                key={chat.id}
                className="w-full p-3 rounded-lg text-left hover:bg-sidebar-accent transition-colors group"
              >
                <div className="flex items-start gap-3">
                  <MessageSquare className="w-4 h-4 mt-0.5 text-sidebar-foreground/50 group-hover:text-sidebar-primary" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{chat.title}</p>
                    <p className="text-xs text-sidebar-foreground/50">
                      {chat.time}
                    </p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-sidebar-border space-y-1">
          <button className="w-full p-3 rounded-lg text-left hover:bg-sidebar-accent transition-colors flex items-center gap-3">
            <BookOpen className="w-4 h-4 text-sidebar-foreground/50" />
            <span className="text-sm">База знаний</span>
          </button>
          <button className="w-full p-3 rounded-lg text-left hover:bg-sidebar-accent transition-colors flex items-center gap-3">
            <Settings className="w-4 h-4 text-sidebar-foreground/50" />
            <span className="text-sm">Настройки</span>
          </button>
        </div>
      </motion.aside>
    </>
  );
};
