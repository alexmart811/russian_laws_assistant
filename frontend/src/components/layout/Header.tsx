import { motion } from "framer-motion";
import { Menu, Scale, Bell, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";

interface HeaderProps {
  onMenuClick: () => void;
  isSidebarOpen: boolean;
}

export const Header = ({ onMenuClick, isSidebarOpen }: HeaderProps) => {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="h-16 border-b border-border bg-card/80 backdrop-blur-sm flex items-center justify-between px-4 lg:px-6"
    >
      <div className="flex items-center gap-4">
        {!isSidebarOpen && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onMenuClick}
            className="text-foreground"
          >
            <Menu className="w-5 h-5" />
          </Button>
        )}

        {!isSidebarOpen && (
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
              <Scale className="w-4 h-4 text-primary-foreground" />
            </div>
            <span className="font-display font-bold text-foreground hidden sm:block">
              ЮрАссистент
            </span>
          </div>
        )}
      </div>

      <div className="flex items-center gap-6">
        <nav className="hidden md:flex items-center gap-4 text-sm">
          <Link
            to="/about"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            О нас
          </Link>
          <Link
            to="/contacts"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            Контакты
          </Link>
        </nav>

        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            className="text-muted-foreground"
          >
            <Bell className="w-5 h-5" />
          </Button>
          <Link to="/profile">
            <Button
              variant="ghost"
              size="icon"
              className="text-muted-foreground"
            >
              <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
                <User className="w-4 h-4" />
              </div>
            </Button>
          </Link>
        </div>
      </div>
    </motion.header>
  );
};
