import { motion } from "framer-motion";
import {
  FileText,
  Briefcase,
  Home,
  Users,
  Shield,
  Car,
  Landmark,
  HeartHandshake,
} from "lucide-react";

interface QuickAction {
  icon: React.ReactNode;
  title: string;
  description: string;
  query: string;
}

const quickActions: QuickAction[] = [
  {
    icon: <Briefcase className="w-5 h-5" />,
    title: "Трудовое право",
    description: "Увольнение, отпуска, зарплата",
    query: "Какие права есть у работника при увольнении?",
  },
  {
    icon: <Home className="w-5 h-5" />,
    title: "Недвижимость",
    description: "Покупка, аренда, споры",
    query: "Как проверить квартиру перед покупкой?",
  },
  {
    icon: <Users className="w-5 h-5" />,
    title: "Семейное право",
    description: "Развод, алименты, опека",
    query: "Как рассчитываются алименты?",
  },
  {
    icon: <Shield className="w-5 h-5" />,
    title: "Защита прав",
    description: "Потребители, жалобы",
    query: "Как вернуть товар ненадлежащего качества?",
  },
  {
    icon: <Car className="w-5 h-5" />,
    title: "ДТП и авто",
    description: "Страховки, штрафы, аварии",
    query: "Что делать после ДТП?",
  },
  {
    icon: <Landmark className="w-5 h-5" />,
    title: "Налоги",
    description: "Вычеты, декларации, споры",
    query: "Как получить налоговый вычет?",
  },
  {
    icon: <FileText className="w-5 h-5" />,
    title: "Документы",
    description: "Договоры, заявления",
    query: "Как составить исковое заявление?",
  },
  {
    icon: <HeartHandshake className="w-5 h-5" />,
    title: "Наследство",
    description: "Завещания, доли, споры",
    query: "Как вступить в наследство?",
  },
];

interface QuickActionsProps {
  onSelect: (query: string) => void;
}

export const QuickActions = ({ onSelect }: QuickActionsProps) => {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-4">
      {quickActions.map((action, index) => (
        <motion.button
          key={action.title}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
          onClick={() => onSelect(action.query)}
          className="group p-4 bg-card rounded-xl border border-border hover:border-legal-gold/50 hover:shadow-card transition-all duration-300 text-left"
        >
          <div className="w-10 h-10 rounded-lg bg-primary/10 text-primary flex items-center justify-center mb-3 group-hover:bg-legal-gold/20 group-hover:text-legal-gold transition-colors">
            {action.icon}
          </div>
          <h3 className="font-semibold text-sm text-foreground mb-1 font-display">
            {action.title}
          </h3>
          <p className="text-xs text-muted-foreground">{action.description}</p>
        </motion.button>
      ))}
    </div>
  );
};
