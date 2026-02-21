import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";

const Contacts = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const navigate = useNavigate();

  return (
    <div className="h-screen flex w-full bg-background">
      <Sidebar
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        onHomeClick={() => navigate("/")}
      />

      <div className="flex-1 flex flex-col min-w-0">
        <Header
          onMenuClick={() => setIsSidebarOpen(true)}
          isSidebarOpen={isSidebarOpen}
        />

        <main className="flex-1 overflow-auto px-4 py-6">
          <div className="max-w-2xl mx-auto text-left">
            <h1 className="text-3xl md:text-4xl font-display font-bold mb-4">
              Контакты
            </h1>
            <p className="text-muted-foreground mb-4">
              По вопросам сотрудничества и развития сервиса вы можете написать
              нам на электронную почту:
            </p>
            <p className="font-medium">
              <a
                href="mailto:support@jur-assistant.ru"
                className="text-primary hover:underline"
              >
                support@jur-assistant.ru
              </a>
            </p>
          </div>
        </main>
      </div>
    </div>
  );
};

export default Contacts;
