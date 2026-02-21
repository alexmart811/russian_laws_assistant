import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";

const About = () => {
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
              О нас
            </h1>
            <p className="text-muted-foreground mb-4">
              ЮрАссистент — это AI-помощник, который помогает ориентироваться в
              российском законодательстве, формулировать запросы и получать
              справочные ответы в удобной форме.
            </p>
            <p className="text-muted-foreground">
              Сервис не заменяет профессиональную юридическую консультацию, а
              служит отправной точкой для понимания ваших прав и возможных
              действий.
            </p>
          </div>
        </main>
      </div>
    </div>
  );
};

export default About;
