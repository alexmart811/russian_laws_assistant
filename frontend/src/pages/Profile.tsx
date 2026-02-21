import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import { Button } from "@/components/ui/button";

const Profile = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const navigate = useNavigate();

  const handleLogout = () => {
    // TODO: добавить реальную логику выхода, когда появится backend/auth
    console.log("Logout clicked");
    navigate("/");
  };

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
          <div className="max-w-2xl mx-auto">
            <h1 className="text-3xl md:text-4xl font-display font-bold mb-4">
              Профиль
            </h1>

            <div className="rounded-xl border bg-card p-4 md:p-6 space-y-4">
              <div>
                <h2 className="text-lg font-semibold mb-2">Основная информация</h2>
                <p className="text-sm text-muted-foreground">
                  Имя: <span className="font-medium">Иван Иванов</span>
                </p>
                <p className="text-sm text-muted-foreground">
                  E-mail:{" "}
                  <span className="font-medium">user@example.com</span>
                </p>
              </div>

              <div className="pt-2 space-y-2">
                <h2 className="text-lg font-semibold">Управление аккаунтом</h2>
                <div className="flex flex-col sm:flex-row gap-2">
                  <Button variant="destructive" onClick={handleLogout}>
                    Выйти из аккаунта
                  </Button>
                  <Button
                    variant="outline"
                    onClick={handleLogout}
                  >
                    Выйти на всех устройствах
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default Profile;


