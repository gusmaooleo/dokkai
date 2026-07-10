"use client";

/**
 * Authenticated shell: guards every route under `(app)` behind a session
 * (redirects unauthenticated visitors to /login), then renders the sidebar,
 * header, default-admin banner and content area, faithful to
 * `dokkai-ui-prototype.html`'s `showApp` block (~L111-483).
 */

import { useEffect, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { LoaderCircle } from "lucide-react";

import { useAuth } from "@/lib/auth";
import { ProjectProvider } from "@/lib/project";
import { ToastProvider } from "@/lib/toast";
import { AdminBanner } from "@/components/shell/admin-banner";
import { Header } from "@/components/shell/header";
import { MobileNav } from "@/components/shell/mobile-nav";
import { Sidebar } from "@/components/shell/sidebar";

export default function AppLayout({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  useEffect(() => {
    if (!loading && !user) router.replace("/login");
  }, [loading, user, router]);

  if (loading || !user) {
    return (
      <div className="grid h-screen place-items-center bg-background">
        <LoaderCircle
          size={28}
          strokeWidth={2.4}
          className="animate-[dk-spin_0.7s_linear_infinite] text-muted-foreground"
        />
      </div>
    );
  }

  return (
    <ToastProvider>
      <ProjectProvider>
        <div className="flex h-screen overflow-hidden bg-background">
          <Sidebar />
          <MobileNav open={mobileNavOpen} onClose={() => setMobileNavOpen(false)} />
          <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
            <Header onOpenMobileNav={() => setMobileNavOpen(true)} />
            <AdminBanner />
            <main className="min-h-0 flex-1 overflow-auto">{children}</main>
          </div>
        </div>
      </ProjectProvider>
    </ToastProvider>
  );
}
