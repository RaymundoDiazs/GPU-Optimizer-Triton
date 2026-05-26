import type { ReactNode } from "react";
import { AdminShell } from "@/components/admin-shell";
import { requireSession } from "@/lib/auth/session";

export default async function AdminLayout({ children }: { children: ReactNode }) {
  const session = await requireSession();

  return <AdminShell session={session}>{children}</AdminShell>;
}
