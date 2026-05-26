import Link from "next/link";
import {
  Activity,
  Boxes,
  ClipboardList,
  Glasses,
  LogOut,
  Search,
  ShieldCheck,
  Users,
} from "lucide-react";
import type { ReactNode } from "react";
import { logoutAction } from "@/app/login/actions";
import type { SessionPayload } from "@/lib/auth/session-core";

const nav = [
  { label: "Dashboard", href: "/admin", icon: Activity },
  { label: "Clientes", href: "/admin/clientes", icon: Users },
  { label: "Inventario", href: "/admin/inventario", icon: Boxes },
  { label: "Productos", href: "/admin/productos", icon: ClipboardList },
];

type AdminShellProps = {
  children: ReactNode;
  session: SessionPayload;
};

export function AdminShell({ children, session }: AdminShellProps) {
  return (
    <main className="min-h-screen bg-[#f4f2ee] text-zinc-950">
      <div className="grid min-h-screen lg:grid-cols-[280px_1fr]">
        <aside className="border-r border-zinc-200 bg-white px-5 py-6">
          <Link href="/" className="flex items-center gap-3">
            <span className="grid size-9 place-items-center rounded-full bg-black text-white">
              <Glasses size={20} />
            </span>
            <div>
              <p className="text-lg font-semibold">Optica Nova</p>
              <p className="text-xs uppercase text-zinc-500">CRM operativo</p>
            </div>
          </Link>

          <nav className="mt-10 grid gap-1">
            {nav.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="flex items-center gap-3 rounded-md px-3 py-3 text-sm font-medium text-zinc-600 transition hover:bg-zinc-100 hover:text-black"
              >
                <item.icon size={18} />
                {item.label}
              </Link>
            ))}
          </nav>

          <div className="mt-10 rounded-lg border border-zinc-200 bg-[#fbfbfb] p-4">
            <div className="flex items-center gap-2 text-sm font-semibold">
              <ShieldCheck size={18} />
              Sesion activa
            </div>
            <p className="mt-3 text-sm leading-5 text-zinc-600">
              {session.name}
              <br />
              {session.roles.join(", ")}
            </p>
          </div>
        </aside>

        <section className="min-w-0 px-5 py-6 sm:px-8">
          <header className="flex flex-col justify-between gap-4 border-b border-zinc-200 pb-6 md:flex-row md:items-center">
            <div>
              <p className="text-sm uppercase text-zinc-500">Matriz</p>
              <h1 className="mt-1 text-3xl font-semibold">Panel administrativo</h1>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <div className="hidden h-11 w-[320px] items-center gap-3 rounded-full bg-white px-4 text-sm text-zinc-500 md:flex">
                <Search size={18} />
                Buscar cliente, venta o SKU
              </div>
              <Link
                href="/"
                className="inline-flex h-11 items-center gap-2 rounded-full bg-white px-5 text-sm font-semibold text-black transition hover:bg-zinc-100"
              >
                Sitio publico
              </Link>
              <form action={logoutAction}>
                <button className="inline-flex h-11 items-center gap-2 rounded-full bg-black px-5 text-sm font-semibold text-white transition hover:bg-zinc-800">
                  Salir
                  <LogOut size={17} />
                </button>
              </form>
            </div>
          </header>

          {children}
        </section>
      </div>
    </main>
  );
}
