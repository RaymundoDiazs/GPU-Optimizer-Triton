import Link from "next/link";
import { redirect } from "next/navigation";
import { Glasses, LockKeyhole, Mail } from "lucide-react";
import { loginAction } from "./actions";
import { getCurrentSession } from "@/lib/auth/session";

type LoginPageProps = {
  searchParams: Promise<{
    error?: string;
    next?: string;
  }>;
};

export default async function LoginPage({ searchParams }: LoginPageProps) {
  const params = await searchParams;
  const next = params.next?.startsWith("/admin") ? params.next : "/admin";
  const session = await getCurrentSession();

  if (session) {
    redirect(next);
  }

  return (
    <main className="grid min-h-screen bg-[#f4f2ee] p-4 text-zinc-950 lg:grid-cols-[1fr_0.9fr]">
      <section className="hidden overflow-hidden rounded-lg bg-black p-10 text-white lg:flex lg:flex-col lg:justify-between">
        <Link href="/" className="flex items-center gap-3">
          <span className="grid size-10 place-items-center rounded-full bg-white text-black">
            <Glasses size={22} />
          </span>
          <span className="text-2xl font-semibold">Optica Nova</span>
        </Link>
        <div>
          <p className="text-sm uppercase text-white/50">CRM protegido</p>
          <h1 className="mt-4 max-w-xl text-6xl font-semibold leading-none">
            Operacion, inventario y clientes en una sola vista.
          </h1>
        </div>
      </section>

      <section className="flex items-center justify-center px-2 py-10 sm:px-8">
        <div className="w-full max-w-md rounded-lg bg-white p-7 shadow-[0_24px_80px_rgba(20,20,20,0.08)]">
          <div className="flex items-center gap-3 lg:hidden">
            <span className="grid size-10 place-items-center rounded-full bg-black text-white">
              <Glasses size={22} />
            </span>
            <span className="text-2xl font-semibold">Optica Nova</span>
          </div>

          <div className="mt-8 lg:mt-0">
            <p className="text-sm uppercase text-zinc-500">Acceso interno</p>
            <h2 className="mt-2 text-3xl font-semibold">Iniciar sesion</h2>
          </div>

          {params.error ? (
            <div className="mt-5 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {params.error}
            </div>
          ) : null}

          <form action={loginAction} className="mt-6 grid gap-4">
            <input type="hidden" name="next" value={next} />
            <label className="grid gap-2 text-sm font-medium">
              Correo
              <span className="flex h-12 items-center gap-3 rounded-md border border-zinc-200 px-3">
                <Mail size={18} className="text-zinc-400" />
                <input
                  className="h-full flex-1 bg-transparent outline-none"
                  name="email"
                  type="email"
                  placeholder="admin@opticanova.local"
                  autoComplete="email"
                  required
                />
              </span>
            </label>
            <label className="grid gap-2 text-sm font-medium">
              Contrasena
              <span className="flex h-12 items-center gap-3 rounded-md border border-zinc-200 px-3">
                <LockKeyhole size={18} className="text-zinc-400" />
                <input
                  className="h-full flex-1 bg-transparent outline-none"
                  name="password"
                  type="password"
                  placeholder="Admin123!"
                  autoComplete="current-password"
                  required
                />
              </span>
            </label>
            <button className="mt-2 h-12 rounded-full bg-black px-6 text-sm font-semibold text-white transition hover:bg-zinc-800">
              Entrar al CRM
            </button>
          </form>

          <div className="mt-6 rounded-md bg-zinc-50 p-4 text-sm leading-6 text-zinc-600">
            Usuario inicial del seed: <strong>admin@opticanova.local</strong>
            <br />
            Contrasena inicial: <strong>Admin123!</strong>
          </div>
        </div>
      </section>
    </main>
  );
}
