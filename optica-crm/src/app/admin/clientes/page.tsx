import Link from "next/link";
import { Edit3, Plus, Trash2 } from "lucide-react";
import { prisma } from "@/lib/db/prisma";
import { requireSession } from "@/lib/auth/session";
import { formatDate } from "@/lib/format";
import { createCustomerAction, deleteCustomerAction } from "./actions";

export const dynamic = "force-dynamic";

async function getCustomers(organizationId: string) {
  try {
    return await prisma.customer.findMany({
      where: {
        organizationId,
        deletedAt: null,
      },
      orderBy: { createdAt: "desc" },
      take: 50,
    });
  } catch {
    return [];
  }
}

export default async function CustomersPage() {
  const session = await requireSession();
  const customers = await getCustomers(session.organizationId);

  return (
    <div className="mt-7 grid gap-6 xl:grid-cols-[380px_1fr]">
      <section className="rounded-lg bg-white p-5">
        <div className="flex items-center gap-2">
          <Plus size={18} />
          <h2 className="text-lg font-semibold">Nuevo cliente</h2>
        </div>
        <form action={createCustomerAction} className="mt-5 grid gap-4">
          <label className="grid gap-2 text-sm font-medium">
            Nombre
            <input name="firstName" className="h-11 rounded-md border border-zinc-200 px-3" required />
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Apellidos
            <input name="lastName" className="h-11 rounded-md border border-zinc-200 px-3" />
          </label>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-1">
            <label className="grid gap-2 text-sm font-medium">
              Telefono
              <input name="phone" className="h-11 rounded-md border border-zinc-200 px-3" />
            </label>
            <label className="grid gap-2 text-sm font-medium">
              Email
              <input name="email" type="email" className="h-11 rounded-md border border-zinc-200 px-3" />
            </label>
          </div>
          <label className="grid gap-2 text-sm font-medium">
            Contacto preferido
            <select name="preferredContactMethod" className="h-11 rounded-md border border-zinc-200 px-3">
              <option value="">Sin preferencia</option>
              <option value="phone">Telefono</option>
              <option value="whatsapp">WhatsApp</option>
              <option value="email">Email</option>
            </select>
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Notas
            <textarea name="notes" className="min-h-24 rounded-md border border-zinc-200 p-3" />
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input name="marketingOptIn" type="checkbox" />
            Acepta promociones
          </label>
          <button className="h-11 rounded-full bg-black px-5 text-sm font-semibold text-white">
            Guardar cliente
          </button>
        </form>
      </section>

      <section className="rounded-lg bg-white p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Clientes</h2>
          <span className="text-sm text-zinc-500">{customers.length} registros</span>
        </div>
        <div className="mt-5 overflow-x-auto rounded-lg border border-zinc-100">
          <div className="min-w-[760px]">
            {customers.length ? (
              customers.map((customer) => (
                <div
                  key={customer.id}
                  className="grid grid-cols-[1fr_160px_220px_110px_100px] gap-4 border-b border-zinc-100 px-4 py-4 text-sm last:border-b-0"
                >
                  <div>
                    <p className="font-semibold">
                      {customer.firstName} {customer.lastName}
                    </p>
                    <p className="text-zinc-500">{formatDate(customer.createdAt)}</p>
                  </div>
                  <span>{customer.phone ?? "Sin telefono"}</span>
                  <span>{customer.email ?? "Sin email"}</span>
                  <Link
                    href={`/admin/clientes/${customer.id}`}
                    className="inline-flex items-center gap-2 font-semibold"
                  >
                    Editar
                    <Edit3 size={15} />
                  </Link>
                  <form action={deleteCustomerAction}>
                    <input type="hidden" name="id" value={customer.id} />
                    <button className="inline-flex items-center gap-2 text-red-600">
                      Baja
                      <Trash2 size={15} />
                    </button>
                  </form>
                </div>
              ))
            ) : (
              <p className="px-4 py-4 text-sm text-zinc-500">
                No hay clientes para mostrar.
              </p>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
