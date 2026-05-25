import Link from "next/link";
import { notFound } from "next/navigation";
import { prisma } from "@/lib/db/prisma";
import { requireSession } from "@/lib/auth/session";
import { updateCustomerAction } from "../actions";

export const dynamic = "force-dynamic";

type CustomerEditPageProps = {
  params: Promise<{ id: string }>;
};

export default async function CustomerEditPage({ params }: CustomerEditPageProps) {
  const { id } = await params;
  const session = await requireSession();
  const customer = await prisma.customer.findFirst({
    where: {
      id,
      organizationId: session.organizationId,
      deletedAt: null,
    },
  });

  if (!customer) {
    notFound();
  }

  return (
    <section className="mt-7 max-w-3xl rounded-lg bg-white p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Editar cliente</h2>
        <Link href="/admin/clientes" className="text-sm font-semibold">
          Volver
        </Link>
      </div>
      <form action={updateCustomerAction} className="mt-5 grid gap-4">
        <input type="hidden" name="id" value={customer.id} />
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="grid gap-2 text-sm font-medium">
            Nombre
            <input
              name="firstName"
              defaultValue={customer.firstName}
              className="h-11 rounded-md border border-zinc-200 px-3"
              required
            />
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Apellidos
            <input
              name="lastName"
              defaultValue={customer.lastName ?? ""}
              className="h-11 rounded-md border border-zinc-200 px-3"
            />
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Telefono
            <input
              name="phone"
              defaultValue={customer.phone ?? ""}
              className="h-11 rounded-md border border-zinc-200 px-3"
            />
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Email
            <input
              name="email"
              type="email"
              defaultValue={customer.email ?? ""}
              className="h-11 rounded-md border border-zinc-200 px-3"
            />
          </label>
        </div>
        <label className="grid gap-2 text-sm font-medium">
          Contacto preferido
          <select
            name="preferredContactMethod"
            defaultValue={customer.preferredContactMethod ?? ""}
            className="h-11 rounded-md border border-zinc-200 px-3"
          >
            <option value="">Sin preferencia</option>
            <option value="phone">Telefono</option>
            <option value="whatsapp">WhatsApp</option>
            <option value="email">Email</option>
          </select>
        </label>
        <label className="grid gap-2 text-sm font-medium">
          Notas
          <textarea
            name="notes"
            defaultValue={customer.notes ?? ""}
            className="min-h-28 rounded-md border border-zinc-200 p-3"
          />
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            name="marketingOptIn"
            type="checkbox"
            defaultChecked={customer.marketingOptIn}
          />
          Acepta promociones
        </label>
        <button className="h-11 rounded-full bg-black px-5 text-sm font-semibold text-white">
          Guardar cambios
        </button>
      </form>
    </section>
  );
}
