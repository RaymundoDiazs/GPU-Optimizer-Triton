import {
  ArrowUpRight,
  Boxes,
  CalendarDays,
  CircleDollarSign,
  Users,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { prisma } from "@/lib/db/prisma";
import { requireSession } from "@/lib/auth/session";
import { formatCurrency, formatTime } from "@/lib/format";

export const dynamic = "force-dynamic";

type Stat = {
  label: string;
  value: string;
  detail: string;
  icon: LucideIcon;
};

async function getDashboardData(organizationId: string) {
  const startOfDay = new Date();
  startOfDay.setHours(0, 0, 0, 0);

  try {
    const [
      salesToday,
      customersCount,
      lowInventory,
      appointmentsToday,
      upcomingAppointments,
      inventoryItems,
    ] = await Promise.all([
      prisma.sale.aggregate({
        where: {
          organizationId,
          createdAt: { gte: startOfDay },
        },
        _sum: { total: true },
        _count: true,
      }),
      prisma.customer.count({
        where: {
          organizationId,
          deletedAt: null,
        },
      }),
      prisma.inventoryStock.count({
        where: {
          organizationId,
          product: {
            reorderPoint: { not: null },
          },
          quantityAvailable: { lte: 5 },
        },
      }),
      prisma.appointment.count({
        where: {
          organizationId,
          scheduledStart: { gte: startOfDay },
        },
      }),
      prisma.appointment.findMany({
        where: {
          organizationId,
          scheduledStart: { gte: startOfDay },
        },
        include: {
          customer: true,
        },
        orderBy: { scheduledStart: "asc" },
        take: 3,
      }),
      prisma.inventoryStock.findMany({
        where: { organizationId },
        include: {
          branch: true,
          product: true,
        },
        orderBy: { updatedAt: "desc" },
        take: 3,
      }),
    ]);

    return {
      stats: [
        {
          label: "Ventas hoy",
          value: formatCurrency(salesToday._sum.total?.toString()),
          detail: `${salesToday._count} operaciones`,
          icon: CircleDollarSign,
        },
        {
          label: "Inventario bajo",
          value: String(lowInventory),
          detail: "SKU por reordenar",
          icon: Boxes,
        },
        {
          label: "Citas",
          value: String(appointmentsToday),
          detail: "programadas desde hoy",
          icon: CalendarDays,
        },
        {
          label: "Clientes activos",
          value: customersCount.toLocaleString("es-MX"),
          detail: "registros vigentes",
          icon: Users,
        },
      ] satisfies Stat[],
      appointments: upcomingAppointments.map((appointment) => [
        formatTime(appointment.scheduledStart),
        `${appointment.customer.firstName} ${appointment.customer.lastName ?? ""}`.trim(),
        appointment.reason ?? "Consulta",
        appointment.status,
      ]),
      inventory: inventoryItems.map((stock) => [
        stock.product.sku ?? "SERVICIO",
        stock.product.name,
        String(stock.quantityAvailable),
        stock.branch.name,
      ]),
      connected: true,
    };
  } catch {
    return {
      stats: [
        {
          label: "Ventas hoy",
          value: "$0",
          detail: "conecta PostgreSQL",
          icon: CircleDollarSign,
        },
        {
          label: "Inventario bajo",
          value: "0",
          detail: "sin lectura de DB",
          icon: Boxes,
        },
        {
          label: "Citas",
          value: "0",
          detail: "sin lectura de DB",
          icon: CalendarDays,
        },
        {
          label: "Clientes activos",
          value: "0",
          detail: "sin lectura de DB",
          icon: Users,
        },
      ] satisfies Stat[],
      appointments: [],
      inventory: [],
      connected: false,
    };
  }
}

export default async function AdminPage() {
  const session = await requireSession();
  const data = await getDashboardData(session.organizationId);

  return (
    <>
      {!data.connected ? (
        <div className="mt-7 rounded-lg border border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-800">
          El CRM ya esta conectado por codigo, pero falta configurar una base PostgreSQL
          real en <code>DATABASE_URL</code>, ejecutar migraciones y correr el seed.
        </div>
      ) : null}

      <section className="mt-7 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {data.stats.map((stat) => (
          <article key={stat.label} className="rounded-lg bg-white p-5">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-zinc-500">{stat.label}</span>
              <span className="grid size-10 place-items-center rounded-full bg-[#edf7f6]">
                <stat.icon size={19} />
              </span>
            </div>
            <p className="mt-5 text-3xl font-semibold">{stat.value}</p>
            <p className="mt-1 text-sm text-zinc-500">{stat.detail}</p>
          </article>
        ))}
      </section>

      <section className="mt-7 grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
        <article className="rounded-lg bg-white p-5">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Agenda de hoy</h2>
            <button className="grid size-9 place-items-center rounded-full bg-zinc-100">
              <ArrowUpRight size={17} />
            </button>
          </div>
          <div className="mt-5 overflow-x-auto rounded-lg border border-zinc-100">
            <div className="min-w-[620px]">
              {data.appointments.length ? (
                data.appointments.map(([time, customer, reason, status]) => (
                  <div
                    key={`${time}-${customer}`}
                    className="grid grid-cols-[80px_1fr_1fr_120px] gap-4 border-b border-zinc-100 px-4 py-4 text-sm last:border-b-0"
                  >
                    <span className="font-semibold">{time}</span>
                    <span>{customer}</span>
                    <span className="text-zinc-500">{reason}</span>
                    <span className="text-right font-medium">{status}</span>
                  </div>
                ))
              ) : (
                <p className="px-4 py-4 text-sm text-zinc-500">
                  No hay citas registradas para hoy.
                </p>
              )}
            </div>
          </div>
        </article>

        <article className="rounded-lg bg-black p-5 text-white">
          <p className="text-sm uppercase text-white/60">Flujo comercial</p>
          <h2 className="mt-3 text-3xl font-semibold leading-tight">
            Cotizacion, receta, apartado y entrega en una sola vista.
          </h2>
          <div className="mt-8 grid grid-cols-2 gap-3 text-sm">
            <div className="rounded-lg bg-white/10 p-4">
              <p className="text-white/60">Pagado</p>
              <p className="mt-2 text-2xl font-semibold">76%</p>
            </div>
            <div className="rounded-lg bg-white/10 p-4">
              <p className="text-white/60">En proceso</p>
              <p className="mt-2 text-2xl font-semibold">14</p>
            </div>
          </div>
        </article>
      </section>

      <section className="mt-7 rounded-lg bg-white p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Inventario destacado</h2>
          <a
            href="/admin/inventario"
            className="inline-flex items-center gap-2 rounded-full border border-zinc-200 px-4 py-2 text-sm font-semibold"
          >
            Nuevo movimiento
            <Boxes size={16} />
          </a>
        </div>
        <div className="mt-5 overflow-x-auto rounded-lg border border-zinc-100">
          <div className="min-w-[620px]">
            {data.inventory.length ? (
              data.inventory.map(([sku, product, stock, branch]) => (
                <div
                  key={sku}
                  className="grid grid-cols-[120px_1fr_80px_100px] gap-4 border-b border-zinc-100 px-4 py-4 text-sm last:border-b-0"
                >
                  <span className="font-semibold">{sku}</span>
                  <span>{product}</span>
                  <span>{stock}</span>
                  <span className="text-zinc-500">{branch}</span>
                </div>
              ))
            ) : (
              <p className="px-4 py-4 text-sm text-zinc-500">
                No hay inventario disponible para mostrar.
              </p>
            )}
          </div>
        </div>
      </section>
    </>
  );
}
