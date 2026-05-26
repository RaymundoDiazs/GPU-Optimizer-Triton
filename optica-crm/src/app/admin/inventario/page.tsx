import { Boxes, History, MoveRight } from "lucide-react";
import { InventoryMovementType } from "@/generated/prisma/client";
import { requireSession } from "@/lib/auth/session";
import { prisma } from "@/lib/db/prisma";
import { formatDate } from "@/lib/format";
import { adjustInventoryAction } from "./actions";

export const dynamic = "force-dynamic";

type InventoryPageProps = {
  searchParams: Promise<{
    error?: string;
    updated?: string;
  }>;
};

async function getInventoryData(organizationId: string) {
  try {
    const [stock, products, branches, movements] = await Promise.all([
      prisma.inventoryStock.findMany({
        where: {
          organizationId,
          product: {
            deletedAt: null,
          },
        },
        include: {
          branch: true,
          product: true,
        },
        orderBy: { updatedAt: "desc" },
        take: 80,
      }),
      prisma.product.findMany({
        where: {
          organizationId,
          deletedAt: null,
          isActive: true,
          trackInventory: true,
        },
        orderBy: { name: "asc" },
      }),
      prisma.branch.findMany({
        where: {
          organizationId,
          isActive: true,
        },
        orderBy: { name: "asc" },
      }),
      prisma.inventoryMovement.findMany({
        where: { organizationId },
        include: {
          branch: true,
          product: true,
        },
        orderBy: { createdAt: "desc" },
        take: 8,
      }),
    ]);

    return { stock, products, branches, movements, connected: true };
  } catch {
    return { stock: [], products: [], branches: [], movements: [], connected: false };
  }
}

export default async function InventoryPage({ searchParams }: InventoryPageProps) {
  const params = await searchParams;
  const session = await requireSession();
  const { stock, products, branches, movements, connected } = await getInventoryData(
    session.organizationId,
  );

  return (
    <div className="mt-7 grid gap-6 xl:grid-cols-[380px_1fr]">
      <section className="rounded-lg bg-white p-5">
        <div className="flex items-center gap-2">
          <Boxes size={18} />
          <h2 className="text-lg font-semibold">Ajustar inventario</h2>
        </div>

        {params.error ? (
          <div className="mt-4 rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {params.error}
          </div>
        ) : null}

        {!connected ? (
          <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
            Configura PostgreSQL y ejecuta migraciones para registrar movimientos.
          </div>
        ) : null}

        <form action={adjustInventoryAction} className="mt-5 grid gap-4">
          <label className="grid gap-2 text-sm font-medium">
            Producto
            <select name="productId" className="h-11 rounded-md border border-zinc-200 px-3" required>
              <option value="">Selecciona producto</option>
              {products.map((product) => (
                <option key={product.id} value={product.id}>
                  {product.sku ?? "SKU"} - {product.name}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Sucursal
            <select name="branchId" className="h-11 rounded-md border border-zinc-200 px-3" required>
              <option value="">Selecciona sucursal</option>
              {branches.map((branch) => (
                <option key={branch.id} value={branch.id}>
                  {branch.name}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Tipo
            <select name="movementType" className="h-11 rounded-md border border-zinc-200 px-3">
              <option value={InventoryMovementType.adjustment}>Ajuste</option>
              <option value={InventoryMovementType.purchase}>Entrada compra</option>
              <option value={InventoryMovementType.return}>Devolucion</option>
              <option value={InventoryMovementType.transfer_in}>Transferencia entrada</option>
              <option value={InventoryMovementType.transfer_out}>Transferencia salida</option>
            </select>
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Cantidad
            <input
              name="quantity"
              type="number"
              className="h-11 rounded-md border border-zinc-200 px-3"
              placeholder="Usa negativo para salida"
              required
            />
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Motivo
            <textarea name="reason" className="min-h-24 rounded-md border border-zinc-200 p-3" required />
          </label>
          <button className="h-11 rounded-full bg-black px-5 text-sm font-semibold text-white">
            Registrar movimiento
          </button>
        </form>
      </section>

      <section className="grid gap-6">
        <article className="rounded-lg bg-white p-5">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Existencias</h2>
            <span className="text-sm text-zinc-500">{stock.length} registros</span>
          </div>
          <div className="mt-5 overflow-x-auto rounded-lg border border-zinc-100">
            <div className="min-w-[820px]">
              {stock.length ? (
                stock.map((item) => (
                  <div
                    key={item.id}
                    className="grid grid-cols-[140px_1fr_100px_100px_140px] gap-4 border-b border-zinc-100 px-4 py-4 text-sm last:border-b-0"
                  >
                    <span className="font-semibold">{item.product.sku ?? "SERVICIO"}</span>
                    <span>{item.product.name}</span>
                    <span>{item.quantityOnHand}</span>
                    <span>{item.quantityAvailable}</span>
                    <span className="text-zinc-500">{item.branch.name}</span>
                  </div>
                ))
              ) : (
                <p className="px-4 py-4 text-sm text-zinc-500">
                  No hay existencias registradas.
                </p>
              )}
            </div>
          </div>
        </article>

        <article className="rounded-lg bg-white p-5">
          <div className="flex items-center gap-2">
            <History size={18} />
            <h2 className="text-lg font-semibold">Ultimos movimientos</h2>
          </div>
          <div className="mt-5 grid gap-3">
            {movements.length ? (
              movements.map((movement) => (
                <div
                  key={movement.id}
                  className="flex flex-col justify-between gap-3 rounded-lg border border-zinc-100 p-4 text-sm md:flex-row md:items-center"
                >
                  <div>
                    <p className="font-semibold">{movement.product.name}</p>
                    <p className="text-zinc-500">
                      {movement.movementType} · {movement.reason ?? "Sin motivo"}
                    </p>
                  </div>
                  <div className="flex items-center gap-3 font-semibold">
                    <span>{formatDate(movement.createdAt)}</span>
                    <MoveRight size={16} />
                    <span>{movement.quantity}</span>
                    <span className="text-zinc-500">{movement.branch.name}</span>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-sm text-zinc-500">No hay movimientos recientes.</p>
            )}
          </div>
        </article>
      </section>
    </div>
  );
}
