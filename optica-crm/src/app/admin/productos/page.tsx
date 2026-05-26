import Link from "next/link";
import { Edit3, Plus, Trash2 } from "lucide-react";
import { ProductType } from "@/generated/prisma/client";
import { requireSession } from "@/lib/auth/session";
import { prisma } from "@/lib/db/prisma";
import { formatCurrency } from "@/lib/format";
import { createProductAction, deleteProductAction } from "./actions";

export const dynamic = "force-dynamic";

async function getCatalogData(organizationId: string) {
  try {
    const [products, categories, brands] = await Promise.all([
      prisma.product.findMany({
        where: {
          organizationId,
          deletedAt: null,
        },
        include: {
          category: true,
          brand: true,
          inventoryStock: true,
        },
        orderBy: { createdAt: "desc" },
        take: 50,
      }),
      prisma.category.findMany({
        where: {
          organizationId,
          isActive: true,
        },
        orderBy: { name: "asc" },
      }),
      prisma.brand.findMany({
        where: {
          organizationId,
          isActive: true,
        },
        orderBy: { name: "asc" },
      }),
    ]);

    return { products, categories, brands };
  } catch {
    return { products: [], categories: [], brands: [] };
  }
}

export default async function ProductsPage() {
  const session = await requireSession();
  const { products, categories, brands } = await getCatalogData(session.organizationId);

  return (
    <div className="mt-7 grid gap-6 xl:grid-cols-[380px_1fr]">
      <section className="rounded-lg bg-white p-5">
        <div className="flex items-center gap-2">
          <Plus size={18} />
          <h2 className="text-lg font-semibold">Nuevo producto</h2>
        </div>
        <form action={createProductAction} className="mt-5 grid gap-4">
          <label className="grid gap-2 text-sm font-medium">
            Nombre
            <input name="name" className="h-11 rounded-md border border-zinc-200 px-3" required />
          </label>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-1">
            <label className="grid gap-2 text-sm font-medium">
              SKU
              <input name="sku" className="h-11 rounded-md border border-zinc-200 px-3" />
            </label>
            <label className="grid gap-2 text-sm font-medium">
              Tipo
              <select name="productType" className="h-11 rounded-md border border-zinc-200 px-3">
                {Object.values(ProductType).map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <label className="grid gap-2 text-sm font-medium">
            Categoria
            <select name="categoryId" className="h-11 rounded-md border border-zinc-200 px-3" required>
              <option value="">Selecciona categoria</option>
              {categories.map((category) => (
                <option key={category.id} value={category.id}>
                  {category.name}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Marca
            <select name="brandId" className="h-11 rounded-md border border-zinc-200 px-3">
              <option value="">Sin marca</option>
              {brands.map((brand) => (
                <option key={brand.id} value={brand.id}>
                  {brand.name}
                </option>
              ))}
            </select>
          </label>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-1">
            <label className="grid gap-2 text-sm font-medium">
              Precio venta
              <input name="salePrice" type="number" step="0.01" min="0" className="h-11 rounded-md border border-zinc-200 px-3" required />
            </label>
            <label className="grid gap-2 text-sm font-medium">
              Costo
              <input name="costPrice" type="number" step="0.01" min="0" className="h-11 rounded-md border border-zinc-200 px-3" />
            </label>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-1">
            <label className="grid gap-2 text-sm font-medium">
              Stock inicial
              <input name="initialStock" type="number" min="0" className="h-11 rounded-md border border-zinc-200 px-3" />
            </label>
            <label className="grid gap-2 text-sm font-medium">
              Punto reorden
              <input name="reorderPoint" type="number" min="0" className="h-11 rounded-md border border-zinc-200 px-3" />
            </label>
          </div>
          <label className="flex items-center gap-2 text-sm">
            <input name="trackInventory" type="checkbox" defaultChecked />
            Controla inventario
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input name="isPublic" type="checkbox" defaultChecked />
            Visible en sitio publico
          </label>
          <button className="h-11 rounded-full bg-black px-5 text-sm font-semibold text-white">
            Guardar producto
          </button>
        </form>
      </section>

      <section className="rounded-lg bg-white p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Productos</h2>
          <span className="text-sm text-zinc-500">{products.length} registros</span>
        </div>
        <div className="mt-5 overflow-x-auto rounded-lg border border-zinc-100">
          <div className="min-w-[900px]">
            {products.length ? (
              products.map((product) => {
                const stock = product.inventoryStock.reduce(
                  (total, item) => total + item.quantityAvailable,
                  0,
                );

                return (
                  <div
                    key={product.id}
                    className="grid grid-cols-[130px_1fr_140px_120px_90px_160px] gap-4 border-b border-zinc-100 px-4 py-4 text-sm last:border-b-0"
                  >
                    <span className="font-semibold">{product.sku ?? "SERVICIO"}</span>
                    <div>
                      <p className="font-semibold">{product.name}</p>
                      <p className="text-zinc-500">{product.category.name}</p>
                    </div>
                    <span>{product.brand?.name ?? "Sin marca"}</span>
                    <span>{formatCurrency(product.salePrice.toString())}</span>
                    <span>{product.trackInventory ? stock : "N/A"}</span>
                    <div className="flex items-center gap-4">
                      <Link
                        href={`/admin/productos/${product.id}`}
                        className="inline-flex items-center gap-2 font-semibold"
                      >
                        Editar
                        <Edit3 size={15} />
                      </Link>
                      <form action={deleteProductAction}>
                        <input type="hidden" name="id" value={product.id} />
                        <button className="inline-flex items-center gap-2 text-red-600">
                          Baja
                          <Trash2 size={15} />
                        </button>
                      </form>
                    </div>
                  </div>
                );
              })
            ) : (
              <p className="px-4 py-4 text-sm text-zinc-500">
                No hay productos para mostrar.
              </p>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
