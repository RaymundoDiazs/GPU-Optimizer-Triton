import Link from "next/link";
import { notFound } from "next/navigation";
import { ProductType } from "@/generated/prisma/client";
import { requireSession } from "@/lib/auth/session";
import { prisma } from "@/lib/db/prisma";
import { updateProductAction } from "../actions";

export const dynamic = "force-dynamic";

type ProductEditPageProps = {
  params: Promise<{ id: string }>;
};

export default async function ProductEditPage({ params }: ProductEditPageProps) {
  const { id } = await params;
  const session = await requireSession();
  const [product, categories, brands] = await Promise.all([
    prisma.product.findFirst({
      where: {
        id,
        organizationId: session.organizationId,
        deletedAt: null,
      },
    }),
    prisma.category.findMany({
      where: {
        organizationId: session.organizationId,
        isActive: true,
      },
      orderBy: { name: "asc" },
    }),
    prisma.brand.findMany({
      where: {
        organizationId: session.organizationId,
        isActive: true,
      },
      orderBy: { name: "asc" },
    }),
  ]);

  if (!product) {
    notFound();
  }

  return (
    <section className="mt-7 max-w-4xl rounded-lg bg-white p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Editar producto</h2>
        <Link href="/admin/productos" className="text-sm font-semibold">
          Volver
        </Link>
      </div>
      <form action={updateProductAction} className="mt-5 grid gap-4">
        <input type="hidden" name="id" value={product.id} />
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="grid gap-2 text-sm font-medium">
            Nombre
            <input
              name="name"
              defaultValue={product.name}
              className="h-11 rounded-md border border-zinc-200 px-3"
              required
            />
          </label>
          <label className="grid gap-2 text-sm font-medium">
            SKU
            <input
              name="sku"
              defaultValue={product.sku ?? ""}
              className="h-11 rounded-md border border-zinc-200 px-3"
            />
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Tipo
            <select
              name="productType"
              defaultValue={product.productType}
              className="h-11 rounded-md border border-zinc-200 px-3"
            >
              {Object.values(ProductType).map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Categoria
            <select
              name="categoryId"
              defaultValue={product.categoryId}
              className="h-11 rounded-md border border-zinc-200 px-3"
              required
            >
              {categories.map((category) => (
                <option key={category.id} value={category.id}>
                  {category.name}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Marca
            <select
              name="brandId"
              defaultValue={product.brandId ?? ""}
              className="h-11 rounded-md border border-zinc-200 px-3"
            >
              <option value="">Sin marca</option>
              {brands.map((brand) => (
                <option key={brand.id} value={brand.id}>
                  {brand.name}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Precio venta
            <input
              name="salePrice"
              type="number"
              step="0.01"
              min="0"
              defaultValue={product.salePrice.toString()}
              className="h-11 rounded-md border border-zinc-200 px-3"
              required
            />
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Costo
            <input
              name="costPrice"
              type="number"
              step="0.01"
              min="0"
              defaultValue={product.costPrice?.toString() ?? ""}
              className="h-11 rounded-md border border-zinc-200 px-3"
            />
          </label>
          <label className="grid gap-2 text-sm font-medium">
            Punto reorden
            <input
              name="reorderPoint"
              type="number"
              min="0"
              defaultValue={product.reorderPoint ?? ""}
              className="h-11 rounded-md border border-zinc-200 px-3"
            />
          </label>
        </div>
        <input type="hidden" name="initialStock" value="0" />
        <label className="flex items-center gap-2 text-sm">
          <input
            name="trackInventory"
            type="checkbox"
            defaultChecked={product.trackInventory}
          />
          Controla inventario
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input name="isPublic" type="checkbox" defaultChecked={product.isPublic} />
          Visible en sitio publico
        </label>
        <button className="h-11 rounded-full bg-black px-5 text-sm font-semibold text-white">
          Guardar cambios
        </button>
      </form>
    </section>
  );
}
