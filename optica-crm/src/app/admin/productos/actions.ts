"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { prisma } from "@/lib/db/prisma";
import { requireSession } from "@/lib/auth/session";
import { formNumber, formString } from "@/lib/form";
import { slugify } from "@/lib/slug";
import { productSchema } from "@/lib/validations/product";
import { ProductType } from "@/generated/prisma/client";

async function uniqueProductSlug(organizationId: string, name: string) {
  const base = slugify(name) || "producto";
  let candidate = base;
  let suffix = 1;

  while (
    await prisma.product.findUnique({
      where: {
        organizationId_slug: {
          organizationId,
          slug: candidate,
        },
      },
      select: { id: true },
    })
  ) {
    suffix += 1;
    candidate = `${base}-${suffix}`;
  }

  return candidate;
}

function productPayload(formData: FormData) {
  return productSchema.parse({
    name: formString(formData, "name"),
    sku: formString(formData, "sku"),
    categoryId: formString(formData, "categoryId"),
    brandId: formString(formData, "brandId"),
    productType: formString(formData, "productType"),
    salePrice: formNumber(formData, "salePrice"),
    costPrice: formNumber(formData, "costPrice"),
    reorderPoint: formNumber(formData, "reorderPoint"),
    initialStock: formNumber(formData, "initialStock"),
    trackInventory: formData.get("trackInventory") === "on",
    isPublic: formData.get("isPublic") === "on",
  });
}

export async function createProductAction(formData: FormData) {
  const session = await requireSession();
  const data = productPayload(formData);
  const branch = await prisma.branch.findFirst({
    where: {
      organizationId: session.organizationId,
      isActive: true,
    },
    orderBy: { createdAt: "asc" },
  });

  const product = await prisma.product.create({
    data: {
      organizationId: session.organizationId,
      categoryId: data.categoryId,
      brandId: data.brandId,
      sku: data.sku,
      name: data.name,
      slug: await uniqueProductSlug(session.organizationId, data.name),
      productType: data.productType as ProductType,
      salePrice: data.salePrice,
      costPrice: data.costPrice,
      trackInventory: data.trackInventory,
      reorderPoint: data.trackInventory ? data.reorderPoint ?? 0 : null,
      isPublic: data.isPublic,
    },
  });

  if (data.trackInventory && branch) {
    const initialStock = data.initialStock ?? 0;

    await prisma.inventoryStock.create({
      data: {
        organizationId: session.organizationId,
        branchId: branch.id,
        productId: product.id,
        quantityOnHand: initialStock,
        quantityAvailable: initialStock,
      },
    });

    if (initialStock > 0) {
      await prisma.inventoryMovement.create({
        data: {
          organizationId: session.organizationId,
          branchId: branch.id,
          productId: product.id,
          movementType: "purchase",
          quantity: initialStock,
          reason: "Stock inicial",
          referenceType: "manual",
          createdByUserId: session.userId,
        },
      });
    }
  }

  revalidatePath("/admin/productos");
  revalidatePath("/admin/inventario");
  redirect("/admin/productos?created=1");
}

export async function updateProductAction(formData: FormData) {
  const session = await requireSession();
  const id = String(formData.get("id") ?? "");
  const data = productPayload(formData);

  await prisma.product.update({
    where: {
      id,
      organizationId: session.organizationId,
    },
    data: {
      categoryId: data.categoryId,
      brandId: data.brandId,
      sku: data.sku,
      name: data.name,
      productType: data.productType as ProductType,
      salePrice: data.salePrice,
      costPrice: data.costPrice,
      trackInventory: data.trackInventory,
      reorderPoint: data.trackInventory ? data.reorderPoint ?? 0 : null,
      isPublic: data.isPublic,
    },
  });

  revalidatePath("/admin/productos");
  revalidatePath("/admin/inventario");
  redirect("/admin/productos?updated=1");
}

export async function deleteProductAction(formData: FormData) {
  const session = await requireSession();
  const id = String(formData.get("id") ?? "");

  await prisma.product.update({
    where: {
      id,
      organizationId: session.organizationId,
    },
    data: {
      isActive: false,
      isPublic: false,
      deletedAt: new Date(),
    },
  });

  revalidatePath("/admin/productos");
}
