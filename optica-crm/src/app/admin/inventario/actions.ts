"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { InventoryMovementType } from "@/generated/prisma/client";
import { requireSession } from "@/lib/auth/session";
import { prisma } from "@/lib/db/prisma";
import { formNumber, formString } from "@/lib/form";

function inventoryErrorUrl(message: string) {
  return `/admin/inventario?error=${encodeURIComponent(message)}`;
}

export async function adjustInventoryAction(formData: FormData) {
  const session = await requireSession();
  const productId = formString(formData, "productId");
  const branchId = formString(formData, "branchId");
  const quantity = formNumber(formData, "quantity");
  const movementType = formString(formData, "movementType") as
    | InventoryMovementType
    | undefined;
  const reason = formString(formData, "reason");

  if (!productId || !branchId || !quantity || !movementType) {
    redirect(inventoryErrorUrl("Completa producto, sucursal, tipo y cantidad."));
  }

  if (!reason) {
    redirect(inventoryErrorUrl("El motivo del movimiento es obligatorio."));
  }

  try {
    await prisma.$transaction(async (tx) => {
      const product = await tx.product.findFirst({
        where: {
          id: productId,
          organizationId: session.organizationId,
          deletedAt: null,
        },
      });

      if (!product || !product.trackInventory) {
        throw new Error("Producto invalido para inventario.");
      }

      const currentStock = await tx.inventoryStock.findUnique({
        where: {
          branchId_productId: {
            branchId,
            productId,
          },
        },
      });

      const currentOnHand = currentStock?.quantityOnHand ?? 0;
      const reserved = currentStock?.quantityReserved ?? 0;
      const nextOnHand = currentOnHand + quantity;
      const nextAvailable = nextOnHand - reserved;

      if (nextOnHand < 0 || nextAvailable < 0) {
        throw new Error("El movimiento dejaria inventario negativo.");
      }

      if (currentStock) {
        await tx.inventoryStock.update({
          where: { id: currentStock.id },
          data: {
            quantityOnHand: nextOnHand,
            quantityAvailable: nextAvailable,
          },
        });
      } else {
        await tx.inventoryStock.create({
          data: {
            organizationId: session.organizationId,
            branchId,
            productId,
            quantityOnHand: nextOnHand,
            quantityAvailable: nextAvailable,
          },
        });
      }

      await tx.inventoryMovement.create({
        data: {
          organizationId: session.organizationId,
          branchId,
          productId,
          movementType,
          quantity,
          reason,
          referenceType: "manual",
          createdByUserId: session.userId,
        },
      });
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "No se pudo registrar el movimiento.";
    redirect(inventoryErrorUrl(message));
  }

  revalidatePath("/admin/inventario");
  revalidatePath("/admin");
  redirect("/admin/inventario?updated=1");
}
