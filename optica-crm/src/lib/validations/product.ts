import { z } from "zod";

export const productSchema = z
  .object({
    name: z.string().min(2),
    sku: z.string().optional(),
    categoryId: z.uuid(),
    brandId: z.uuid().optional(),
    productType: z.enum([
      "frame",
      "sunglasses",
      "contact_lens",
      "lens",
      "treatment",
      "accessory",
      "service",
    ]),
    salePrice: z.number().nonnegative(),
    costPrice: z.number().nonnegative().optional(),
    reorderPoint: z.number().int().nonnegative().optional(),
    initialStock: z.number().int().nonnegative().optional(),
    trackInventory: z.boolean(),
    isPublic: z.boolean(),
  })
  .refine((value) => !value.trackInventory || Boolean(value.sku), {
    message: "Los productos con inventario necesitan SKU.",
    path: ["sku"],
  });

export type ProductInput = z.infer<typeof productSchema>;
