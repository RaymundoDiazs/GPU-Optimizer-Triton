import { z } from "zod";

export const customerSchema = z
  .object({
    firstName: z.string().min(2),
    lastName: z.string().optional(),
    phone: z.string().optional(),
    email: z.email().optional(),
    preferredContactMethod: z.enum(["phone", "whatsapp", "email"]).optional(),
    marketingOptIn: z.boolean().default(false),
    notes: z.string().optional(),
  })
  .refine((value) => value.phone || value.email, {
    message: "El cliente necesita al menos telefono o email.",
    path: ["phone"],
  });

export type CustomerInput = z.infer<typeof customerSchema>;
