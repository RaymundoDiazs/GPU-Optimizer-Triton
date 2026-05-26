"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { prisma } from "@/lib/db/prisma";
import { requireSession } from "@/lib/auth/session";
import { customerSchema } from "@/lib/validations/customer";
import { formString } from "@/lib/form";

function customerPayload(formData: FormData) {
  return customerSchema.parse({
    firstName: formString(formData, "firstName"),
    lastName: formString(formData, "lastName"),
    phone: formString(formData, "phone"),
    email: formString(formData, "email"),
    preferredContactMethod: formString(formData, "preferredContactMethod"),
    marketingOptIn: formData.get("marketingOptIn") === "on",
    notes: formString(formData, "notes"),
  });
}

export async function createCustomerAction(formData: FormData) {
  const session = await requireSession();
  const data = customerPayload(formData);

  await prisma.customer.create({
    data: {
      organizationId: session.organizationId,
      firstName: data.firstName,
      lastName: data.lastName,
      phone: data.phone,
      email: data.email,
      preferredContactMethod: data.preferredContactMethod,
      marketingOptIn: data.marketingOptIn,
      notes: data.notes,
    },
  });

  revalidatePath("/admin/clientes");
  redirect("/admin/clientes?created=1");
}

export async function updateCustomerAction(formData: FormData) {
  const session = await requireSession();
  const id = String(formData.get("id") ?? "");
  const data = customerPayload(formData);

  await prisma.customer.update({
    where: {
      id,
      organizationId: session.organizationId,
    },
    data: {
      firstName: data.firstName,
      lastName: data.lastName,
      phone: data.phone,
      email: data.email,
      preferredContactMethod: data.preferredContactMethod,
      marketingOptIn: data.marketingOptIn,
      notes: data.notes,
    },
  });

  revalidatePath("/admin/clientes");
  redirect("/admin/clientes?updated=1");
}

export async function deleteCustomerAction(formData: FormData) {
  const session = await requireSession();
  const id = String(formData.get("id") ?? "");

  await prisma.customer.update({
    where: {
      id,
      organizationId: session.organizationId,
    },
    data: {
      deletedAt: new Date(),
    },
  });

  revalidatePath("/admin/clientes");
}
