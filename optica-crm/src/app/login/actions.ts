"use server";

import { redirect } from "next/navigation";
import { prisma } from "@/lib/db/prisma";
import { setSessionCookie, clearSessionCookie } from "@/lib/auth/session";
import { verifyPassword } from "@/lib/auth/password";
import { UserStatus } from "@/generated/prisma/client";

function safeNextPath(value: FormDataEntryValue | null) {
  const next = String(value ?? "/admin");
  return next.startsWith("/admin") ? next : "/admin";
}

function loginErrorUrl(next: string, message: string) {
  const params = new URLSearchParams({
    next,
    error: message,
  });

  return `/login?${params.toString()}`;
}

export async function loginAction(formData: FormData) {
  const email = String(formData.get("email") ?? "").trim().toLowerCase();
  const password = String(formData.get("password") ?? "");
  const next = safeNextPath(formData.get("next"));

  if (!email || !password) {
    redirect(loginErrorUrl(next, "Ingresa correo y contrasena."));
  }

  const user = await prisma.user.findFirst({
    where: {
      email,
      status: UserStatus.active,
    },
    include: {
      userRoles: {
        include: {
          role: true,
        },
      },
    },
  });

  const isValid = await verifyPassword(password, user?.passwordHash ?? null);

  if (!user || !isValid) {
    redirect(loginErrorUrl(next, "Credenciales incorrectas."));
  }

  await prisma.user.update({
    where: { id: user.id },
    data: { lastLoginAt: new Date() },
  });

  await setSessionCookie({
    userId: user.id,
    organizationId: user.organizationId,
    email: user.email,
    name: user.name,
    roles: user.userRoles.map((userRole) => userRole.role.key),
  });

  redirect(next);
}

export async function logoutAction() {
  await clearSessionCookie();
  redirect("/login");
}
