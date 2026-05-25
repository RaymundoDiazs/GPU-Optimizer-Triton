import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import {
  createSessionPayload,
  SESSION_COOKIE_NAME,
  SESSION_MAX_AGE_SECONDS,
  signSessionPayload,
  verifySessionToken,
  type SessionPayload,
} from "./session-core";

export async function getCurrentSession() {
  const cookieStore = await cookies();
  return verifySessionToken(cookieStore.get(SESSION_COOKIE_NAME)?.value);
}

export async function setSessionCookie(user: Omit<SessionPayload, "exp">) {
  const cookieStore = await cookies();
  const token = await signSessionPayload(createSessionPayload(user));

  cookieStore.set(SESSION_COOKIE_NAME, token, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    maxAge: SESSION_MAX_AGE_SECONDS,
    path: "/",
  });
}

export async function clearSessionCookie() {
  const cookieStore = await cookies();
  cookieStore.delete(SESSION_COOKIE_NAME);
}

export async function requireSession() {
  const session = await getCurrentSession();

  if (!session) {
    redirect("/login?next=/admin");
  }

  return session;
}
