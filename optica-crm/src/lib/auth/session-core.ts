export const SESSION_COOKIE_NAME = "optica_session";
export const SESSION_MAX_AGE_SECONDS = 60 * 60 * 8;

export type SessionPayload = {
  userId: string;
  organizationId: string;
  email: string;
  name: string;
  roles: string[];
  exp: number;
};

const encoder = new TextEncoder();

function getSessionSecret() {
  const secret =
    process.env.AUTH_SECRET ??
    process.env.SESSION_SECRET ??
    "optica-crm-development-secret-change-me";

  if (
    process.env.NODE_ENV === "production" &&
    secret === "optica-crm-development-secret-change-me"
  ) {
    throw new Error("AUTH_SECRET is required in production.");
  }

  return secret;
}

function toBase64Url(bytes: Uint8Array) {
  const base64 =
    typeof Buffer !== "undefined"
      ? Buffer.from(bytes).toString("base64")
      : btoa(String.fromCharCode(...bytes));

  return base64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function fromBase64Url(value: string) {
  const base64 = value.replace(/-/g, "+").replace(/_/g, "/");
  const padded = base64.padEnd(base64.length + ((4 - (base64.length % 4)) % 4), "=");

  if (typeof Buffer !== "undefined") {
    return new Uint8Array(Buffer.from(padded, "base64"));
  }

  return Uint8Array.from(atob(padded), (character) => character.charCodeAt(0));
}

async function getSigningKey() {
  return crypto.subtle.importKey(
    "raw",
    encoder.encode(getSessionSecret()),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"],
  );
}

export async function signSessionPayload(payload: SessionPayload) {
  const encodedPayload = toBase64Url(encoder.encode(JSON.stringify(payload)));
  const key = await getSigningKey();
  const signature = await crypto.subtle.sign(
    "HMAC",
    key,
    encoder.encode(encodedPayload),
  );

  return `${encodedPayload}.${toBase64Url(new Uint8Array(signature))}`;
}

export async function verifySessionToken(token?: string) {
  if (!token) {
    return null;
  }

  const [encodedPayload, encodedSignature] = token.split(".");

  if (!encodedPayload || !encodedSignature) {
    return null;
  }

  const key = await getSigningKey();
  const isValid = await crypto.subtle.verify(
    "HMAC",
    key,
    fromBase64Url(encodedSignature),
    encoder.encode(encodedPayload),
  );

  if (!isValid) {
    return null;
  }

  const payload = JSON.parse(
    new TextDecoder().decode(fromBase64Url(encodedPayload)),
  ) as SessionPayload;

  if (!payload.exp || payload.exp < Math.floor(Date.now() / 1000)) {
    return null;
  }

  return payload;
}

export function createSessionPayload(
  user: Omit<SessionPayload, "exp">,
): SessionPayload {
  return {
    ...user,
    exp: Math.floor(Date.now() / 1000) + SESSION_MAX_AGE_SECONDS,
  };
}
