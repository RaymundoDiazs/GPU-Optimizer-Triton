export const rolePermissions = {
  admin: ["*"],
  seller: ["customers:read", "customers:write", "sales:write", "inventory:read"],
  optometrist: ["appointments:read", "prescriptions:write", "customers:read"],
  inventory_manager: ["products:write", "inventory:write", "suppliers:write"],
  hr: ["employees:write", "payroll:write"],
} as const satisfies Record<string, readonly string[]>;

export type RoleKey = keyof typeof rolePermissions;

export function can(role: RoleKey, permission: string) {
  const permissions: readonly string[] = rolePermissions[role];
  return permissions.includes("*") || permissions.includes(permission);
}
