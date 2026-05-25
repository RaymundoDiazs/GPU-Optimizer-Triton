export function formString(formData: FormData, key: string) {
  const value = String(formData.get(key) ?? "").trim();
  return value.length ? value : undefined;
}

export function formNumber(formData: FormData, key: string) {
  const value = formString(formData, key);
  return value ? Number(value) : undefined;
}
