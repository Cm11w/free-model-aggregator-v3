// lib/util.ts — small shared helpers. No clock, no randomness anywhere in this
// entry: every value that reaches a request or a response is derived from the
// brief, the client request, or a provider answer, so a replayed trial behaves
// byte for byte like the live one.

export function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

export function asRecord(v: unknown): Record<string, unknown> | null {
  return isRecord(v) ? v : null;
}

export function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

export function strOrNull(v: unknown): string | null {
  return typeof v === 'string' ? v : null;
}

/** A string, or a list of strings, normalized to a list of strings. */
export function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/**
 * The trial's fixed "now", the only timestamp this entry may emit. Read from
 * MAKEFUN_TRIAL_NOW (RFC 3339 or epoch milliseconds); 0 when absent or
 * unparseable. Never the wall clock.
 */
export function trialNowSeconds(): number {
  const raw = process.env.MAKEFUN_TRIAL_NOW;
  if (raw === undefined || raw === '') return 0;
  const asNumber = Number(raw);
  if (Number.isFinite(asNumber) && raw.trim() !== '') {
    // Epoch milliseconds when large, epoch seconds when small.
    return Math.floor(asNumber > 1e12 ? asNumber / 1000 : asNumber);
  }
  const ms = Date.parse(raw);
  if (Number.isFinite(ms)) return Math.floor(ms / 1000);
  return 0;
}

/** Join a base URL and a path without producing a double slash. */
export function joinUrl(base: string, path: string): string {
  const b = base.endsWith('/') ? base.slice(0, -1) : base;
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${b}${p}`;
}

/** Parse JSON leniently: returns null instead of throwing. */
export function tryJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/** Deep-clone a JSON value (objects/arrays/primitives only). */
export function cloneJson<T>(v: T): T {
  if (v === null || typeof v !== 'object') return v;
  if (Array.isArray(v)) return v.map(cloneJson) as T;
  const out: Record<string, unknown> = {};
  for (const [k, val] of Object.entries(v as Record<string, unknown>)) out[k] = cloneJson(val);
  return out as T;
}
