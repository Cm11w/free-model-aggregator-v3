// lib/catalog.ts — the catalog document: one entry per provider, probed live
// from each provider's own catalog route, with the brief's quality table as
// fallback when a probe fails. The providers' own listings are the truth the
// catalog is scored against, so probes run before registration and their
// results are kept verbatim (free flags and context lengths included).

import type { Brief, ProviderDir, Wire } from './brief.ts';
import { isRecord, joinUrl, numOrNull, strOrNull } from './util.ts';
import { authHeaders } from './upstream.ts';

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

export interface CatalogBundle {
  /** The catalog object, as registered. */
  catalog: Catalog;
  /** Its one serialization, reused for --list so the bytes always match. */
  json: string;
  /** The free model id set, first-seen order, for GET /v1/models. */
  freeIds: string[];
  /** Free flags as the probes reported them, keyed "provider/model". */
  freeByPair: Map<string, boolean>;
}

function catalogPath(wire: Wire): string {
  return wire === 'gemini' ? '/v1beta/models' : '/v1/models';
}

/** Parse one provider's listing into model rows, whatever its wire. */
export function parseListing(wire: Wire, body: unknown): CatalogModel[] | null {
  if (!isRecord(body)) return null;
  const rows: CatalogModel[] = [];
  if (wire === 'gemini') {
    if (!Array.isArray(body.models)) return null;
    for (const raw of body.models) {
      if (!isRecord(raw)) continue;
      const name = strOrNull(raw.name);
      if (name === null) continue;
      const id = name.startsWith('models/') ? name.slice('models/'.length) : name;
      if (id === '') continue;
      rows.push({
        id,
        free: raw.free === true,
        context_length: raw.context_length === null ? null : numOrNull(raw.context_length) ?? numOrNull(raw.inputTokenLimit),
      });
    }
    return rows;
  }
  if (!Array.isArray(body.data)) return null;
  for (const raw of body.data) {
    if (!isRecord(raw)) continue;
    const id = strOrNull(raw.id);
    if (id === null || id === '') continue;
    rows.push({
      id,
      free: raw.free === true,
      context_length: raw.context_length === null ? null : numOrNull(raw.context_length),
    });
  }
  return rows;
}

async function probeProvider(envUrl: string, provider: ProviderDir, signal: AbortSignal): Promise<CatalogModel[] | null> {
  const url = joinUrl(joinUrl(envUrl, provider.path), catalogPath(provider.wire));
  const headers: Record<string, string> = { ...authHeaders(provider) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let res: Response;
  try {
    res = await fetch(url, { method: 'GET', headers, signal });
  } catch {
    return null;
  }
  if (res.status < 200 || res.status >= 300) {
    try {
      await res.arrayBuffer();
    } catch {
      // Ignore body drain errors.
    }
    return null;
  }
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    return null;
  }
  return parseListing(provider.wire, body);
}

/**
 * Build the catalog. Providers are probed one at a time, in brief order, so
 * the result never depends on which probe answered first. A failed probe
 * falls back to the brief's rows for that provider (free models only — the
 * paid set is then absent rather than wrong, which the catalog check
 * tolerates: it fails only a paid model marked free).
 */
export async function buildCatalog(envUrl: string, brief: Brief): Promise<CatalogBundle> {
  const providers: CatalogProvider[] = [];
  const freeByPair = new Map<string, boolean>();
  for (const provider of brief.providers) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10000);
    let models: CatalogModel[] | null = null;
    try {
      models = await probeProvider(envUrl, provider, controller.signal);
    } finally {
      clearTimeout(timer);
    }
    if (models === null) {
      models = brief.ratings
        .filter((r) => r.provider === provider.id)
        .map((r) => ({ id: r.model, free: r.free, context_length: r.context_length }));
    }
    // De-duplicate by id, keeping the first listing's row.
    const seen = new Set<string>();
    const deduped = models.filter((m) => {
      if (seen.has(m.id)) return false;
      seen.add(m.id);
      return true;
    });
    for (const m of deduped) freeByPair.set(`${provider.id}/${m.id}`, m.free);
    providers.push({
      id: provider.id,
      base_url: provider.path,
      key_env: provider.key_env,
      wire: provider.wire,
      models: deduped,
    });
  }
  const catalog: Catalog = { providers };
  const json = JSON.stringify(catalog);
  const freeIds: string[] = [];
  const idSeen = new Set<string>();
  for (const p of providers) {
    for (const m of p.models) {
      if (m.free && !idSeen.has(m.id)) {
        idSeen.add(m.id);
        freeIds.push(m.id);
      }
    }
  }
  return { catalog, json, freeIds, freeByPair };
}

/** The empty catalog, printed by --list outside a trial. */
export function emptyCatalogBundle(): CatalogBundle {
  return { catalog: { providers: [] }, json: '{"providers":[]}', freeIds: [], freeByPair: new Map() };
}
