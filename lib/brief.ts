// lib/brief.ts — the trial brief: three JSON documents embedded in prose
// (provider directory, quality table, workload manifest). Parsed with a
// brace-matching scan anchored on each document's schema marker, so prose
// around the documents never matters.

import { isRecord, numOrNull, strOrNull } from './util.ts';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export interface ProviderDir {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  key_env: string;
  key: string;
  /** The provider's own path, e.g. "/p/acme" — never joined to the env URL in the catalog. */
  path: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirements {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface ManifestReq {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirements;
}

export interface Brief {
  providers: ProviderDir[];
  ratings: RatingRow[];
  best: Map<string, number>;
  manifest: ManifestReq[];
}

const EMPTY_BRIEF: Brief = { providers: [], ratings: [], best: new Map(), manifest: [] };

/**
 * Extract the JSON object whose text contains `schemaMarker`. The document
 * opens at the `{` nearest before the marker; braces are matched with string
 * awareness so braces inside strings never confuse the scan.
 */
function extractDocument(text: string, schemaMarker: string): Record<string, unknown> | null {
  const at = text.indexOf(schemaMarker);
  if (at === -1) return null;
  const start = text.lastIndexOf('{', at);
  if (start === -1) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) {
        try {
          const v: unknown = JSON.parse(text.slice(start, i + 1));
          return isRecord(v) ? v : null;
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

function wireOf(v: unknown): Wire | null {
  return v === 'openai' || v === 'anthropic' || v === 'gemini' ? v : null;
}

function authOf(v: unknown): AuthScheme | null {
  return v === 'bearer' || v === 'x-api-key' || v === 'x-goog-api-key' || v === 'none' ? v : null;
}

function boolOf(v: unknown): boolean {
  return v === true;
}

export function parseBrief(text: string): Brief {
  const providersDoc = extractDocument(text, 'free-model-aggregator-01/providers/v1');
  const ratingsDoc = extractDocument(text, 'free-model-aggregator-01/ratings/v1');
  const workloadDoc = extractDocument(text, 'free-model-aggregator-01/workload/v1');
  if (providersDoc === null || ratingsDoc === null || workloadDoc === null) return EMPTY_BRIEF;

  const providers: ProviderDir[] = [];
  if (Array.isArray(providersDoc.providers)) {
    for (const raw of providersDoc.providers) {
      if (!isRecord(raw)) continue;
      const wire = wireOf(raw.wire);
      const auth = authOf(raw.auth);
      const id = strOrNull(raw.id);
      const path = strOrNull(raw.path);
      if (wire === null || auth === null || id === null || path === null) continue;
      providers.push({
        id,
        name: strOrNull(raw.name) ?? id,
        wire,
        auth,
        key_env: strOrNull(raw.key_env) ?? '',
        key: strOrNull(raw.key) ?? '',
        path,
      });
    }
  }

  const ratings: RatingRow[] = [];
  if (Array.isArray(ratingsDoc.ratings)) {
    for (const raw of ratingsDoc.ratings) {
      if (!isRecord(raw)) continue;
      const model = strOrNull(raw.model);
      const provider = strOrNull(raw.provider);
      const rating = numOrNull(raw.rating_milli);
      if (model === null || provider === null || rating === null) continue;
      ratings.push({
        model,
        provider,
        rating_milli: rating,
        context_length: raw.context_length === null ? null : numOrNull(raw.context_length),
        tools: boolOf(raw.tools),
        vision: boolOf(raw.vision),
        json: boolOf(raw.json),
        free: boolOf(raw.free),
      });
    }
  }

  const best = new Map<string, number>();
  if (Array.isArray(ratingsDoc.best)) {
    for (const raw of ratingsDoc.best) {
      if (!isRecord(raw)) continue;
      const id = strOrNull(raw.id);
      const rating = numOrNull(raw.rating_milli);
      if (id !== null && rating !== null) best.set(id, rating);
    }
  }

  const manifest: ManifestReq[] = [];
  if (Array.isArray(workloadDoc.manifest)) {
    for (const raw of workloadDoc.manifest) {
      if (!isRecord(raw)) continue;
      const id = strOrNull(raw.id);
      const api = raw.api === 'openai' || raw.api === 'anthropic' ? raw.api : null;
      if (id === null || api === null) continue;
      const requires: Requirements = {};
      if (isRecord(raw.requires)) {
        if (raw.requires.tools !== undefined) requires.tools = boolOf(raw.requires.tools);
        if (raw.requires.vision !== undefined) requires.vision = boolOf(raw.requires.vision);
        if (raw.requires.json !== undefined) requires.json = boolOf(raw.requires.json);
        const mc = numOrNull(raw.requires.min_context);
        if (mc !== null) requires.min_context = mc;
      }
      manifest.push({
        id,
        api,
        stream: raw.stream === true,
        weight: numOrNull(raw.weight) ?? 1,
        requires,
      });
    }
  }

  return { providers, ratings, best, manifest };
}

/** A model row meets a request's requirements when every demanded capability holds. */
export function meetsRequirements(row: RatingRow, req: Requirements): boolean {
  if (req.tools === true && !row.tools) return false;
  if (req.vision === true && !row.vision) return false;
  if (req.json === true && !row.json) return false;
  if (req.min_context !== undefined && (row.context_length === null || row.context_length < req.min_context)) return false;
  return true;
}
