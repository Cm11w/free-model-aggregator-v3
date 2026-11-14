// lib/router.ts — candidate ranking and failover. Candidates come from the
// brief's quality table, filtered by the request's requirements and by what
// each provider's wire can carry, and ranked by rating (ties broken by
// provider id, then model id, so the order is a pure function of the brief).
//
// Failover reads what the provider said: a permanent refusal ("budget is
// spent", "is out") retires the provider for the trial; a transient refusal
// ("refused this call", a concurrency clash, a network error) is retried on
// the same candidate a bounded number of times before moving on, because the
// failure schedule refuses call numbers, not providers. An auth failure
// retires the provider; a bad-request failure retires the pair.

import type { Brief, ProviderDir, RatingRow } from './brief.ts';
import { meetsRequirements } from './brief.ts';
import type { CanonicalRequest } from './canonical.ts';
import { requirementsOf } from './canonical.ts';

export interface Candidate {
  provider: ProviderDir;
  row: RatingRow;
}

/** Rank the candidates for one request. Pure: same inputs, same order. */
export function rankCandidates(
  brief: Brief,
  freeByPair: Map<string, boolean>,
  req: CanonicalRequest,
): Candidate[] {
  const need = requirementsOf(req);
  const requires = {
    tools: need.tools,
    vision: need.vision,
    json: need.json,
    min_context: undefined as number | undefined,
  };
  const byId = new Map(brief.providers.map((p) => [p.id, p]));
  const out: Candidate[] = [];
  for (const row of brief.ratings) {
    if (!row.free) continue;
    const provider = byId.get(row.provider);
    if (provider === undefined) continue;
    // The catalog probe is the truth about free flags; when a probe ran, a
    // model it marked paid is never picked, whatever the brief says.
    const probed = freeByPair.get(`${row.provider}/${row.model}`);
    if (probed === false) continue;
    if (!meetsRequirements(row, requires)) continue;
    // JSON mode can only travel on wires that express it. The Anthropic
    // wire echoes json_mode as absent, so it must never carry such a request.
    if (need.json && provider.wire === 'anthropic') continue;
    out.push({ provider, row });
  }
  out.sort((a, b) => {
    if (b.row.rating_milli !== a.row.rating_milli) return b.row.rating_milli - a.row.rating_milli;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.row.model < b.row.model ? -1 : a.row.model > b.row.model ? 1 : 0;
  });
  return out;
}

/**
 * Trial-long failover memory. Providers retire only on permanent refusals
 * or auth failures; transient refusals never retire anything, because a
 * failure schedule is a list of call numbers, not a dead provider.
 */
export class RouterState {
  private dead = new Set<string>();
  private badPairs = new Set<string>();

  isDead(providerId: string): boolean {
    return this.dead.has(providerId);
  }

  isBadPair(providerId: string, model: string): boolean {
    return this.badPairs.has(`${providerId}/${model}`);
  }

  retireProvider(providerId: string): void {
    this.dead.add(providerId);
  }

  retirePair(providerId: string, model: string): void {
    this.badPairs.add(`${providerId}/${model}`);
  }

  /** The candidates still worth trying, in rank order. */
  usable(candidates: Candidate[]): Candidate[] {
    return candidates.filter((c) => !this.dead.has(c.provider.id) && !this.badPairs.has(`${c.provider.id}/${c.row.model}`));
  }
}

/** Attempts allowed on one candidate while its refusals stay transient. */
export const MAX_TRANSIENT_ATTEMPTS = 3;
