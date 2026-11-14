// lib/serve.ts — serving one request: rank, call, fail over, render.
//
// The loop walks the ranked candidates. A provider refusal is read for
// what it is: permanent ("budget is spent", "is out") retires the provider,
// auth retires the provider, a bad request retires the pair, and a
// transient refusal is retried on the same candidate a bounded number of
// times, because a failure schedule refuses call numbers, not providers.
// An answer that carries no usage is kept as a fallback and only served
// when nothing better answers: the client is checked against the
// provider's counts, so usage-less answers are a last resort, never a
// first choice.

import type { Brief } from './brief.ts';
import type { CatalogBundle } from './catalog.ts';
import type { CanonicalRequest } from './canonical.ts';
import { toAnthropic, toGemini, toOpenAI } from './translate.ts';
import { callJson, callStream, ProviderError } from './upstream.ts';
import {
  answerFromJson,
  answerFromStream,
  emitAnthropicJson,
  emitAnthropicSse,
  emitOpenAIJson,
  emitOpenAISse,
  isUsable,
  passthroughWithRepair,
  type ParsedAnswer,
} from './answer.ts';
import { MAX_TRANSIENT_ATTEMPTS, rankCandidates, RouterState, type Candidate } from './router.ts';

export interface ServeContext {
  envUrl: string;
  brief: Brief;
  bundle: CatalogBundle;
  router: RouterState;
  created: number;
}

export interface ServedAnswer {
  answer: ParsedAnswer;
  providerWire: 'openai' | 'anthropic' | 'gemini';
  model: string;
  streamed: boolean;
}

type TryResult =
  | { ok: true; answer: ParsedAnswer }
  | { ok: false; error: ProviderError }
  | { ok: false; unusable: true };

async function tryCandidate(ctx: ServeContext, candidate: Candidate, req: CanonicalRequest): Promise<TryResult> {
  const { provider, row } = candidate;
  const body =
    provider.wire === 'openai'
      ? toOpenAI(req, row.model, req.stream, true)
      : provider.wire === 'anthropic'
        ? toAnthropic(req, row.model, req.stream)
        : toGemini(req, row.model);
  try {
    if (req.stream) {
      const result = await callStream({ envUrl: ctx.envUrl, provider, model: row.model, body, stream: true, anthropicVersion: req.anthropicVersion });
      const answer = answerFromStream(provider.wire, result.frames);
      if (!isUsable(answer)) return { ok: false, unusable: true };
      return { ok: true, answer };
    }
    const raw = await callJson({ envUrl: ctx.envUrl, provider, model: row.model, body, stream: false, anthropicVersion: req.anthropicVersion });
    const answer = answerFromJson(provider.wire, raw);
    if (!isUsable(answer)) return { ok: false, unusable: true };
    return { ok: true, answer };
  } catch (err) {
    if (err instanceof ProviderError) return { ok: false, error: err };
    throw err;
  }
}

export async function serveRequest(ctx: ServeContext, req: CanonicalRequest): Promise<ServedAnswer | null> {
  const candidates = ctx.router.usable(rankCandidates(ctx.brief, ctx.bundle.freeByPair, req));
  let fallback: ServedAnswer | null = null;
  for (const candidate of candidates) {
    let attempts = 0;
    for (;;) {
      attempts++;
      const result = await tryCandidate(ctx, candidate, req);
      if (result.ok) {
        const served: ServedAnswer = { answer: result.answer, providerWire: candidate.provider.wire, model: candidate.row.model, streamed: req.stream };
        if (result.answer.finish?.usage !== null && result.answer.finish?.usage !== undefined) return served;
        fallback ??= served;
        break;
      }
      if ('unusable' in result) {
        if (attempts < 2) continue;
        break;
      }
      const error = result.error;
      if (error.kind === 'permanent' || error.kind === 'auth') {
        ctx.router.retireProvider(candidate.provider.id);
        break;
      }
      if (error.kind === 'bad-request') {
        ctx.router.retirePair(candidate.provider.id, candidate.row.model);
        break;
      }
      if (attempts < MAX_TRANSIENT_ATTEMPTS) continue;
      break;
    }
  }
  return fallback;
}

export function renderResponse(
  ctx: ServeContext,
  clientWire: 'openai' | 'anthropic',
  served: ServedAnswer,
): { status: number; contentType: string; body: string } {
  const emitCtx = { model: served.model, created: ctx.created };
  if (clientWire === 'openai') {
    if (!served.streamed) {
      return { status: 200, contentType: 'application/json', body: emitOpenAIJson(served.answer, emitCtx) };
    }
    if (served.providerWire === 'openai') {
      const passthrough = passthroughWithRepair('openai', served.answer, emitCtx);
      if (passthrough !== null) {
        return { status: 200, contentType: 'text/event-stream; charset=utf-8', body: passthrough };
      }
    }
    return { status: 200, contentType: 'text/event-stream; charset=utf-8', body: emitOpenAISse(served.answer, emitCtx) };
  }
  if (!served.streamed) {
    return { status: 200, contentType: 'application/json', body: emitAnthropicJson(served.answer, emitCtx) };
  }
  if (served.providerWire === 'anthropic') {
    const passthrough = passthroughWithRepair('anthropic', served.answer, emitCtx);
    if (passthrough !== null) {
      return { status: 200, contentType: 'text/event-stream; charset=utf-8', body: passthrough };
    }
  }
  return { status: 200, contentType: 'text/event-stream; charset=utf-8', body: emitAnthropicSse(served.answer, emitCtx) };
}

export function errorBody(wire: 'openai' | 'anthropic', status: number, message: string): { status: number; contentType: string; body: string } {
  if (wire === 'openai') {
    const type = status === 429 ? 'rate_limit_error' : status >= 500 ? 'api_error' : 'invalid_request_error';
    return {
      status,
      contentType: 'application/json',
      body: `${JSON.stringify({ error: { message, type, param: null, code: 'upstream_unavailable' } })}\n`,
    };
  }
  const type = status === 401 ? 'authentication_error' : status === 400 ? 'invalid_request_error' : 'api_error';
  return {
    status,
    contentType: 'application/json',
    body: `${JSON.stringify({ type: 'error', error: { type, message } })}\n`,
  };
}
