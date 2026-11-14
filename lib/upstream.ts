// lib/upstream.ts — calls to one provider: auth, error envelopes, and the
// two call shapes (one JSON answer, one SSE stream). Every provider's error
// is parsed in its own wire's envelope and classified, because the failover
// decision depends on what the provider actually said:
//
//   - "budget is spent" / "is out"  -> permanent: the provider serves no more
//     of this trial, whatever its remaining budget looked like.
//   - "refused this call"           -> transient: the provider's failure
//     schedule refused this call number; the next call may succeed.
//   - concurrent_requests           -> transient: another call was in flight.
//   - 401 / 400                     -> ours to fix, never retried blindly.
//   - anything else                 -> transient, retried once, then we move on.
//
// One provider call is ever in flight (a promise-chain mutex): the client
// sends one request at a time, and a provider that serves one call at a time
// must never see two.

import type { ProviderDir, Wire } from './brief.ts';
import { isRecord, joinUrl, strOrNull } from './util.ts';
import { SseParser, type SseFrame } from './sse.ts';

export type FailureKind = 'permanent' | 'transient' | 'auth' | 'bad-request' | 'network';

export class ProviderError extends Error {
  readonly status: number;
  readonly code: string;
  readonly kind: FailureKind;
  readonly retryAfter: number | null;

  constructor(status: number, code: string, message: string, kind: FailureKind, retryAfter: number | null = null) {
    super(message);
    this.status = status;
    this.code = code;
    this.kind = kind;
    this.retryAfter = retryAfter;
  }
}

export function authHeaders(provider: ProviderDir): Record<string, string> {
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  if (provider.auth === 'x-goog-api-key') return { 'x-goog-api-key': provider.key };
  return {};
}

/** Parse an error body in the provider's own envelope. */
export function parseErrorBody(wire: Wire, status: number, bodyText: string, retryAfterHeader: string | null): ProviderError {
  let code = '';
  let message = bodyText.slice(0, 500);
  let retryAfter: number | null = null;
  if (retryAfterHeader !== null) {
    const n = Number(retryAfterHeader);
    if (Number.isFinite(n) && n >= 0) retryAfter = n;
  }
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(bodyText);
  } catch {
    parsed = null;
  }
  if (isRecord(parsed) && isRecord(parsed.error)) {
    const err = parsed.error;
    if (typeof err.message === 'string' && err.message !== '') message = err.message;
    if (wire === 'openai') {
      code = strOrNull(err.code) ?? '';
    } else if (wire === 'anthropic') {
      code = strOrNull(err.type) ?? '';
    } else {
      code = strOrNull(err.status) ?? '';
    }
    const ra = err.retry_after;
    if (typeof ra === 'number' && Number.isFinite(ra) && ra >= 0) retryAfter = ra;
  }
  return classify(status, code, message, retryAfter);
}

/** Classify a refusal from its status, code and message. */
export function classify(status: number, code: string, message: string, retryAfter: number | null): ProviderError {
  const lower = message.toLowerCase();
  if (status === 401) return new ProviderError(status, code, message, 'auth', retryAfter);
  if (status === 400) return new ProviderError(status, code, message, 'bad-request', retryAfter);
  if (code === 'concurrent_requests') return new ProviderError(status, code, message, 'transient', retryAfter);
  if (lower.includes('budget is spent') || lower.includes('budget spent') || lower.includes('is out') || lower.includes('outage')) {
    return new ProviderError(status, code, message, 'permanent', retryAfter);
  }
  if (lower.includes('refused this call') || lower.includes('refused')) {
    return new ProviderError(status, code, message, 'transient', retryAfter);
  }
  if (status === 402 || status === 429 || status >= 500) {
    return new ProviderError(status, code, message, 'transient', retryAfter);
  }
  return new ProviderError(status, code, message, 'transient', retryAfter);
}

function inferPath(provider: ProviderDir, model: string, stream: boolean): string {
  if (provider.wire === 'openai') return '/v1/chat/completions';
  if (provider.wire === 'anthropic') return '/v1/messages';
  const verb = stream ? 'streamGenerateContent' : 'generateContent';
  return `/v1beta/models/${encodeURIComponent(model)}:${verb}`;
}

const CALL_TIMEOUT_MS = 20000;

/** Serialize provider calls: one in flight, in the order they were made. */
class CallMutex {
  private tail: Promise<void> = Promise.resolve();

  async run<T>(fn: () => Promise<T>): Promise<T> {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const previous = this.tail;
    this.tail = this.tail.then(() => gate);
    await previous;
    try {
      return await fn();
    } finally {
      release();
    }
  }
}

const mutex = new CallMutex();

export interface CallOptions {
  envUrl: string;
  provider: ProviderDir;
  model: string;
  body: Record<string, unknown>;
  stream: boolean;
  anthropicVersion: string | null;
}

function requestInit(provider: ProviderDir, body: Record<string, unknown>, anthropicVersion: string | null, signal: AbortSignal): RequestInit {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider),
  };
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = anthropicVersion ?? '2023-06-01';
  }
  return { method: 'POST', headers, body: JSON.stringify(body), signal };
}

async function throwForError(wire: Wire, res: Response): Promise<never> {
  let text = '';
  try {
    text = await res.text();
  } catch {
    text = '';
  }
  throw parseErrorBody(wire, res.status, text, res.headers.get('retry-after'));
}

/**
 * One JSON call: returns the parsed body. Throws ProviderError on any
 * refusal or transport failure.
 */
export async function callJson(opts: CallOptions): Promise<unknown> {
  return mutex.run(async () => {
    const url = joinUrl(joinUrl(opts.envUrl, opts.provider.path), inferPath(opts.provider, opts.model, false));
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CALL_TIMEOUT_MS);
    try {
      let res: Response;
      try {
        res = await fetch(url, requestInit(opts.provider, opts.body, opts.anthropicVersion, controller.signal));
      } catch (err) {
        throw new ProviderError(0, 'network_error', err instanceof Error ? err.message : 'network error', 'network');
      }
      if (res.status < 200 || res.status >= 300) await throwForError(opts.provider.wire, res);
      const text = await res.text();
      try {
        return JSON.parse(text) as unknown;
      } catch {
        throw new ProviderError(res.status, 'invalid_response', 'the provider answered a body that is not JSON', 'transient');
      }
    } finally {
      clearTimeout(timer);
    }
  });
}

export interface StreamResult {
  /** The frames, in arrival order, raw text preserved for passthrough. */
  frames: SseFrame[];
}

/**
 * One streaming call: consumes the provider's SSE stream to its end and
 * returns every frame. Consuming fully before answering is deliberate: a
 * stream that dies mid-way must fail over cleanly instead of leaving the
 * client a torn answer, and cross-wire streams need the final usage before
 * their first event can be written. Throws ProviderError on refusal,
 * transport failure, or a stream that ends with no frames at all.
 */
export async function callStream(opts: CallOptions): Promise<StreamResult> {
  return mutex.run(async () => {
    const url = joinUrl(joinUrl(opts.envUrl, opts.provider.path), inferPath(opts.provider, opts.model, true));
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CALL_TIMEOUT_MS);
    try {
      let res: Response;
      try {
        res = await fetch(url, requestInit(opts.provider, opts.body, opts.anthropicVersion, controller.signal));
      } catch (err) {
        throw new ProviderError(0, 'network_error', err instanceof Error ? err.message : 'network error', 'network');
      }
      if (res.status < 200 || res.status >= 300) await throwForError(opts.provider.wire, res);
      if (res.body === null) {
        throw new ProviderError(res.status, 'empty_response', 'the provider answered an empty stream', 'transient');
      }
      const parser = new SseParser();
      const decoder = new TextDecoder('utf-8');
      const frames: SseFrame[] = [];
      const reader = res.body.getReader();
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          const text = decoder.decode(value, { stream: true });
          if (text !== '') frames.push(...parser.feed(text));
        }
        const tail = decoder.decode();
        if (tail !== '') frames.push(...parser.feed(tail));
        frames.push(...parser.flush());
      } finally {
        reader.releaseLock();
      }
      if (frames.length === 0) {
        throw new ProviderError(res.status, 'empty_response', 'the provider answered no SSE events', 'transient');
      }
      return { frames };
    } finally {
      clearTimeout(timer);
    }
  });
}
