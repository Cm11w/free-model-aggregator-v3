// main.ts — the free-model aggregator: one server, two client wires
// (OpenAI Chat Completions, Anthropic Messages), ten simulated free-tier
// providers behind it.
//
// Flow: read the brief, probe every provider's catalog route, build one
// catalog, listen, register with the env, serve the client's twenty
// requests, poll /work until the env says done, exit 0.
//
// Every request is translated into a canonical form, routed to the best
// free model that meets its requirements over a wire that can carry it,
// and answered in the wire the client used — streamed when the client
// asked for a stream, from the provider's own stream, translated event by
// event. Nothing here reads a clock or an unseeded random source: the only
// timestamp emitted comes from MAKEFUN_TRIAL_NOW, and every choice is a
// function of the brief, the requests, and the providers' answers.

import { readFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { parseBrief, type Brief } from './lib/brief.ts';
import { buildCatalog, emptyCatalogBundle, type CatalogBundle } from './lib/catalog.ts';
import { parseAnthropicRequest, parseOpenAIRequest } from './lib/canonical.ts';
import { RouterState } from './lib/router.ts';
import { errorBody, renderResponse, serveRequest, type ServeContext } from './lib/serve.ts';
import { isRecord, joinUrl, trialNowSeconds } from './lib/util.ts';

const MAX_BODY_BYTES = 2 * 1024 * 1024;
const WORK_POLL_MS = 100;

function log(message: string): void {
  process.stderr.write(`aggregator-v3: ${message}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Brief + catalog
// ---------------------------------------------------------------------------

function readBrief(): Brief {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (file === undefined || file === '') return parseBrief('');
  try {
    return parseBrief(readFileSync(file, 'utf8'));
  } catch {
    return parseBrief('');
  }
}

async function buildBundle(brief: Brief): Promise<CatalogBundle> {
  const envUrl = process.env.MAKEFUN_ENV_URL;
  if (envUrl === undefined || envUrl === '') {
    // No env to probe: the catalog is the brief's rows alone.
    const providers = brief.providers.map((p) => ({
      id: p.id,
      base_url: p.path,
      key_env: p.key_env,
      wire: p.wire,
      models: brief.ratings.filter((r) => r.provider === p.id).map((r) => ({ id: r.model, free: r.free, context_length: r.context_length })),
    }));
    const catalog = { providers };
    const freeIds: string[] = [];
    const seen = new Set<string>();
    for (const p of providers) {
      for (const m of p.models) {
        if (m.free && !seen.has(m.id)) {
          seen.add(m.id);
          freeIds.push(m.id);
        }
      }
    }
    const freeByPair = new Map<string, boolean>();
    for (const p of providers) for (const m of p.models) freeByPair.set(`${p.id}/${m.id}`, m.free);
    return { catalog, json: JSON.stringify(catalog), freeIds, freeByPair };
  }
  return buildCatalog(envUrl, brief);
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

function readBody(req: IncomingMessage): Promise<Buffer | null> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let tooLarge = false;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) tooLarge = true;
      else chunks.push(chunk);
    });
    req.on('end', () => resolve(tooLarge ? null : Buffer.concat(chunks)));
    req.on('error', () => resolve(null));
  });
}

function send(res: ServerResponse, status: number, contentType: string, body: string): void {
  const bytes = Buffer.from(body, 'utf8');
  res.writeHead(status, { 'content-type': contentType, 'content-length': String(bytes.length) });
  res.end(bytes);
}

function modelsList(ctx: ServeContext, req: IncomingMessage): { status: number; contentType: string; body: string } {
  const isAnthropic = req.headers['x-api-key'] !== undefined;
  if (isAnthropic) {
    return {
      status: 200,
      contentType: 'application/json',
      body: `${JSON.stringify({
        data: ctx.bundle.freeIds.map((id) => ({ type: 'model', id, display_name: id, created_at: '2026-01-01T00:00:00Z', free: true })),
        has_more: false,
      })}\n`,
    };
  }
  return {
    status: 200,
    contentType: 'application/json',
    body: `${JSON.stringify({
      object: 'list',
      data: ctx.bundle.freeIds.map((id) => ({ id, object: 'model', created: ctx.created, owned_by: 'aggregator-v3', free: true })),
    })}\n`,
  };
}

async function handleRequest(ctx: ServeContext, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  const path = url.pathname;
  try {
    if (req.method === 'GET' && path === '/v1/models') {
      const out = modelsList(ctx, req);
      send(res, out.status, out.contentType, out.body);
      return;
    }
    const wire: 'openai' | 'anthropic' | null =
      req.method === 'POST' && path === '/v1/chat/completions'
        ? 'openai'
        : req.method === 'POST' && path === '/v1/messages'
          ? 'anthropic'
          : null;
    if (wire === null) {
      send(res, 404, 'application/json', '{"error":"not_found"}\n');
      return;
    }
    const raw = await readBody(req);
    if (raw === null) {
      const out = errorBody(wire, 413, 'the request body is too large');
      send(res, out.status, out.contentType, out.body);
      return;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw.toString('utf8'));
    } catch {
      const out = errorBody(wire, 400, 'the request body is not JSON');
      send(res, out.status, out.contentType, out.body);
      return;
    }
    if (!isRecord(parsed)) {
      const out = errorBody(wire, 400, 'the request body is not one JSON object');
      send(res, out.status, out.contentType, out.body);
      return;
    }
    const headerVersion = req.headers['anthropic-version'];
    const anthropicVersion = typeof headerVersion === 'string' ? headerVersion : null;
    const canon = wire === 'openai' ? parseOpenAIRequest(parsed, anthropicVersion) : parseAnthropicRequest(parsed, anthropicVersion);
    const served = await serveRequest(ctx, canon);
    if (served === null) {
      const out = errorBody(wire, 502, 'no provider could serve the request');
      send(res, out.status, out.contentType, out.body);
      return;
    }
    const out = renderResponse(ctx, wire, served);
    send(res, out.status, out.contentType, out.body);
  } catch (err) {
    log(`request failed: ${err instanceof Error ? err.message : String(err)}`);
    if (!res.headersSent) {
      send(res, 500, 'application/json', '{"error":"internal"}\n');
    } else {
      res.end();
    }
  }
}

// ---------------------------------------------------------------------------
// Register + work poll
// ---------------------------------------------------------------------------

async function register(envUrl: string, port: number, bundle: CatalogBundle): Promise<boolean> {
  for (let attempt = 1; attempt <= 60; attempt++) {
    try {
      const res = await fetch(joinUrl(envUrl, '/register'), {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port, catalog: bundle.catalog }),
      });
      if (res.status >= 200 && res.status < 300) {
        try {
          await res.arrayBuffer();
        } catch {
          // Ignore drain errors.
        }
        return true;
      }
      try {
        await res.arrayBuffer();
      } catch {
        // Ignore drain errors.
      }
      log(`register answered HTTP ${res.status} (attempt ${attempt})`);
    } catch (err) {
      log(`register failed (attempt ${attempt}): ${err instanceof Error ? err.message : String(err)}`);
    }
    await sleep(500);
  }
  return false;
}

async function pollWork(envUrl: string): Promise<void> {
  for (;;) {
    try {
      const res = await fetch(joinUrl(envUrl, '/work'), { method: 'GET' });
      if (res.status >= 200 && res.status < 300) {
        const body: unknown = await res.json();
        if (isRecord(body) && body.done === true) return;
      } else {
        try {
          await res.arrayBuffer();
        } catch {
          // Ignore drain errors.
        }
      }
    } catch {
      // The env may not be ready yet; keep polling.
    }
    await sleep(WORK_POLL_MS);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.includes('--list')) {
    const brief = readBrief();
    const hasBrief = brief.providers.length > 0;
    const hasEnv = process.env.MAKEFUN_ENV_URL !== undefined && process.env.MAKEFUN_ENV_URL !== '';
    if (!hasBrief && !hasEnv) {
      process.stdout.write(`${emptyCatalogBundle().json}\n`);
      return;
    }
    const bundle = await buildBundle(brief);
    process.stdout.write(`${bundle.json}\n`);
    return;
  }

  const envUrl = process.env.MAKEFUN_ENV_URL ?? '';
  const brief = readBrief();
  const bundle = await buildBundle(brief);
  const ctx: ServeContext = {
    envUrl,
    brief,
    bundle,
    router: new RouterState(),
    created: trialNowSeconds(),
  };
  log(`brief: ${brief.providers.length} providers, ${brief.ratings.length} rated models; catalog: ${bundle.freeIds.length} free models`);

  const server = createServer((req, res) => {
    void handleRequest(ctx, req, res);
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen({ port: 0, host: '0.0.0.0' }, () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') {
    log('the listener has no TCP address');
    process.exit(1);
  }
  const port = address.port;
  log(`listening on 0.0.0.0:${port}`);

  if (envUrl === '') {
    log('MAKEFUN_ENV_URL is not set; serving without an env until stopped');
    return;
  }
  const ok = await register(envUrl, port, bundle);
  if (!ok) {
    log('register never succeeded; exiting');
    process.exit(1);
  }
  log('registered; waiting for the workload to finish');
  await pollWork(envUrl);
  log('work done; exiting 0');
  server.close();
  process.exit(0);
}

main().catch((err: unknown) => {
  log(`fatal: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
