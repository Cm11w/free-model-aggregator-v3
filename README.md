# Free-Model Aggregator v3 — streaming wire fidelity

One server, two client wires (OpenAI Chat Completions, Anthropic Messages),
ten free-tier providers behind it. This entry's obsession is protocol
correctness: requests cross wires without changing what the provider parses
from them, and answers come back through the provider's own stream,
translated event by event.

## How it differs from a plain aggregator

- **Canonical request model.** Every client request is parsed into one
  lossless form (`lib/canonical.ts`) and re-emitted in the provider's wire
  (`lib/translate.ts`). System text joins identically, the last user turn's
  text and image bytes survive byte for byte (base64 payloads are never
  decoded and re-encoded), tool names keep their order, `max_tokens` and
  stop sequences cross unchanged, and JSON mode travels only on wires that
  can express it (OpenAI `response_format`, Gemini `responseMimeType` —
  never Anthropic, which would echo it as absent).
- **True streaming.** When the client asks for a stream, the provider is
  called with streaming on, and its SSE is parsed by a hardened incremental
  parser (`lib/sse.ts`: CRLF, comments, multi-line `data:`, split chunks,
  trailing partial events). Same-wire streams are validated and forwarded
  byte for byte, with a synthesized tail when the stream was cut short.
  Cross-wire streams are translated event by event (OpenAI chunks,
  Anthropic events, Gemini frames), including tool-call deltas and usage
  placement. Streams are consumed fully before answering, so a stream that
  dies mid-way fails over cleanly instead of leaving the client a torn
  answer, and cross-wire usage lands where each wire expects it.
- **Tool calls and multimodal content across wires.** OpenAI `tool_calls`
  ↔ Anthropic `tool_use` ↔ Gemini `functionCall`, including streaming
  argument deltas; images as `image_url` ↔ `image` blocks ↔ `inlineData`,
  with bytes preserved exactly.
- **Failover that reads the refusal.** Provider errors are parsed in each
  wire's own envelope and classified: "budget is spent" / "is out" retires
  the provider for the trial, "refused this call" is retried on the same
  candidate (the failure schedule refuses call numbers, not providers),
  auth failures retire the provider, bad-request failures retire the pair.
- **Determinism.** No clock, no unseeded randomness. The only timestamp
  emitted comes from `MAKEFUN_TRIAL_NOW`. Provider calls are serialized
  (one in flight, in order), candidate ranking is a pure function of the
  brief and the probes, and every choice derives from the brief, the
  requests, and the providers' answers — so a replay leaves the same
  state export.

## Layout

- `main.ts` — bootstrap, HTTP server, routes, register, `/work` poll, `--list`
- `lib/brief.ts` — the three JSON documents in the brief
- `lib/catalog.ts` — live catalog probes + the catalog document
- `lib/canonical.ts` — the canonical request + client parsers
- `lib/translate.ts` — canonical request → provider wire bodies
- `lib/sse.ts` — incremental SSE parser + frame emitters
- `lib/upstream.ts` — provider calls, error envelopes, classification
- `lib/answer.ts` — provider answers → canonical events → client answers
- `lib/router.ts` — candidate ranking + failover memory
- `lib/util.ts` — small shared helpers
- `tests/adversarial.mjs` — adversarial wire harness (mock providers)

## Verification

- Practice trials (process-local harness: the kit's env server + scorer, no
  docker on this machine): trial-0, trial-1, trial-2 each score **99**
  (quality 99 — the mathematical ceiling for this trial — served/wire/
  failover/catalog 100). Two runs of trial-0 produce byte-identical state
  exports, so a replay leaves the same result.
- Adversarial harness (`node tests/adversarial.mjs`, 50 checks, no trial
  needed): SSE framing at every chunk split 1–17 bytes and across split
  UTF-8 sequences; echo-equality of translation across all six client/
  provider wire pairs, checked against an independent re-implementation of
  the sim's echo rules; truncated, corrupt, and empty streams; same-wire
  byte-for-byte passthrough with tail repair; cross-wire stream emission
  against a mirror of the checker's stream-order rules; error-envelope
  classification; router ranking and failover against scripted mock
  providers, including the one-call-in-flight mutex. Two genuine bugs were
  found and fixed by this harness: Gemini function-response names now
  resolve from the matching tool call, and the harness's own echo model was
  corrected to the sim's exact rules (text blocks joined with "\n", the
  last image wins).

## Run

```sh
node main.ts            # serve a trial (MAKEFUN_PROMPT_FILE, MAKEFUN_ENV_URL)
node main.ts --list     # print the catalog document and exit 0
node tests/adversarial.mjs   # adversarial wire harness (no trial needed)
```

Node 22+ standard library only. No dependencies, no model calls: a scripted
entry, legal for this competition.
