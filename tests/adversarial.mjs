// tests/adversarial.mjs — adversarial wire harness for aggregator v3.
//
// This is the entry's own proof of protocol fidelity. It does not trust the
// practice trials alone: it drives the entry's own modules against mock
// providers that misbehave on purpose — split bytes, truncated streams,
// corrupt frames, hostile error envelopes, failure schedules — and checks
// that what comes out the other side is still correct.
//
//   node tests/adversarial.mjs
//
// No trial, no network beyond 127.0.0.1, no randomness. Exit 0 when every
// check passes.

import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { writeFileSync, unlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { SseParser, isDoneFrame, frameJson, eventFrame, DONE_FRAME } from '../lib/sse.ts';
import { parseOpenAIRequest, parseAnthropicRequest } from '../lib/canonical.ts';
import { toOpenAI, toAnthropic, toGemini } from '../lib/translate.ts';
import {
  answerFromJson,
  answerFromStream,
  isUsable,
  textOf,
  toolCallsOf,
  emitOpenAIJson,
  emitOpenAISse,
  emitAnthropicJson,
  emitAnthropicSse,
  passthroughWithRepair,
} from '../lib/answer.ts';
import { callJson, callStream, ProviderError, classify, parseErrorBody, authHeaders } from '../lib/upstream.ts';
import { parseBrief } from '../lib/brief.ts';
import { parseListing, buildCatalog } from '../lib/catalog.ts';
import { rankCandidates, RouterState } from '../lib/router.ts';
import { serveRequest, renderResponse } from '../lib/serve.ts';

// ---------------------------------------------------------------------------
// Tiny framework
// ---------------------------------------------------------------------------

let passed = 0;
let failed = 0;
const failures = [];

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

function eq(actual, expected, label) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  assert(a === e, `${label}: expected ${e}, got ${a}`);
}

async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`ok   - ${name}`);
  } catch (err) {
    failed++;
    failures.push(name);
    console.log(`FAIL - ${name}: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// ---------------------------------------------------------------------------
// Echo computers: an independent re-implementation of what the sim's
// provider parses from a request body, per wire — mirroring its rules
// exactly (text blocks joined with "\n", the LAST image wins, system parts
// joined with "\n", max_tokens falling back to max_completion_tokens,
// json_object and json_schema both counting as JSON mode). Translation is
// faithful when the echo of the translated body equals the echo of the
// client's body, field by field.
// ---------------------------------------------------------------------------

const sha8 = (text) => createHash('sha256').update(text, 'utf8').digest('hex').slice(0, 8);
const sha8buf = (buf) => createHash('sha256').update(buf).digest('hex').slice(0, 8);

function imageShaOfUrl(url) {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:[^,]*;base64,(.*)$/s.exec(url);
  if (data !== null) return sha8buf(Buffer.from(data[1], 'base64'));
  return sha8(url);
}

function blocksText(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts = [];
  for (const block of content) {
    if (block !== null && typeof block === 'object' && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function geminiPartsText(parts) {
  if (!Array.isArray(parts)) return '';
  const texts = [];
  for (const part of parts) {
    if (part !== null && typeof part === 'object' && typeof part.text === 'string') texts.push(part.text);
  }
  return texts.join('\n');
}

function stringList(v) {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x) => typeof x === 'string');
}

function echoOf(systemText, userText, tools, maxTokens, stops, imageSha8, jsonMode) {
  return {
    systemPresent: systemText !== '',
    systemSha: systemText === '' ? null : sha8(systemText),
    userSha: sha8(userText),
    tools,
    maxTokens,
    stops,
    imageSha: imageSha8,
    jsonMode,
  };
}

function echoOpenAI(body) {
  const msgs = Array.isArray(body.messages) ? body.messages.filter((m) => m !== null && typeof m === 'object') : [];
  const systemParts = [];
  let userText = '';
  let imageSha8 = null;
  for (const message of msgs) {
    if (message.role === 'system') {
      const text = blocksText(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user') {
      userText = blocksText(message.content);
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (block !== null && typeof block === 'object' && block.type === 'image_url' && block.image_url !== null && typeof block.image_url === 'object') {
            imageSha8 = imageShaOfUrl(block.image_url.url) ?? imageSha8;
          }
        }
      }
    }
  }
  const tools = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (tool !== null && typeof tool === 'object' && tool.function !== null && typeof tool.function === 'object' && typeof tool.function.name === 'string') tools.push(tool.function.name);
    }
  }
  const format = body.response_format !== null && typeof body.response_format === 'object' ? body.response_format.type : undefined;
  const maxTokens = (typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens) ? body.max_tokens : null) ?? (typeof body.max_completion_tokens === 'number' && Number.isFinite(body.max_completion_tokens) ? body.max_completion_tokens : null);
  return echoOf(systemParts.join('\n'), userText, tools, maxTokens, stringList(body.stop), imageSha8, format === 'json_object' || format === 'json_schema');
}

function echoAnthropic(body) {
  const systemText = blocksText(body.system);
  const msgs = Array.isArray(body.messages) ? body.messages.filter((m) => m !== null && typeof m === 'object') : [];
  let userText = '';
  let imageSha8 = null;
  for (const message of msgs) {
    if (message.role !== 'user') continue;
    userText = blocksText(message.content);
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (block === null || typeof block !== 'object' || block.type !== 'image' || block.source === null || typeof block.source !== 'object') continue;
        if (block.source.type === 'base64' && typeof block.source.data === 'string') imageSha8 = sha8buf(Buffer.from(block.source.data, 'base64'));
        else if (block.source.type === 'url') imageSha8 = imageShaOfUrl(block.source.url) ?? imageSha8;
      }
    }
  }
  const tools = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (tool !== null && typeof tool === 'object' && typeof tool.name === 'string') tools.push(tool.name);
    }
  }
  const maxTokens = typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens) ? body.max_tokens : null;
  return echoOf(systemText, userText, tools, maxTokens, stringList(body.stop_sequences), imageSha8, false);
}

function echoGemini(body) {
  const systemText = body.systemInstruction !== null && typeof body.systemInstruction === 'object' ? geminiPartsText(body.systemInstruction.parts) : '';
  const contents = Array.isArray(body.contents) ? body.contents.filter((c) => c !== null && typeof c === 'object') : [];
  let userText = '';
  let imageSha8 = null;
  for (const content of contents) {
    if (content.role === 'model') continue;
    userText = geminiPartsText(content.parts);
    if (Array.isArray(content.parts)) {
      for (const part of content.parts) {
        if (part === null || typeof part !== 'object') continue;
        if (part.inlineData !== null && typeof part.inlineData === 'object' && typeof part.inlineData.data === 'string') imageSha8 = sha8buf(Buffer.from(part.inlineData.data, 'base64'));
        else if (part.fileData !== null && typeof part.fileData === 'object' && typeof part.fileData.fileUri === 'string') imageSha8 = imageShaOfUrl(part.fileData.fileUri) ?? imageSha8;
      }
    }
  }
  const tools = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (tool === null || typeof tool !== 'object' || !Array.isArray(tool.functionDeclarations)) continue;
      for (const fn of tool.functionDeclarations) {
        if (fn !== null && typeof fn === 'object' && typeof fn.name === 'string') tools.push(fn.name);
      }
    }
  }
  const config = body.generationConfig !== null && typeof body.generationConfig === 'object' ? body.generationConfig : {};
  const maxTokens = typeof config.maxOutputTokens === 'number' && Number.isFinite(config.maxOutputTokens) ? config.maxOutputTokens : null;
  return echoOf(systemText, userText, tools, maxTokens, stringList(config.stopSequences), imageSha8, config.responseMimeType === 'application/json');
}

const ECHO_OF = { openai: echoOpenAI, anthropic: echoAnthropic, gemini: echoGemini };

// ---------------------------------------------------------------------------
// Checker mirrors: the stream-order rules the env's checker enforces,
// re-implemented here so emitted streams can be validated without a trial.
// ---------------------------------------------------------------------------

function parseSseText(text) {
  const parser = new SseParser();
  const frames = parser.feed(text);
  frames.push(...parser.flush());
  return frames;
}

function mirrorOpenAIStream(text) {
  const issues = [];
  const frames = parseSseText(text);
  let doneCount = 0;
  let finishCount = 0;
  let sawFinish = false;
  let sawDone = false;
  let collected = '';
  let finish = null;
  let usage = null;
  frames.forEach((f, i) => {
    if (isDoneFrame(f)) {
      doneCount++;
      if (i !== frames.length - 1) issues.push('[DONE] is not last');
      sawDone = true;
      return;
    }
    if (sawDone) issues.push('frame after [DONE]');
    const obj = frameJson(f);
    if (obj === null) {
      issues.push(`frame ${i} is not JSON`);
      return;
    }
    if (!Array.isArray(obj.choices)) {
      issues.push(`frame ${i} has no choices array`);
      return;
    }
    for (const ch of obj.choices) {
      const delta = ch.delta ?? {};
      if (typeof delta.content === 'string') {
        if (sawFinish) issues.push('content delta after the finish chunk');
        collected += delta.content;
      }
      if (ch.finish_reason !== null && ch.finish_reason !== undefined) {
        finishCount++;
        finish = ch.finish_reason;
        sawFinish = true;
      }
    }
    if (obj.usage) usage = obj.usage;
  });
  if (doneCount !== 1) issues.push(`[DONE] appears ${doneCount} times`);
  if (finishCount !== 1) issues.push(`${finishCount} finish chunks`);
  return { ok: issues.length === 0, issues, text: collected, finish, usage };
}

function mirrorAnthropicStream(text) {
  const issues = [];
  const frames = parseSseText(text);
  let sawStart = false;
  let sawDelta = false;
  let sawStop = false;
  const open = new Set();
  let collected = '';
  let stopReason = null;
  let inputTokens = null;
  let outputTokens = null;
  const toolArgs = new Map();
  frames.forEach((f, i) => {
    const obj = frameJson(f);
    if (obj === null) {
      issues.push(`frame ${i} is not JSON`);
      return;
    }
    const t = obj.type;
    if (t === 'message_start') {
      if (i !== 0) issues.push('message_start is not first');
      if (sawStart) issues.push('message_start twice');
      sawStart = true;
      inputTokens = obj.message?.usage?.input_tokens ?? null;
    } else if (t === 'content_block_start') {
      if (!sawStart) issues.push('block start before message_start');
      if (open.has(obj.index)) issues.push(`block ${obj.index} reopened`);
      open.add(obj.index);
    } else if (t === 'content_block_delta') {
      if (!open.has(obj.index)) issues.push(`delta for unopened block ${obj.index}`);
      if (obj.delta?.type === 'text_delta') collected += obj.delta.text ?? '';
      if (obj.delta?.type === 'input_json_delta') toolArgs.set(obj.index, (toolArgs.get(obj.index) ?? '') + (obj.delta.partial_json ?? ''));
    } else if (t === 'content_block_stop') {
      if (!open.has(obj.index)) issues.push(`stop for unopened block ${obj.index}`);
      open.delete(obj.index);
    } else if (t === 'message_delta') {
      if (open.size > 0) issues.push('message_delta with an open block');
      if (sawDelta) issues.push('message_delta twice');
      sawDelta = true;
      stopReason = obj.delta?.stop_reason ?? null;
      outputTokens = obj.usage?.output_tokens ?? null;
    } else if (t === 'message_stop') {
      if (!sawDelta) issues.push('message_stop before message_delta');
      if (i !== frames.length - 1) issues.push('message_stop is not last');
      sawStop = true;
    } else if (t === 'ping') {
      // Allowed anywhere.
    } else {
      issues.push(`unknown event ${t}`);
    }
  });
  if (!sawStart) issues.push('no message_start');
  if (!sawDelta) issues.push('no message_delta');
  if (!sawStop) issues.push('no message_stop');
  return { ok: issues.length === 0, issues, text: collected, stopReason, inputTokens, outputTokens, toolArgs };
}

// ---------------------------------------------------------------------------
// Mock providers
// ---------------------------------------------------------------------------

async function withMockServer(handler, fn) {
  const server = createServer(handler);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  const base = `http://127.0.0.1:${address.port}`;
  try {
    return await fn(base);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

function writeChunked(res, body, chunkSize) {
  const buf = Buffer.from(body, 'utf8');
  for (let i = 0; i < buf.length; i += chunkSize) res.write(buf.subarray(i, i + chunkSize));
  res.end();
}

function mockProvider(id, wire, auth = 'bearer') {
  return { id, name: id, wire, auth, key_env: 'ADV_MOCK_KEY', key: 'mock-key', path: `/p/${id}` };
}

const PNG_1X1_B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';

function openAIStreamText({ text, toolCalls = [], usage = { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }, includeUsage = true, finish = 'stop' }) {
  const frames = [];
  const head = { id: 'chatcmpl-mock', object: 'chat.completion.chunk', created: 1700000000, model: 'mock-model' };
  frames.push(`data: ${JSON.stringify({ ...head, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] })}\n\n`);
  for (let i = 0; i < text.length; i += 13) {
    frames.push(`data: ${JSON.stringify({ ...head, choices: [{ index: 0, delta: { content: text.slice(i, i + 13) }, finish_reason: null }] })}\n\n`);
  }
  toolCalls.forEach((tc, idx) => {
    frames.push(
      `data: ${JSON.stringify({ ...head, choices: [{ index: 0, delta: { tool_calls: [{ index: idx, id: tc.id, type: 'function', function: { name: tc.name, arguments: '' } }] }, finish_reason: null }] })}\n\n`,
    );
    for (let i = 0; i < tc.args.length; i += 7) {
      frames.push(
        `data: ${JSON.stringify({ ...head, choices: [{ index: 0, delta: { tool_calls: [{ index: idx, function: { arguments: tc.args.slice(i, i + 7) } }] }, finish_reason: null }] })}\n\n`,
      );
    }
  });
  frames.push(`data: ${JSON.stringify({ ...head, choices: [{ index: 0, delta: {}, finish_reason: finish }] })}\n\n`);
  if (includeUsage) frames.push(`data: ${JSON.stringify({ ...head, choices: [], usage })}\n\n`);
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

function anthropicStreamText({ text, toolCalls = [], inputTokens = 12, outputTokens = 6, stop = 'end_turn' }) {
  const frames = [];
  frames.push(
    eventFrame('message_start', {
      type: 'message_start',
      message: { id: 'msg_mock', type: 'message', role: 'assistant', model: 'mock-model', content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
    }),
  );
  let idx = 0;
  if (text !== '') {
    frames.push(eventFrame('content_block_start', { type: 'content_block_start', index: idx, content_block: { type: 'text', text: '' } }));
    for (let i = 0; i < text.length; i += 13) {
      frames.push(eventFrame('content_block_delta', { type: 'content_block_delta', index: idx, delta: { type: 'text_delta', text: text.slice(i, i + 13) } }));
    }
    frames.push(eventFrame('content_block_stop', { type: 'content_block_stop', index: idx }));
    idx++;
  }
  for (const tc of toolCalls) {
    frames.push(eventFrame('content_block_start', { type: 'content_block_start', index: idx, content_block: { type: 'tool_use', id: tc.id, name: tc.name, input: {} } }));
    for (let i = 0; i < tc.args.length; i += 7) {
      frames.push(eventFrame('content_block_delta', { type: 'content_block_delta', index: idx, delta: { type: 'input_json_delta', partial_json: tc.args.slice(i, i + 7) } }));
    }
    frames.push(eventFrame('content_block_stop', { type: 'content_block_stop', index: idx }));
    idx++;
  }
  frames.push(eventFrame('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: outputTokens } }));
  frames.push(eventFrame('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

function geminiStreamText({ text, toolCalls = [], inputTokens = 11, outputTokens = 5, finishReason = 'STOP' }) {
  const frames = [];
  for (let i = 0; i < text.length; i += 13) {
    frames.push(`data: ${JSON.stringify({ candidates: [{ content: { role: 'model', parts: [{ text: text.slice(i, i + 13) }] }, index: 0 }] })}\n\n`);
  }
  for (const tc of toolCalls) {
    frames.push(
      `data: ${JSON.stringify({ candidates: [{ content: { role: 'model', parts: [{ functionCall: { name: tc.name, args: JSON.parse(tc.args) } }] }, index: 0 }] })}\n\n`,
    );
  }
  frames.push(
    `data: ${JSON.stringify({ candidates: [{ content: { role: 'model', parts: [] }, finishReason, index: 0 }], usageMetadata: { promptTokenCount: inputTokens, candidatesTokenCount: outputTokens, totalTokenCount: inputTokens + outputTokens } })}\n\n`,
  );
  return frames.join('');
}

const STREAM_BUILDER = { openai: openAIStreamText, anthropic: anthropicStreamText, gemini: geminiStreamText };

function openAIJson({ text, toolCalls = [], usage = { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }, finish = 'stop' }) {
  return JSON.stringify({
    id: 'chatcmpl-mock',
    object: 'chat.completion',
    created: 1700000000,
    model: 'mock-model',
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: text,
          tool_calls: toolCalls.map((tc) => ({ id: tc.id, type: 'function', function: { name: tc.name, arguments: tc.args } })),
        },
        finish_reason: finish,
      },
    ],
    usage,
  });
}

function anthropicJson({ text, toolCalls = [], inputTokens = 12, outputTokens = 6, stop = 'end_turn' }) {
  const content = [];
  if (text !== '') content.push({ type: 'text', text });
  for (const tc of toolCalls) content.push({ type: 'tool_use', id: tc.id, name: tc.name, input: JSON.parse(tc.args) });
  return JSON.stringify({ id: 'msg_mock', type: 'message', role: 'assistant', model: 'mock-model', content, stop_reason: stop, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: outputTokens } });
}

function geminiJson({ text, toolCalls = [], inputTokens = 11, outputTokens = 5, finishReason = 'STOP' }) {
  const parts = [];
  if (text !== '') parts.push({ text });
  for (const tc of toolCalls) parts.push({ functionCall: { name: tc.name, args: JSON.parse(tc.args) } });
  return JSON.stringify({ candidates: [{ content: { role: 'model', parts }, finishReason, index: 0 }], usageMetadata: { promptTokenCount: inputTokens, candidatesTokenCount: outputTokens, totalTokenCount: inputTokens + outputTokens } });
}

const JSON_BUILDER = { openai: openAIJson, anthropic: anthropicJson, gemini: geminiJson };

// ---------------------------------------------------------------------------
// SSE parser
// ---------------------------------------------------------------------------

await test('sse: LF frames parse', () => {
  const frames = parseSseText('data: {"a":1}\n\ndata: {"a":2}\n\n');
  eq(frames.length, 2, 'frame count');
  eq(frameJson(frames[0]), { a: 1 }, 'first payload');
  eq(frameJson(frames[1]), { a: 2 }, 'second payload');
});

await test('sse: CRLF frames parse, mixed with LF', () => {
  const frames = parseSseText('data: {"a":1}\r\n\r\ndata: {"a":2}\n\n');
  eq(frames.length, 2, 'frame count');
  eq(frameJson(frames[0]), { a: 1 }, 'CRLF payload');
  eq(frameJson(frames[1]), { a: 2 }, 'LF payload');
});

await test('sse: comments, id, retry ignored; event names captured', () => {
  const frames = parseSseText(': hello\nid: 7\nretry: 3000\nevent: message_start\ndata: {"x":1}\n\ndata: {"x":2}\n\n');
  eq(frames.length, 2, 'frame count');
  eq(frames[0].event, 'message_start', 'event name');
  eq(frames[1].event, null, 'no event on second frame');
  eq(frameJson(frames[0]), { x: 1 }, 'payload survives the comment block');
});

await test('sse: multi-line data joins with newline', () => {
  const frames = parseSseText('data: line1\ndata: line2\n\n');
  eq(frames.length, 1, 'frame count');
  eq(frames[0].data, 'line1\nline2', 'joined data');
});

await test('sse: trailing partial frame dispatched by flush', () => {
  const parser = new SseParser();
  const first = parser.feed('data: {"a":1}\n\ndata: {"a":2}');
  eq(first.length, 1, 'complete frames only');
  const rest = parser.flush();
  eq(rest.length, 1, 'flushed partial frame');
  eq(frameJson(rest[0]), { a: 2 }, 'partial payload');
  eq(parser.flush().length, 0, 'flush is idempotent');
});

await test('sse: split invariance at every chunk size 1..17', () => {
  const body = openAIStreamText({ text: 'Hello, streaming world! This is a longer answer with several deltas.' });
  const whole = parseSseText(body).map((f) => f.raw);
  for (let size = 1; size <= 17; size++) {
    const parser = new SseParser();
    const frames = [];
    for (let i = 0; i < body.length; i += size) frames.push(...parser.feed(body.slice(i, i + size)));
    frames.push(...parser.flush());
    eq(frames.map((f) => f.raw), whole, `chunk size ${size}`);
  }
});

await test('sse: UTF-8 split across chunk boundaries survives a streaming decoder', () => {
  const text = 'Halo dunia — ünïcödé 😀 dan emoji lain 🎉 di dalam delta.';
  const body = openAIStreamText({ text });
  const buf = Buffer.from(body, 'utf8');
  const decoder = new TextDecoder();
  const parser = new SseParser();
  const frames = [];
  // 1-byte chunks guarantee splits inside multibyte sequences.
  for (let i = 0; i < buf.length; i++) frames.push(...parser.feed(decoder.decode(buf.subarray(i, i + 1), { stream: true })));
  frames.push(...parser.feed(decoder.decode()));
  frames.push(...parser.flush());
  const answer = answerFromStream('openai', frames);
  eq(textOf(answer.events), text, 'decoded text intact');
  assert(isUsable(answer), 'answer usable');
});

await test('sse: [DONE] detection, garbage lines, empty input', () => {
  const frames = parseSseText('garbage line without a colon\ndata: [DONE]\n\n');
  eq(frames.length, 1, 'garbage line yields no frame');
  assert(isDoneFrame(frames[0]), '[DONE] recognized');
  eq(DONE_FRAME, 'data: [DONE]\n\n', 'DONE frame literal');
  eq(parseSseText('').length, 0, 'empty input');
  eq(parseSseText(': only a comment\n\n').length, 0, 'comment-only input');
  const answer = answerFromStream('openai', parseSseText('data: [DONE]\n\n'));
  assert(!isUsable(answer), '[DONE]-only stream is not usable');
});

await test('sse: eventFrame/dataFrame round-trip through the parser', () => {
  const body = eventFrame('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn' } });
  const frames = parseSseText(body);
  eq(frames.length, 1, 'one frame');
  eq(frames[0].event, 'message_delta', 'event preserved');
  eq(frameJson(frames[0]).delta.stop_reason, 'end_turn', 'payload preserved');
});

// ---------------------------------------------------------------------------
// Translation echo-equality: the provider must parse exactly what the
// client sent, on every wire pair.
// ---------------------------------------------------------------------------

const TOOLS = [
  { name: 'lookup_customer', description: 'Look up a customer.', input_schema: { type: 'object', properties: { id: { type: 'string' } }, required: ['id'] } },
  { name: 'create_ticket', description: 'Create a ticket.', input_schema: { type: 'object', properties: { title: { type: 'string' } } } },
];

function openAIClientRequest(overrides = {}) {
  return {
    model: 'auto',
    messages: [
      { role: 'system', content: 'You are a careful assistant.' },
      { role: 'system', content: 'Answer concisely.' },
      { role: 'user', content: 'Who is customer 42?' },
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Describe this chart and the linked photo.' },
          { type: 'image_url', image_url: { url: `data:image/png;base64,${PNG_1X1_B64}` } },
          { type: 'image_url', image_url: { url: 'https://example.invalid/photo.png' } },
        ],
      },
    ],
    tools: TOOLS.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.input_schema } })),
    max_tokens: 512,
    stop: '\n\n###',
    ...overrides,
  };
}

function anthropicClientRequest(overrides = {}) {
  return {
    model: 'auto',
    system: 'You are a careful assistant.\n\nAnswer concisely.',
    messages: [
      { role: 'user', content: 'Who is customer 42?' },
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Describe this chart and the linked photo.' },
          { type: 'image', source: { type: 'base64', media_type: 'image/png', data: PNG_1X1_B64 } },
          { type: 'image', source: { type: 'url', url: 'https://example.invalid/photo.png' } },
        ],
      },
    ],
    tools: TOOLS,
    max_tokens: 512,
    stop_sequences: ['\n\nHuman:'],
    ...overrides,
  };
}

const TO_WIRE = { openai: (req, model) => toOpenAI(req, model, false, false), anthropic: (req, model) => toAnthropic(req, model, false), gemini: (req, model) => toGemini(req, model) };

await test('translate: echo equality across all client/provider wire pairs', () => {
  const cases = [
    ['openai', openAIClientRequest(), parseOpenAIRequest],
    ['openai', openAIClientRequest({ messages: [{ role: 'user', content: 'plain hello' }] }), parseOpenAIRequest],
    ['anthropic', anthropicClientRequest(), parseAnthropicRequest],
    ['anthropic', anthropicClientRequest({ messages: [{ role: 'user', content: 'plain hello' }], system: undefined, tools: undefined, stop_sequences: undefined, max_tokens: undefined }), parseAnthropicRequest],
  ];
  for (const [clientWire, body, parse] of cases) {
    const canon = parse(body, null);
    const clientEcho = ECHO_OF[clientWire](body);
    for (const providerWire of ['openai', 'anthropic', 'gemini']) {
      const translated = TO_WIRE[providerWire](canon, 'mock-model');
      const providerEcho = ECHO_OF[providerWire](translated);
      eq(providerEcho, clientEcho, `${clientWire} -> ${providerWire}`);
    }
  }
});

await test('translate: JSON mode survives on OpenAI and Gemini, never on Anthropic', () => {
  const canon = parseOpenAIRequest(openAIClientRequest({ response_format: { type: 'json_object' } }), null);
  eq(ECHO_OF.openai(toOpenAI(canon, 'm', false, false)).jsonMode, true, 'openai json');
  eq(ECHO_OF.gemini(toGemini(canon, 'm')).jsonMode, true, 'gemini json');
  // The router must never send a JSON-mode request to an Anthropic-wire provider.
  const brief = parseBrief(`x ${JSON.stringify({ schema: 'free-model-aggregator-01/providers/v1', providers: [{ id: 'anth', name: 'anth', wire: 'anthropic', auth: 'x-api-key', key_env: 'K', key: 'k', path: '/p/anth' }, { id: 'oai', name: 'oai', wire: 'openai', auth: 'bearer', key_env: 'K', key: 'k', path: '/p/oai' }] })} y ${JSON.stringify({ schema: 'free-model-aggregator-01/ratings/v1', ratings: [{ model: 'm-a', provider: 'anth', rating_milli: 1500, context_length: 200000, tools: true, vision: true, json: true, free: true }, { model: 'm-o', provider: 'oai', rating_milli: 1400, context_length: 200000, tools: true, vision: true, json: true, free: true }] })} z ${JSON.stringify({ schema: 'free-model-aggregator-01/workload/v1', manifest: [] })}`);
  const ranked = rankCandidates(brief, new Map(), canon);
  eq(ranked.map((c) => c.provider.id), ['oai'], 'anthropic-wire provider excluded for JSON mode');
});

await test('translate: image bytes and URLs survive verbatim through every wire', () => {
  const canon = parseOpenAIRequest(openAIClientRequest(), null);
  const asAnthropic = toAnthropic(canon, 'm', false);
  const imgBlocks = asAnthropic.messages[1].content.filter((b) => b.type === 'image');
  eq(imgBlocks[0].source.data, PNG_1X1_B64, 'anthropic base64 verbatim');
  eq(imgBlocks[1].source.url, 'https://example.invalid/photo.png', 'anthropic url verbatim');
  const asGemini = toGemini(canon, 'm');
  const parts = asGemini.contents[1].parts;
  eq(parts.find((p) => p.inlineData).inlineData.data, PNG_1X1_B64, 'gemini base64 verbatim');
  eq(parts.find((p) => p.fileData).fileData.fileUri, 'https://example.invalid/photo.png', 'gemini url verbatim');
  // And back: Anthropic -> OpenAI keeps the same bytes.
  const back = toOpenAI(parseAnthropicRequest(anthropicClientRequest(), null), 'm', false, false);
  const multi = back.messages.find((m) => Array.isArray(m.content));
  const backImgs = multi.content.filter((p) => p.type === 'image_url');
  eq(backImgs[0].image_url.url, `data:image/png;base64,${PNG_1X1_B64}`, 'openai data url verbatim');
  eq(backImgs[1].image_url.url, 'https://example.invalid/photo.png', 'openai url verbatim');
});

await test('translate: tool results survive across wires with id and content', () => {
  const withTools = openAIClientRequest({
    messages: [
      { role: 'user', content: 'Look up 42' },
      { role: 'assistant', content: null, tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'lookup_customer', arguments: '{"id":"42"}' } }] },
      { role: 'tool', tool_call_id: 'call_1', content: 'Customer 42: Ada, premium.' },
      { role: 'user', content: 'Summarize.' },
    ],
  });
  const canon = parseOpenAIRequest(withTools, null);
  const asAnthropic = toAnthropic(canon, 'm', false);
  const resultBlock = asAnthropic.messages.flatMap((m) => (Array.isArray(m.content) ? m.content : [])).find((b) => b.type === 'tool_result');
  assert(resultBlock !== undefined, 'tool_result present');
  eq(resultBlock.tool_use_id, 'call_1', 'tool_result id');
  assert(JSON.stringify(resultBlock.content).includes('Ada, premium'), 'tool_result content');
  const asGemini = toGemini(canon, 'm');
  const fnResp = asGemini.contents.flatMap((c) => c.parts).find((p) => p.functionResponse);
  assert(fnResp !== undefined, 'functionResponse present');
  eq(fnResp.functionResponse.name, 'lookup_customer', 'functionResponse name');
  assert(JSON.stringify(fnResp.functionResponse.response).includes('Ada, premium'), 'functionResponse content');
  // Anthropic -> OpenAI direction.
  const anth = anthropicClientRequest({
    messages: [
      { role: 'user', content: 'Look up 42' },
      { role: 'assistant', content: [{ type: 'tool_use', id: 'call_9', name: 'lookup_customer', input: { id: '42' } }] },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'call_9', content: 'Customer 42: Ada.' }] },
    ],
  });
  const back = toOpenAI(parseAnthropicRequest(anth, null), 'm', false, false);
  const toolMsg = back.messages.find((m) => m.role === 'tool');
  assert(toolMsg !== undefined, 'tool message present');
  eq(toolMsg.tool_call_id, 'call_9', 'tool message id');
  assert(toolMsg.content.includes('Ada'), 'tool message content');
});

await test('translate: Gemini schema sanitizer drops forbidden keys, OpenAI keeps them', () => {
  const schema = { type: 'object', $schema: 'http://example.invalid/x', properties: { n: { type: 'number', exclusiveMinimum: 0, exclusiveMaximum: 10 } }, required: ['n'], additionalProperties: false };
  const canon = parseOpenAIRequest(openAIClientRequest({ tools: [{ type: 'function', function: { name: 'f', description: 'd', parameters: schema } }] }), null);
  const gem = toGemini(canon, 'm');
  const decl = gem.tools[0].functionDeclarations[0];
  assert(!('additionalProperties' in decl.parameters), 'additionalProperties dropped');
  assert(!('$schema' in decl.parameters), '$schema dropped');
  assert(!('exclusiveMinimum' in decl.parameters.properties.n), 'exclusiveMinimum dropped');
  assert(!('exclusiveMaximum' in decl.parameters.properties.n), 'exclusiveMaximum dropped');
  const oai = toOpenAI(canon, 'm', false, false);
  eq(oai.tools[0].function.parameters.additionalProperties, false, 'openai keeps additionalProperties');
});

await test('translate: empty and degenerate requests do not crash and echo equal', () => {
  const cases = [
    ['openai', { model: 'auto', messages: [{ role: 'user', content: '' }] }, parseOpenAIRequest],
    ['openai', { model: 'auto', messages: [{ role: 'user', content: [] }] }, parseOpenAIRequest],
    ['anthropic', { model: 'auto', messages: [{ role: 'user', content: '' }], max_tokens: 1 }, parseAnthropicRequest],
    ['openai', { model: 'auto', messages: [{ role: 'system', content: 'sys' }, { role: 'user', content: 'hi' }], stop: [] }, parseOpenAIRequest],
  ];
  for (const [clientWire, body, parse] of cases) {
    const canon = parse(body, null);
    for (const providerWire of ['openai', 'anthropic', 'gemini']) {
      const translated = TO_WIRE[providerWire](canon, 'mock-model');
      eq(ECHO_OF[providerWire](translated), ECHO_OF[clientWire](body), `${clientWire} -> ${providerWire} degenerate`);
    }
  }
});

// ---------------------------------------------------------------------------
// Streaming translation against mock providers
// ---------------------------------------------------------------------------

async function streamFromMock(wire, bodyText, chunkSize) {
  const provider = mockProvider('mock', wire);
  return withMockServer((req, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    writeChunked(res, bodyText, chunkSize);
  }, async (base) => {
    const result = await callStream({ envUrl: base, provider, model: 'mock-model', body: {}, stream: true, anthropicVersion: null });
    return answerFromStream(wire, result.frames);
  });
}

await test('stream: OpenAI 13-char deltas at 1-byte chunks reassemble exactly', async () => {
  const text = 'ANSWER ECHO{"system_present":true,"user_sha8":"deadbeef"} and more text after the echo, long enough to span many deltas.';
  const answer = await streamFromMock('openai', openAIStreamText({ text }), 1);
  assert(isUsable(answer), 'usable');
  eq(textOf(answer.events), text, 'text exact');
  eq(answer.finish.usage, { input: 10, output: 5 }, 'usage exact');
  eq(answer.finish.stop, 'stop', 'stop family');
  assert(answer.complete, 'complete');
  assert(!answer.corrupt, 'not corrupt');
});

await test('stream: Anthropic deltas at 3-byte chunks, with a tool call', async () => {
  const args = '{"id":"42","verbose":true}';
  const answer = await streamFromMock('anthropic', anthropicStreamText({ text: 'Looking that up now.', toolCalls: [{ id: 'call_1', name: 'lookup_customer', args }] }), 3);
  assert(isUsable(answer), 'usable');
  eq(textOf(answer.events), 'Looking that up now.', 'text exact');
  const calls = toolCallsOf(answer.events);
  eq(calls.length, 1, 'one call');
  eq(calls[0].id, 'call_1', 'call id');
  eq(calls[0].name, 'lookup_customer', 'call name');
  eq(calls[0].argsText, args, 'args reassembled');
  eq(JSON.parse(calls[0].argsText), { id: '42', verbose: true }, 'args parse');
  eq(answer.finish.usage, { input: 12, output: 6 }, 'usage exact');
  eq(answer.finish.stop, 'stop', 'stop family from end_turn');
});

await test('stream: Gemini frames at 5-byte chunks, with a function call', async () => {
  const answer = await streamFromMock('gemini', geminiStreamText({ text: 'Gemini says hello.', toolCalls: [{ id: 'x', name: 'create_ticket', args: '{"title":"help"}' }] }), 5);
  assert(isUsable(answer), 'usable');
  eq(textOf(answer.events), 'Gemini says hello.', 'text exact');
  const calls = toolCallsOf(answer.events);
  eq(calls.length, 1, 'one call');
  eq(calls[0].name, 'create_ticket', 'call name');
  eq(JSON.parse(calls[0].argsText), { title: 'help' }, 'args parse');
  eq(answer.finish.usage, { input: 11, output: 5 }, 'usage exact');
  eq(answer.finish.stop, 'stop', 'STOP maps to stop family');
});

await test('stream: OpenAI tool-call argument deltas assemble in order', async () => {
  const args = '{"customer_id":"42","notes":"a longer argument string that spans several deltas"}';
  const answer = await streamFromMock('openai', openAIStreamText({ text: '', toolCalls: [{ id: 'call_7', name: 'lookup_customer', args }], finish: 'tool_calls' }), 2);
  assert(isUsable(answer), 'usable');
  const calls = toolCallsOf(answer.events);
  eq(calls[0].argsText, args, 'args exact');
  eq(answer.finish.stop, 'tool_calls', 'tool_calls family preserved');
});

await test('stream: truncated stream (no finish, no [DONE]) is unusable', async () => {
  const full = openAIStreamText({ text: 'cut short' });
  const cut = full.split('data:')[0] + 'data:' + full.split('data:')[1] + 'data:' + full.split('data:')[2];
  const answer = await streamFromMock('openai', cut, 4);
  assert(answer.finish === null, 'no finish record');
  assert(!isUsable(answer), 'unusable, so the router fails over');
});

await test('stream: corrupt JSON frame mid-stream marks the answer corrupt', async () => {
  const good = openAIStreamText({ text: 'before the corruption' });
  const frames = good.split('\n\n');
  frames.splice(2, 0, 'data: {not json at all');
  const answer = await streamFromMock('openai', frames.join('\n\n'), 6);
  assert(answer.corrupt, 'corrupt flagged');
  assert(!isUsable(answer), 'unusable');
});

await test('stream: Anthropic error event mid-stream marks the answer corrupt', async () => {
  const body = anthropicStreamText({ text: 'partial' }).replace(
    'event: message_stop',
    'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\nevent: message_stop',
  );
  const answer = await streamFromMock('anthropic', body, 4);
  assert(answer.corrupt, 'corrupt flagged');
  assert(!isUsable(answer), 'unusable');
});

await test('stream: empty stream and JSON-only garbage fail the call (transient), so the router fails over', async () => {
  for (const body of ['', 'this is not sse at all\n\n']) {
    let thrown = null;
    try {
      await streamFromMock('openai', body, 4);
    } catch (err) {
      thrown = err;
    }
    assert(thrown instanceof ProviderError, `a ProviderError is thrown for ${JSON.stringify(body)}`);
    eq(thrown.kind, 'transient', 'empty/garbage stream is transient');
  }
});

await test('stream: unexpected fields and unknown events do not break parsing', async () => {
  const body =
    'event: ping\ndata: {"type":"ping"}\n\n' +
    anthropicStreamText({ text: 'fine' }).replace('"type":"message_start"', '"type":"message_start","mystery_field":{"nested":[1,2,3]}');
  const answer = await streamFromMock('anthropic', body, 4);
  assert(isUsable(answer), 'usable despite extras');
  eq(textOf(answer.events), 'fine', 'text exact');
});

await test('passthrough: complete same-wire streams forward byte-for-byte', async () => {
  for (const wire of ['openai', 'anthropic']) {
    const body = STREAM_BUILDER[wire]({ text: 'Byte-faithful forwarding, please.', toolCalls: wire === 'openai' ? [{ id: 'c1', name: 'lookup_customer', args: '{"id":"1"}' }] : [{ id: 'c1', name: 'lookup_customer', args: '{"id":"1"}' }] });
    const answer = await streamFromMock(wire, body, 9);
    const out = passthroughWithRepair(wire, answer, { model: 'mock-model', created: 1700000000 });
    assert(out !== null, 'passthrough applies');
    eq(out, body, `${wire} bytes identical`);
  }
});

await test('passthrough: missing [DONE] is repaired; missing message_stop is repaired', async () => {
  const oai = openAIStreamText({ text: 'no terminator here' }).replace('data: [DONE]\n\n', '');
  const answerOai = await streamFromMock('openai', oai, 4);
  assert(isUsable(answerOai), 'finish present, usable without [DONE]');
  const repairedOai = passthroughWithRepair('openai', answerOai, { model: 'mock-model', created: 1700000000 });
  assert(repairedOai !== null && repairedOai.endsWith('data: [DONE]\n\n'), '[DONE] appended');
  const mirrorOai = mirrorOpenAIStream(repairedOai);
  assert(mirrorOai.ok, `repaired openai stream passes: ${mirrorOai.issues.join('; ')}`);
  eq(mirrorOai.text, 'no terminator here', 'text survives repair');

  const anth = anthropicStreamText({ text: 'no stop here' }).replace(/event: message_stop\ndata: [^\n]+\n\n$/, '');
  const answerAnth = await streamFromMock('anthropic', anth, 4);
  assert(isUsable(answerAnth), 'message_delta present, usable without message_stop');
  const repairedAnth = passthroughWithRepair('anthropic', answerAnth, { model: 'mock-model', created: 1700000000 });
  assert(repairedAnth !== null, 'repair applies');
  const mirrorAnth = mirrorAnthropicStream(repairedAnth);
  assert(mirrorAnth.ok, `repaired anthropic stream passes: ${mirrorAnth.issues.join('; ')}`);
  eq(mirrorAnth.text, 'no stop here', 'text survives repair');
});

await test('emit: every provider wire x client wire stream passes the checker mirror', async () => {
  const text = 'Cross-wire translation must keep the text, the usage, and the stop.';
  for (const providerWire of ['openai', 'anthropic', 'gemini']) {
    const answer = await streamFromMock(providerWire, STREAM_BUILDER[providerWire]({ text }), 7);
    const ctx = { model: 'mock-model', created: 1700000000 };
    const oai = emitOpenAISse(answer, ctx);
    const mOai = mirrorOpenAIStream(oai);
    assert(mOai.ok, `openai emission from ${providerWire}: ${mOai.issues.join('; ')}`);
    eq(mOai.text, text, `openai text from ${providerWire}`);
    eq(mOai.finish, 'stop', `openai finish from ${providerWire}`);
    const anth = emitAnthropicSse(answer, ctx);
    const mAnth = mirrorAnthropicStream(anth);
    assert(mAnth.ok, `anthropic emission from ${providerWire}: ${mAnth.issues.join('; ')}`);
    eq(mAnth.text, text, `anthropic text from ${providerWire}`);
    eq(mAnth.stopReason, 'end_turn', `anthropic stop from ${providerWire}`);
  }
});

await test('emit: Anthropic message_start carries input usage even when the provider sends usage last', async () => {
  const answer = await streamFromMock('openai', openAIStreamText({ text: 'usage arrives last on this wire', usage: { prompt_tokens: 42, completion_tokens: 9, total_tokens: 51 } }), 5);
  const anth = emitAnthropicSse(answer, { model: 'mock-model', created: 1700000000 });
  const mirror = mirrorAnthropicStream(anth);
  assert(mirror.ok, `mirror: ${mirror.issues.join('; ')}`);
  eq(mirror.inputTokens, 42, 'input tokens in message_start');
  eq(mirror.outputTokens, 9, 'output tokens in message_delta');
});

await test('emit: OpenAI usage chunk carries exact counts from an Anthropic provider', async () => {
  const answer = await streamFromMock('anthropic', anthropicStreamText({ text: 'from anthropic', inputTokens: 33, outputTokens: 8 }), 5);
  const oai = emitOpenAISse(answer, { model: 'mock-model', created: 1700000000 });
  const mirror = mirrorOpenAIStream(oai);
  assert(mirror.ok, `mirror: ${mirror.issues.join('; ')}`);
  eq(mirror.usage, { prompt_tokens: 33, completion_tokens: 8, total_tokens: 41 }, 'usage chunk exact');
});

await test('emit: tool-only answers emit valid streams on both wires', async () => {
  const answer = await streamFromMock('openai', openAIStreamText({ text: '', toolCalls: [{ id: 'c1', name: 'lookup_customer', args: '{"id":"42"}' }], finish: 'tool_calls' }), 4);
  const ctx = { model: 'mock-model', created: 1700000000 };
  const mOai = mirrorOpenAIStream(emitOpenAISse(answer, ctx));
  assert(mOai.ok, `openai: ${mOai.issues.join('; ')}`);
  eq(mOai.finish, 'tool_calls', 'openai finish tool_calls');
  const mAnth = mirrorAnthropicStream(emitAnthropicSse(answer, ctx));
  assert(mAnth.ok, `anthropic: ${mAnth.issues.join('; ')}`);
  eq(mAnth.stopReason, 'tool_use', 'anthropic stop tool_use');
  const argsJson = [...mAnth.toolArgs.values()][0];
  eq(JSON.parse(argsJson), { id: '42' }, 'tool args survive the anthropic stream');
});

await test('emit: stop families map, never inferred from content', async () => {
  const cases = [
    ['openai', openAIStreamText({ text: 'not a tool call at all', finish: 'length' }), 'length', 'max_tokens'],
    ['anthropic', anthropicStreamText({ text: 'tool_use mentioned in text', stop: 'max_tokens' }), 'length', 'max_tokens'],
    ['gemini', geminiStreamText({ text: 'safety words here', finishReason: 'SAFETY' }), 'content_filter', 'refusal'],
    ['gemini', geminiStreamText({ text: '', finishReason: 'MAX_TOKENS' }), 'length', 'max_tokens'],
  ];
  for (const [wire, body, oaiStop, anthStop] of cases) {
    const answer = await streamFromMock(wire, body, 4);
    const ctx = { model: 'mock-model', created: 1700000000 };
    eq(mirrorOpenAIStream(emitOpenAISse(answer, ctx)).finish, oaiStop, `${wire} -> openai stop`);
    eq(mirrorAnthropicStream(emitAnthropicSse(answer, ctx)).stopReason, anthStop, `${wire} -> anthropic stop`);
  }
});

await test('json answers: all three wires parse to the same events', async () => {
  const text = 'JSON answer, same on every wire.';
  for (const wire of ['openai', 'anthropic', 'gemini']) {
    const raw = JSON.parse(JSON_BUILDER[wire]({ text }));
    const answer = answerFromJson(wire, raw);
    assert(isUsable(answer), `${wire} usable`);
    eq(textOf(answer.events), text, `${wire} text`);
    assert(answer.finish.usage !== null, `${wire} usage present`);
    const ctx = { model: 'mock-model', created: 1700000000 };
    eq(JSON.parse(emitOpenAIJson(answer, ctx)).choices[0].message.content, text, `${wire} -> openai json`);
    eq(JSON.parse(emitAnthropicJson(answer, ctx)).content[0].text, text, `${wire} -> anthropic json`);
  }
});

await test('json answers: malformed shapes are unusable, never crash', async () => {
  const bad = [null, 42, 'text', [], {}, { choices: 'nope' }, { candidates: [{ content: null }] }, { type: 'message', content: 'not an array' }];
  for (const raw of bad) {
    for (const wire of ['openai', 'anthropic', 'gemini']) {
      const answer = answerFromJson(wire, raw);
      assert(!isUsable(answer), `${wire} rejects ${JSON.stringify(raw)}`);
    }
  }
});

// ---------------------------------------------------------------------------
// Error envelopes and classification
// ---------------------------------------------------------------------------

await test('errors: auth header shapes per scheme', () => {
  eq(authHeaders(mockProvider('a', 'openai', 'bearer')), { authorization: 'Bearer mock-key' }, 'bearer');
  eq(authHeaders(mockProvider('a', 'anthropic', 'x-api-key')), { 'x-api-key': 'mock-key' }, 'x-api-key');
  eq(authHeaders(mockProvider('a', 'gemini', 'x-goog-api-key')), { 'x-goog-api-key': 'mock-key' }, 'x-goog-api-key');
  eq(authHeaders(mockProvider('a', 'openai', 'none')), {}, 'none');
});

await test('errors: parseErrorBody reads all three envelopes', () => {
  const oai = parseErrorBody('openai', 429, JSON.stringify({ error: { message: "acme's free-tier budget is spent", type: 'rate_limit_error', code: 'insufficient_quota' } }), null);
  eq(oai.kind, 'permanent', 'openai budget spent');
  eq(oai.status, 429, 'status kept');
  const anth = parseErrorBody('anthropic', 429, JSON.stringify({ type: 'error', error: { type: 'rate_limit_error', message: 'hooli refused this call' } }), '7');
  eq(anth.kind, 'transient', 'anthropic refused');
  eq(anth.retryAfter, 7, 'retry-after header parsed');
  const gem = parseErrorBody('gemini', 503, JSON.stringify({ error: { code: 503, message: 'gemini is out', status: 'UNAVAILABLE' } }), null);
  eq(gem.kind, 'permanent', 'gemini outage');
  const bad = parseErrorBody('openai', 400, 'not json at all', null);
  eq(bad.kind, 'bad-request', 'unparseable 400');
  const empty = parseErrorBody('gemini', 500, '', null);
  eq(empty.kind, 'transient', 'empty 500 is transient');
});

await test('errors: classify distinguishes permanent, transient, auth, bad-request', () => {
  eq(classify(429, 'x', "globex's free-tier budget is spent", null).kind, 'permanent', 'budget spent');
  eq(classify(429, 'x', 'initech is out', null).kind, 'permanent', 'outage');
  eq(classify(429, 'insufficient_quota', 'hooli refused this call', null).kind, 'transient', 'refused despite quota code');
  eq(classify(429, 'concurrent_requests', 'too many concurrent requests', null).kind, 'transient', 'concurrency clash');
  eq(classify(401, 'auth', 'bad key', null).kind, 'auth', '401');
  eq(classify(400, 'bad', 'bad request', null).kind, 'bad-request', '400');
  eq(classify(402, 'x', 'payment required', null).kind, 'transient', 'unknown 402 transient');
  eq(classify(500, 'x', 'boom', null).kind, 'transient', '500 transient');
  eq(classify(429, 'x', 'slow down', 12).retryAfter, 12, 'retry-after kept');
});

await test('errors: ProviderError carries status, code, message, kind, retry-after', () => {
  const err = new ProviderError(429, 'c', 'm', 'transient', 3);
  eq([err.status, err.code, err.message, err.kind, err.retryAfter], [429, 'c', 'm', 'transient', 3], 'fields');
  assert(err instanceof Error, 'is an Error');
});

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

function synthBrief() {
  return parseBrief(`pre ${JSON.stringify({
    schema: 'free-model-aggregator-01/providers/v1',
    providers: [
      { id: 'zeta', name: 'zeta', wire: 'openai', auth: 'bearer', key_env: 'K', key: 'k', path: '/p/zeta' },
      { id: 'alpha', name: 'alpha', wire: 'anthropic', auth: 'x-api-key', key_env: 'K', key: 'k', path: '/p/alpha' },
      { id: 'mid', name: 'mid', wire: 'gemini', auth: 'x-goog-api-key', key_env: 'K', key: 'k', path: '/p/mid' },
    ],
  })} mid ${JSON.stringify({
    schema: 'free-model-aggregator-01/ratings/v1',
    ratings: [
      { model: 'm-high', provider: 'zeta', rating_milli: 1500, context_length: 128000, tools: true, vision: true, json: true, free: true },
      { model: 'm-tie-b', provider: 'zeta', rating_milli: 1400, context_length: 128000, tools: true, vision: false, json: true, free: true },
      { model: 'm-tie-a', provider: 'alpha', rating_milli: 1400, context_length: 200000, tools: true, vision: true, json: true, free: true },
      { model: 'm-gem', provider: 'mid', rating_milli: 1450, context_length: 1000000, tools: false, vision: true, json: false, free: true },
      { model: 'm-paid', provider: 'zeta', rating_milli: 1600, context_length: 200000, tools: true, vision: true, json: true, free: false },
      { model: 'm-small', provider: 'zeta', rating_milli: 1300, context_length: 8000, tools: true, vision: true, json: true, free: true },
    ],
  })} post ${JSON.stringify({ schema: 'free-model-aggregator-01/workload/v1', manifest: [] })}`);
}

await test('router: rating order, tie-breaks, paid exclusion', () => {
  const brief = synthBrief();
  const plain = parseOpenAIRequest({ model: 'auto', messages: [{ role: 'user', content: 'hi' }] }, null);
  const ranked = rankCandidates(brief, new Map(), plain);
  eq(ranked.map((c) => c.row.model), ['m-high', 'm-gem', 'm-tie-a', 'm-tie-b', 'm-small'], 'order: rating desc, provider id, model id');
  assert(!ranked.some((c) => c.row.model === 'm-paid'), 'paid model never ranked');
});

await test('router: requirement filtering (tools, vision, min_context)', () => {
  const brief = synthBrief();
  const toolsReq = parseOpenAIRequest({ model: 'auto', messages: [{ role: 'user', content: 'hi' }], tools: [{ type: 'function', function: { name: 'f' } }] }, null);
  assert(!rankCandidates(brief, new Map(), toolsReq).some((c) => c.row.model === 'm-gem'), 'no-tools model excluded when tools required');
  const visionReq = parseOpenAIRequest({ model: 'auto', messages: [{ role: 'user', content: [{ type: 'text', text: 'x' }, { type: 'image_url', image_url: { url: 'https://x.invalid/i.png' } }] }] }, null);
  assert(!rankCandidates(brief, new Map(), visionReq).some((c) => c.row.model === 'm-tie-b'), 'no-vision model excluded when vision required');
});

await test('router: probe free flags override the brief; missing probes fall back to the brief', () => {
  const brief = synthBrief();
  const plain = parseOpenAIRequest({ model: 'auto', messages: [{ role: 'user', content: 'hi' }] }, null);
  const probed = new Map([['zeta/m-high', false]]);
  assert(!rankCandidates(brief, probed, plain).some((c) => c.row.model === 'm-high'), 'probe-paid model excluded');
  assert(rankCandidates(brief, new Map(), plain).some((c) => c.row.model === 'm-high'), 'no probe: brief free flag used');
});

await test('router: RouterState retires providers and pairs', () => {
  const state = new RouterState();
  const brief = synthBrief();
  const plain = parseOpenAIRequest({ model: 'auto', messages: [{ role: 'user', content: 'hi' }] }, null);
  const all = rankCandidates(brief, new Map(), plain);
  state.retireProvider('zeta');
  eq(state.usable(all).map((c) => c.provider.id), ['mid', 'alpha'], 'dead provider removed');
  state.retirePair('mid', 'm-gem');
  eq(state.usable(all).map((c) => c.row.model), ['m-tie-a'], 'bad pair removed');
  assert(state.isDead('zeta') && state.isBadPair('mid', 'm-gem'), 'predicates');
});

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

await test('catalog: parseListing reads all three wires and rejects malformed input', () => {
  const oai = parseListing('openai', { data: [{ id: 'a', free: true, context_length: 100 }, { id: 'b' }, { id: 'c', free: false }] });
  eq(oai, [{ id: 'a', free: true, context_length: 100 }, { id: 'b', free: false, context_length: null }, { id: 'c', free: false, context_length: null }], 'openai listing');
  const anth = parseListing('anthropic', { data: [{ id: 'x', free: true }] });
  eq(anth, [{ id: 'x', free: true, context_length: null }], 'anthropic listing');
  const gem = parseListing('gemini', { models: [{ name: 'models/g-1', free: true }, { name: 'models/g-2' }, { name: 'weird' }] });
  eq(gem, [{ id: 'g-1', free: true, context_length: null }, { id: 'g-2', free: false, context_length: null }, { id: 'weird', free: false, context_length: null }], 'gemini strips models/ prefix, keeps bare names');
  eq(parseListing('openai', null), null, 'null rejected');
  eq(parseListing('openai', 'text'), null, 'string rejected');
  eq(parseListing('openai', {}), null, 'object without a data array is malformed, falls back to the brief');
});

await test('catalog: buildCatalog probes live, falls back to the brief on probe failure', async () => {
  const brief = synthBrief();
  const provider = await withMockServer((req, res) => {
    if (req.url === '/p/zeta/v1/models') {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ data: [{ id: 'm-high', free: true, context_length: 128000 }] }));
    } else if (req.url === '/p/alpha/v1/models') {
      res.writeHead(500, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: 'boom' }));
    } else if (req.url === '/p/mid/v1beta/models') {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ models: [{ name: 'models/m-gem', free: true }] }));
    } else {
      res.writeHead(404);
      res.end();
    }
  }, async (base) => buildCatalog(base, brief));
  eq(provider.freeIds, ['m-high', 'm-tie-a', 'm-gem'], 'free ids: probed + fallback + gemini');
  eq(provider.freeByPair.get('zeta/m-high'), true, 'probed free');
  eq(provider.freeByPair.get('alpha/m-tie-a'), true, 'fallback to brief on probe failure');
  eq(provider.catalog.providers.find((p) => p.id === 'zeta').base_url, '/p/zeta', 'catalog keeps the path, not the full URL');
  eq(provider.json, JSON.stringify(provider.catalog), 'one serialization');
});

await test('catalog: --list prints the registered catalog byte-for-byte', async () => {
  const briefText = `pre ${JSON.stringify({
    schema: 'free-model-aggregator-01/providers/v1',
    providers: [{ id: 'zeta', name: 'zeta', wire: 'openai', auth: 'bearer', key_env: 'K', key: 'k', path: '/p/zeta' }],
  })} mid ${JSON.stringify({
    schema: 'free-model-aggregator-01/ratings/v1',
    ratings: [{ model: 'm-high', provider: 'zeta', rating_milli: 1500, context_length: 128000, tools: true, vision: true, json: true, free: true }],
  })} post ${JSON.stringify({ schema: 'free-model-aggregator-01/workload/v1', manifest: [] })}`;
  const briefFile = join(tmpdir(), `adv-brief-${process.pid}.txt`);
  writeFileSync(briefFile, briefText);
  try {
    await withMockServer((req, res) => {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ data: [{ id: 'm-high', free: true, context_length: 128000 }] }));
    }, async (base) => {
      const expected = (await buildCatalog(base, parseBrief(briefText))).json;
      const out = await new Promise((resolve, reject) => {
        const child = spawn('node', ['main.ts', '--list'], {
          cwd: new URL('..', import.meta.url).pathname,
          env: { ...process.env, MAKEFUN_PROMPT_FILE: briefFile, MAKEFUN_ENV_URL: base },
        });
        let stdout = '';
        let stderr = '';
        child.stdout.on('data', (d) => { stdout += d; });
        child.stderr.on('data', (d) => { stderr += d; });
        child.on('error', reject);
        child.on('close', (code) => resolve({ code, stdout, stderr }));
      });
      eq(out.code, 0, '--list exit 0');
      eq(out.stdout, `${expected}\n`, '--list bytes equal the registered catalog');
    });
  } finally {
    unlinkSync(briefFile);
  }
});

// ---------------------------------------------------------------------------
// Failover end-to-end: serveRequest against scripted mock providers
// ---------------------------------------------------------------------------

function failoverBrief() {
  return parseBrief(`pre ${JSON.stringify({
    schema: 'free-model-aggregator-01/providers/v1',
    providers: [
      { id: 'first', name: 'first', wire: 'openai', auth: 'bearer', key_env: 'K', key: 'k', path: '/p/first' },
      { id: 'second', name: 'second', wire: 'anthropic', auth: 'x-api-key', key_env: 'K', key: 'k', path: '/p/second' },
    ],
  })} mid ${JSON.stringify({
    schema: 'free-model-aggregator-01/ratings/v1',
    ratings: [
      { model: 'm-first', provider: 'first', rating_milli: 1500, context_length: 200000, tools: true, vision: true, json: true, free: true },
      { model: 'm-second', provider: 'second', rating_milli: 1400, context_length: 200000, tools: true, vision: true, json: true, free: true },
    ],
  })} post ${JSON.stringify({ schema: 'free-model-aggregator-01/workload/v1', manifest: [] })}`);
}

function failoverCtx(base, bundle) {
  return { envUrl: base, brief: failoverBrief(), bundle, router: new RouterState(), created: 1700000000 };
}

function failoverBundle() {
  return {
    catalog: { providers: [] },
    json: '{"providers":[]}',
    freeIds: ['m-first', 'm-second'],
    freeByPair: new Map([['first/m-first', true], ['second/m-second', true]]),
  };
}

const PLAIN_REQ = () => parseOpenAIRequest({ model: 'auto', messages: [{ role: 'user', content: 'hello' }] }, null);

await test('failover: transient refusals retry the same candidate; provider is not retired', async () => {
  let firstCalls = 0;
  await withMockServer((req, res) => {
    if (req.url.startsWith('/p/first/')) {
      firstCalls++;
      if (firstCalls <= 2) {
        res.writeHead(429, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: { message: 'first refused this call', type: 'rate_limit_error', code: 'x' } }));
      } else {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(openAIJson({ text: 'served after retries' }));
      }
    } else {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(anthropicJson({ text: 'second' }));
    }
  }, async (base) => {
    const ctx = failoverCtx(base, failoverBundle());
    const served = await serveRequest(ctx, PLAIN_REQ());
    assert(served !== null, 'served');
    eq(textOf(served.answer.events), 'served after retries', 'first provider served on retry 3');
    eq(firstCalls, 3, 'three attempts on the same candidate');
    assert(!ctx.router.isDead('first'), 'transient refusals never retire');
  });
});

await test('failover: permanent refusal retires the provider; next request skips it', async () => {
  let firstCalls = 0;
  await withMockServer((req, res) => {
    if (req.url.startsWith('/p/first/')) {
      firstCalls++;
      res.writeHead(429, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: { message: "first's free-tier budget is spent", type: 'rate_limit_error', code: 'insufficient_quota' } }));
    } else {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(anthropicJson({ text: 'second serves' }));
    }
  }, async (base) => {
    const ctx = failoverCtx(base, failoverBundle());
    const first = await serveRequest(ctx, PLAIN_REQ());
    eq(textOf(first.answer.events), 'second serves', 'failed over');
    assert(ctx.router.isDead('first'), 'provider retired');
    await serveRequest(ctx, PLAIN_REQ());
    eq(firstCalls, 1, 'retired provider never called again');
  });
});

await test('failover: 401 retires the provider; truncated stream fails over', async () => {
  let calls = 0;
  await withMockServer((req, res) => {
    if (req.url.startsWith('/p/first/')) {
      calls++;
      res.writeHead(401, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: { message: 'bad key', type: 'auth', code: 'auth' } }));
    } else {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(anthropicJson({ text: 'second' }));
    }
  }, async (base) => {
    const ctx = failoverCtx(base, failoverBundle());
    await serveRequest(ctx, PLAIN_REQ());
    assert(ctx.router.isDead('first'), '401 retires');
    eq(calls, 1, 'no retry on auth failure');
  });

  // Truncated stream: first provider's stream dies mid-way; second serves.
  await withMockServer((req, res) => {
    if (req.url.startsWith('/p/first/')) {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      res.end('data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"cut"},"finish_reason":null}]}\n\n');
    } else {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      res.end(anthropicStreamText({ text: 'whole answer' }));
    }
  }, async (base) => {
    const ctx = failoverCtx(base, failoverBundle());
    const streamReq = parseOpenAIRequest({ model: 'auto', messages: [{ role: 'user', content: 'hi' }], stream: true }, null);
    const served = await serveRequest(ctx, streamReq);
    assert(served !== null, 'served');
    eq(textOf(served.answer.events), 'whole answer', 'truncated stream failed over cleanly');
    const rendered = renderResponse(ctx, 'openai', served);
    assert(mirrorOpenAIStream(rendered.body).ok, 'rendered stream passes the mirror');
  });
});

await test('failover: usage-bearing answers win over usage-less ones; usage-less is the fallback', async () => {
  const noUsage = JSON.stringify({ id: 'x', object: 'chat.completion', created: 1, model: 'm', choices: [{ index: 0, message: { role: 'assistant', content: 'no usage here' }, finish_reason: 'stop' }] });
  await withMockServer((req, res) => {
    if (req.url.startsWith('/p/first/')) {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(noUsage);
    } else {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(anthropicJson({ text: 'with usage' }));
    }
  }, async (base) => {
    const ctx = failoverCtx(base, failoverBundle());
    const served = await serveRequest(ctx, PLAIN_REQ());
    eq(textOf(served.answer.events), 'with usage', 'usage-bearing answer preferred');
  });
  await withMockServer((req, res) => {
    if (req.url.startsWith('/p/first/')) {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(noUsage);
    } else {
      res.writeHead(429, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ type: 'error', error: { type: 'rate_limit_error', message: "second's free-tier budget is spent" } }));
    }
  }, async (base) => {
    const ctx = failoverCtx(base, failoverBundle());
    const served = await serveRequest(ctx, PLAIN_REQ());
    assert(served !== null, 'fallback served');
    eq(textOf(served.answer.events), 'no usage here', 'usage-less answer served when nothing better exists');
  });
});

await test('failover: all providers failing yields null (the 502 path)', async () => {
  await withMockServer((req, res) => {
    res.writeHead(503, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: { message: 'down', type: 'x', code: 'x' } }));
  }, async (base) => {
    const ctx = failoverCtx(base, failoverBundle());
    eq(await serveRequest(ctx, PLAIN_REQ()), null, 'null when nothing serves');
  });
});

await test('failover: concurrent requests never overlap a provider call (mutex)', async () => {
  let inFlight = 0;
  let maxInFlight = 0;
  await withMockServer((req, res) => {
    inFlight++;
    maxInFlight = Math.max(maxInFlight, inFlight);
    setTimeout(() => {
      inFlight--;
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(req.url.startsWith('/p/first/') ? openAIJson({ text: 'one' }) : anthropicJson({ text: 'two' }));
    }, 20);
  }, async (base) => {
    const ctx = failoverCtx(base, failoverBundle());
    const results = await Promise.all([serveRequest(ctx, PLAIN_REQ()), serveRequest(ctx, PLAIN_REQ()), serveRequest(ctx, PLAIN_REQ())]);
    assert(results.every((r) => r !== null), 'all served');
    eq(maxInFlight, 1, 'one provider call in flight at a time');
  });
});

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log(`\nadversarial harness: ${passed} passed, ${failed} failed`);
if (failed > 0) {
  console.log(`failures: ${failures.join(', ')}`);
  process.exit(1);
}
