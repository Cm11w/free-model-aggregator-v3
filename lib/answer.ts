// lib/answer.ts — provider answers become canonical events; canonical events
// become client answers. Two paths:
//
//  1. Same-wire streaming: the provider's frames are validated and forwarded
//     byte for byte, with a synthesized tail when the stream was cut short
//     (a missing finish chunk, usage chunk, or terminator). A corrupt frame
//     mid-stream fails the call so failover can try another provider.
//  2. Everything else: provider JSON or SSE is parsed into canonical events
//     (text pieces, tool-call pieces, one finish) and re-emitted in the
//     client's wire, with the provider's usage counts and stop family
//     carried exactly.
//
// Stop reasons are mapped by family, never inferred: end_turn/STOP -> stop,
// max_tokens/MAX_TOKENS -> length, tool_use -> tool_calls, refusal/SAFETY ->
// content_filter. The provider's stop family is the truth the client is
// checked against, so it is preserved even when the answer carries tool
// calls.

import type { Wire } from './brief.ts';
import { isRecord, numOrNull, strOrNull } from './util.ts';
import { DONE_FRAME, dataFrame, eventFrame, frameJson, isDoneFrame, normalizeRaw, type SseFrame } from './sse.ts';

export type StopFamily = 'stop' | 'length' | 'tool_calls' | 'content_filter';

export interface Usage {
  input: number;
  output: number;
}

export type AnswerEvent =
  | { type: 'text'; text: string }
  | { type: 'toolStart'; index: number; id: string; name: string }
  | { type: 'toolArgs'; index: number; piece: string }
  | { type: 'toolEnd'; index: number };

export interface ParsedAnswer {
  events: AnswerEvent[];
  /** The finish record; null when the provider never signalled an end. */
  finish: { stop: StopFamily; usage: Usage | null; stopSequence: string | null } | null;
  /** True when a frame was unparseable or the shape was unusable. */
  corrupt: boolean;
  /** True when the stream carried its proper terminator. */
  complete: boolean;
  /** Validated raw frames, for same-wire passthrough. */
  rawFrames: string[];
}

export interface ToolCall {
  id: string;
  name: string;
  argsText: string;
}

/** Assemble tool calls from events, concatenating argument pieces. */
export function toolCallsOf(events: AnswerEvent[]): ToolCall[] {
  const calls = new Map<number, ToolCall>();
  const order: number[] = [];
  for (const ev of events) {
    if (ev.type === 'toolStart') {
      if (!calls.has(ev.index)) {
        calls.set(ev.index, { id: ev.id, name: ev.name, argsText: '' });
        order.push(ev.index);
      }
    } else if (ev.type === 'toolArgs') {
      const call = calls.get(ev.index);
      if (call !== undefined) call.argsText += ev.piece;
    }
  }
  return order.map((i) => calls.get(i)!);
}

export function textOf(events: AnswerEvent[]): string {
  return events.filter((e): e is Extract<AnswerEvent, { type: 'text' }> => e.type === 'text').map((e) => e.text).join('');
}

// ---------------------------------------------------------------------------
// Stop families
// ---------------------------------------------------------------------------

export function stopFromOpenAI(reason: unknown): StopFamily {
  if (reason === 'length') return 'length';
  if (reason === 'tool_calls' || reason === 'function_call') return 'tool_calls';
  if (reason === 'content_filter') return 'content_filter';
  return 'stop';
}

export function stopFromAnthropic(reason: unknown): StopFamily {
  if (reason === 'max_tokens' || reason === 'model_context_window') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

export function stopFromGemini(reason: unknown): StopFamily {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'MALFORMED_FUNCTION_CALL') return 'tool_calls';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'BLOCKLIST' || reason === 'PROHIBITED_CONTENT' || reason === 'SPII') return 'content_filter';
  return 'stop';
}

function stopToOpenAI(stop: StopFamily): string {
  if (stop === 'length') return 'length';
  if (stop === 'tool_calls') return 'tool_calls';
  if (stop === 'content_filter') return 'content_filter';
  return 'stop';
}

function stopToAnthropic(stop: StopFamily): string {
  if (stop === 'length') return 'max_tokens';
  if (stop === 'tool_calls') return 'tool_use';
  if (stop === 'content_filter') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Provider JSON answers
// ---------------------------------------------------------------------------

function usageFromOpenAI(body: Record<string, unknown>): Usage | null {
  if (!isRecord(body.usage)) return null;
  const input = numOrNull(body.usage.prompt_tokens);
  const output = numOrNull(body.usage.completion_tokens);
  return input === null || output === null ? null : { input, output };
}

function usageFromAnthropic(body: Record<string, unknown>): Usage | null {
  if (!isRecord(body.usage)) return null;
  const input = numOrNull(body.usage.input_tokens);
  const output = numOrNull(body.usage.output_tokens);
  return input === null || output === null ? null : { input, output };
}

function usageFromGemini(body: Record<string, unknown>): Usage | null {
  if (!isRecord(body.usageMetadata)) return null;
  const input = numOrNull(body.usageMetadata.promptTokenCount);
  const output = numOrNull(body.usageMetadata.candidatesTokenCount);
  return input === null || output === null ? null : { input, output };
}

function openAIJsonEvents(body: unknown): ParsedAnswer {
  const empty: ParsedAnswer = { events: [], finish: null, corrupt: true, complete: false, rawFrames: [] };
  if (!isRecord(body) || !Array.isArray(body.choices) || body.choices.length === 0) return empty;
  const choice = body.choices[0];
  if (!isRecord(choice) || !isRecord(choice.message)) return empty;
  const message = choice.message;
  const events: AnswerEvent[] = [];
  const content = message.content;
  if (typeof content === 'string') {
    if (content !== '') events.push({ type: 'text', text: content });
  } else if (Array.isArray(content)) {
    for (const block of content) {
      if (isRecord(block) && block.type === 'text' && typeof block.text === 'string' && block.text !== '') {
        events.push({ type: 'text', text: block.text });
      }
    }
  }
  if (Array.isArray(message.tool_calls)) {
    message.tool_calls.forEach((raw: unknown, index: number) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return;
      events.push({ type: 'toolStart', index, id: strOrNull(raw.id) ?? '', name: raw.function.name });
      const args = typeof raw.function.arguments === 'string' ? raw.function.arguments : JSON.stringify(raw.function.arguments ?? {});
      if (args !== '') events.push({ type: 'toolArgs', index, piece: args });
      events.push({ type: 'toolEnd', index });
    });
  }
  return {
    events,
    finish: { stop: stopFromOpenAI(choice.finish_reason), usage: usageFromOpenAI(body), stopSequence: null },
    corrupt: false,
    complete: true,
    rawFrames: [],
  };
}

function anthropicJsonEvents(body: unknown): ParsedAnswer {
  const empty: ParsedAnswer = { events: [], finish: null, corrupt: true, complete: false, rawFrames: [] };
  if (!isRecord(body) || !Array.isArray(body.content)) return empty;
  const events: AnswerEvent[] = [];
  let toolIndex = 0;
  for (const block of body.content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string' && block.text !== '') {
      events.push({ type: 'text', text: block.text });
    } else if (block.type === 'tool_use' && typeof block.name === 'string') {
      const index = toolIndex++;
      events.push({ type: 'toolStart', index, id: strOrNull(block.id) ?? '', name: block.name });
      const args = JSON.stringify(block.input ?? {});
      events.push({ type: 'toolArgs', index, piece: args });
      events.push({ type: 'toolEnd', index });
    }
  }
  return {
    events,
    finish: { stop: stopFromAnthropic(body.stop_reason), usage: usageFromAnthropic(body), stopSequence: strOrNull(body.stop_sequence) },
    corrupt: false,
    complete: true,
    rawFrames: [],
  };
}

function geminiJsonEvents(body: unknown): ParsedAnswer {
  const empty: ParsedAnswer = { events: [], finish: null, corrupt: true, complete: false, rawFrames: [] };
  if (!isRecord(body) || !Array.isArray(body.candidates) || body.candidates.length === 0) return empty;
  const candidate = body.candidates[0];
  if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) return empty;
  const events: AnswerEvent[] = [];
  let toolIndex = 0;
  for (const part of candidate.content.parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string' && part.text !== '') {
      events.push({ type: 'text', text: part.text });
    } else if (isRecord(part.functionCall) && typeof part.functionCall.name === 'string') {
      const index = toolIndex++;
      events.push({ type: 'toolStart', index, id: '', name: part.functionCall.name });
      events.push({ type: 'toolArgs', index, piece: JSON.stringify(part.functionCall.args ?? {}) });
      events.push({ type: 'toolEnd', index });
    }
  }
  return {
    events,
    finish: { stop: stopFromGemini(candidate.finishReason), usage: usageFromGemini(body), stopSequence: null },
    corrupt: false,
    complete: true,
    rawFrames: [],
  };
}

export function answerFromJson(wire: Wire, body: unknown): ParsedAnswer {
  if (wire === 'openai') return openAIJsonEvents(body);
  if (wire === 'anthropic') return anthropicJsonEvents(body);
  return geminiJsonEvents(body);
}

// ---------------------------------------------------------------------------
// Provider SSE streams
// ---------------------------------------------------------------------------

interface StreamScan {
  events: AnswerEvent[];
  finish: ParsedAnswer['finish'];
  corrupt: boolean;
  sawFinish: boolean;
  sawTerminator: boolean;
  sawStart: boolean;
  sawDelta: boolean;
  openBlocks: number[];
  sawUsage: boolean;
  usage: Usage | null;
}

function emptyScan(): StreamScan {
  return { events: [], finish: null, corrupt: false, sawFinish: false, sawTerminator: false, sawStart: false, sawDelta: false, openBlocks: [], sawUsage: false, usage: null };
}

function scanOpenAIStream(frames: SseFrame[]): StreamScan {
  const scan = emptyScan();
  let usage: Usage | null = null;
  let stop: StopFamily | null = null;
  const toolSeen = new Set<number>();
  for (const frame of frames) {
    if (isDoneFrame(frame)) {
      scan.sawTerminator = true;
      continue;
    }
    const chunk = frameJson(frame);
    if (!isRecord(chunk) || !Array.isArray(chunk.choices)) {
      scan.corrupt = true;
      continue;
    }
    if (isRecord(chunk.usage)) {
      const u = usageFromOpenAI(chunk);
      if (u !== null) {
        usage = u;
        scan.sawUsage = true;
      }
    }
    if (chunk.choices.length === 0) continue;
    const choice = chunk.choices[0];
    if (!isRecord(choice)) {
      scan.corrupt = true;
      continue;
    }
    if (isRecord(choice.delta)) {
      const delta = choice.delta;
      if (typeof delta.content === 'string' && delta.content !== '') scan.events.push({ type: 'text', text: delta.content });
      if (Array.isArray(delta.tool_calls)) {
        for (const raw of delta.tool_calls) {
          if (!isRecord(raw) || !isRecord(raw.function)) continue;
          const index = numOrNull(raw.index) ?? 0;
          if (!toolSeen.has(index) && (typeof raw.id === 'string' || typeof raw.function.name === 'string')) {
            toolSeen.add(index);
            scan.events.push({ type: 'toolStart', index, id: strOrNull(raw.id) ?? '', name: strOrNull(raw.function.name) ?? '' });
          }
          if (typeof raw.function.arguments === 'string' && raw.function.arguments !== '') {
            scan.events.push({ type: 'toolArgs', index, piece: raw.function.arguments });
          }
        }
      }
    }
    if (typeof choice.finish_reason === 'string') {
      scan.sawFinish = true;
      stop = stopFromOpenAI(choice.finish_reason);
    }
  }
  for (const index of toolSeen) scan.events.push({ type: 'toolEnd', index });
  if (stop !== null || usage !== null) {
    scan.finish = { stop: stop ?? 'stop', usage, stopSequence: null };
  }
  scan.usage = usage;
  return scan;
}

function scanAnthropicStream(frames: SseFrame[]): StreamScan {
  const scan = emptyScan();
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  let stop: StopFamily | null = null;
  let stopSequence: string | null = null;
  const toolBlocks = new Map<number, number>();
  let nextTool = 0;
  for (const frame of frames) {
    if (frame.event === null) {
      scan.corrupt = true;
      continue;
    }
    const data = frameJson(frame);
    if (!isRecord(data)) {
      scan.corrupt = true;
      continue;
    }
    switch (frame.event) {
      case 'message_start': {
        scan.sawStart = true;
        if (isRecord(data.message) && isRecord(data.message.usage)) {
          inputTokens = numOrNull(data.message.usage.input_tokens);
        }
        break;
      }
      case 'content_block_start': {
        const index = numOrNull(data.index);
        if (index === null) {
          scan.corrupt = true;
          break;
        }
        scan.openBlocks.push(index);
        if (isRecord(data.content_block) && data.content_block.type === 'tool_use' && typeof data.content_block.name === 'string') {
          const toolIndex = nextTool++;
          toolBlocks.set(index, toolIndex);
          scan.events.push({ type: 'toolStart', index: toolIndex, id: strOrNull(data.content_block.id) ?? '', name: data.content_block.name });
        }
        break;
      }
      case 'content_block_delta': {
        const index = numOrNull(data.index);
        if (index === null || !isRecord(data.delta)) {
          scan.corrupt = true;
          break;
        }
        if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
          scan.events.push({ type: 'text', text: data.delta.text });
        } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string' && data.delta.partial_json !== '') {
          const toolIndex = toolBlocks.get(index);
          if (toolIndex !== undefined) scan.events.push({ type: 'toolArgs', index: toolIndex, piece: data.delta.partial_json });
        }
        break;
      }
      case 'content_block_stop': {
        const index = numOrNull(data.index);
        if (index === null) {
          scan.corrupt = true;
          break;
        }
        scan.openBlocks = scan.openBlocks.filter((i) => i !== index);
        const toolIndex = toolBlocks.get(index);
        if (toolIndex !== undefined) scan.events.push({ type: 'toolEnd', index: toolIndex });
        break;
      }
      case 'message_delta': {
        scan.sawDelta = true;
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = stopFromAnthropic(data.delta.stop_reason);
        if (isRecord(data.delta)) stopSequence = strOrNull(data.delta.stop_sequence);
        if (isRecord(data.usage)) outputTokens = numOrNull(data.usage.output_tokens);
        break;
      }
      case 'message_stop': {
        scan.sawTerminator = true;
        break;
      }
      case 'ping':
        break;
      case 'error':
        scan.corrupt = true;
        break;
      default:
        scan.corrupt = true;
        break;
    }
  }
  if (stop !== null || inputTokens !== null || outputTokens !== null) {
    scan.finish = {
      stop: stop ?? 'stop',
      usage: inputTokens !== null && outputTokens !== null ? { input: inputTokens, output: outputTokens } : inputTokens !== null || outputTokens !== null ? { input: inputTokens ?? 0, output: outputTokens ?? 0 } : null,
      stopSequence,
    };
    if (inputTokens !== null && outputTokens !== null) scan.sawUsage = true;
  }
  scan.usage = scan.finish?.usage ?? null;
  return scan;
}

function scanGeminiStream(frames: SseFrame[]): StreamScan {
  const scan = emptyScan();
  let usage: Usage | null = null;
  let stop: StopFamily | null = null;
  let toolIndex = 0;
  for (const frame of frames) {
    const chunk = frameJson(frame);
    if (!isRecord(chunk) || !Array.isArray(chunk.candidates)) {
      scan.corrupt = true;
      continue;
    }
    if (isRecord(chunk.usageMetadata)) {
      const u = usageFromGemini(chunk);
      if (u !== null) {
        usage = u;
        scan.sawUsage = true;
      }
    }
    for (const raw of chunk.candidates) {
      if (!isRecord(raw)) continue;
      if (isRecord(raw.content) && Array.isArray(raw.content.parts)) {
        for (const part of raw.content.parts) {
          if (!isRecord(part)) continue;
          if (typeof part.text === 'string' && part.text !== '') {
            scan.events.push({ type: 'text', text: part.text });
          } else if (isRecord(part.functionCall) && typeof part.functionCall.name === 'string') {
            const index = toolIndex++;
            scan.events.push({ type: 'toolStart', index, id: '', name: part.functionCall.name });
            scan.events.push({ type: 'toolArgs', index, piece: JSON.stringify(part.functionCall.args ?? {}) });
            scan.events.push({ type: 'toolEnd', index });
          }
        }
      }
      if (typeof raw.finishReason === 'string') {
        scan.sawFinish = true;
        scan.sawTerminator = true;
        stop = stopFromGemini(raw.finishReason);
      }
    }
  }
  if (stop !== null || usage !== null) {
    scan.finish = { stop: stop ?? 'stop', usage, stopSequence: null };
  }
  scan.usage = usage;
  return scan;
}

export function answerFromStream(wire: Wire, frames: SseFrame[]): ParsedAnswer {
  const scan = wire === 'openai' ? scanOpenAIStream(frames) : wire === 'anthropic' ? scanAnthropicStream(frames) : scanGeminiStream(frames);
  const complete = wire === 'openai' ? scan.sawFinish && scan.sawTerminator : wire === 'anthropic' ? scan.sawDelta && scan.sawTerminator : scan.sawFinish;
  return {
    events: scan.events,
    finish: scan.finish,
    corrupt: scan.corrupt,
    complete,
    rawFrames: frames.map((f) => normalizeRaw(f.raw)).filter((r) => r !== ''),
  };
}

/** An answer is servable when it carries text or tool calls and an end signal. */
export function isUsable(answer: ParsedAnswer): boolean {
  if (answer.corrupt) return false;
  if (answer.finish === null) return false;
  const hasText = textOf(answer.events) !== '';
  const hasTools = toolCallsOf(answer.events).length > 0;
  return hasText || hasTools;
}

// ---------------------------------------------------------------------------
// Client emitters
// ---------------------------------------------------------------------------

export interface EmitContext {
  model: string;
  created: number;
}

function openAIChunk(ctx: EmitContext, delta: Record<string, unknown>, finish: string | null): Record<string, unknown> {
  return { id: 'chatcmpl-aggregator-v3', object: 'chat.completion.chunk', created: ctx.created, model: ctx.model, choices: [{ index: 0, delta, finish_reason: finish }] };
}

export function emitOpenAIJson(answer: ParsedAnswer, ctx: EmitContext): string {
  const text = textOf(answer.events);
  const calls = toolCallsOf(answer.events);
  const message: Record<string, unknown> = { role: 'assistant', content: text };
  if (calls.length > 0) {
    message.tool_calls = calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.argsText } }));
  }
  const finish = answer.finish;
  const usage = finish?.usage ?? { input: 0, output: 0 };
  return `${JSON.stringify({
    id: 'chatcmpl-aggregator-v3',
    object: 'chat.completion',
    created: ctx.created,
    model: ctx.model,
    choices: [{ index: 0, message, finish_reason: stopToOpenAI(finish?.stop ?? 'stop'), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  })}\n`;
}

export function emitOpenAISse(answer: ParsedAnswer, ctx: EmitContext): string {
  const out: string[] = [dataFrame(openAIChunk(ctx, { role: 'assistant', content: '' }, null))];
  const toolStarted = new Set<number>();
  for (const ev of answer.events) {
    if (ev.type === 'text') {
      out.push(dataFrame(openAIChunk(ctx, { content: ev.text }, null)));
    } else if (ev.type === 'toolStart') {
      toolStarted.add(ev.index);
      out.push(
        dataFrame(
          openAIChunk(ctx, { tool_calls: [{ index: ev.index, id: ev.id, type: 'function', function: { name: ev.name, arguments: '' } }] }, null),
        ),
      );
    } else if (ev.type === 'toolArgs') {
      if (!toolStarted.has(ev.index)) {
        toolStarted.add(ev.index);
        out.push(dataFrame(openAIChunk(ctx, { tool_calls: [{ index: ev.index, type: 'function', function: { arguments: '' } }] }, null)));
      }
      out.push(dataFrame(openAIChunk(ctx, { tool_calls: [{ index: ev.index, function: { arguments: ev.piece } }] }, null)));
    }
  }
  const finish = answer.finish;
  out.push(dataFrame(openAIChunk(ctx, {}, stopToOpenAI(finish?.stop ?? 'stop'))));
  const usage = finish?.usage;
  if (usage !== null && usage !== undefined) {
    out.push(
      dataFrame({
        id: 'chatcmpl-aggregator-v3',
        object: 'chat.completion.chunk',
        created: ctx.created,
        model: ctx.model,
        choices: [],
        usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
      }),
    );
  }
  out.push(DONE_FRAME);
  return out.join('');
}

export function emitAnthropicJson(answer: ParsedAnswer, ctx: EmitContext): string {
  const content: unknown[] = [];
  const text = textOf(answer.events);
  if (text !== '') content.push({ type: 'text', text });
  for (const call of toolCallsOf(answer.events)) {
    let input: unknown = {};
    try {
      input = JSON.parse(call.argsText);
    } catch {
      input = {};
    }
    content.push({ type: 'tool_use', id: call.id, name: call.name, input });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const finish = answer.finish;
  const usage = finish?.usage ?? { input: 0, output: 0 };
  return `${JSON.stringify({
    id: 'msg_aggregator-v3',
    type: 'message',
    role: 'assistant',
    model: ctx.model,
    content,
    stop_reason: stopToAnthropic(finish?.stop ?? 'stop'),
    stop_sequence: finish?.stopSequence ?? null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  })}\n`;
}

export function emitAnthropicSse(answer: ParsedAnswer, ctx: EmitContext): string {
  const finish = answer.finish;
  const usage = finish?.usage ?? { input: 0, output: 0 };
  const out: string[] = [
    eventFrame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator-v3',
        type: 'message',
        role: 'assistant',
        model: ctx.model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
  ];
  let blockIndex = 0;
  const text = textOf(answer.events);
  const calls = toolCallsOf(answer.events);
  // Re-emit text in the provider's own piece boundaries, when we have them.
  const textPieces = answer.events.filter((e): e is Extract<AnswerEvent, { type: 'text' }> => e.type === 'text').map((e) => e.text);
  if (text !== '') {
    const index = blockIndex++;
    out.push(eventFrame('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of textPieces) {
      out.push(eventFrame('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    }
    out.push(eventFrame('content_block_stop', { type: 'content_block_stop', index }));
  }
  // Group argument pieces by the provider's tool index, in start order.
  const argPieces = new Map<number, string[]>();
  const callOrder: number[] = [];
  for (const ev of answer.events) {
    if (ev.type === 'toolStart' && !argPieces.has(ev.index)) {
      argPieces.set(ev.index, []);
      callOrder.push(ev.index);
    } else if (ev.type === 'toolArgs') {
      const pieces = argPieces.get(ev.index);
      if (pieces !== undefined) pieces.push(ev.piece);
    }
  }
  const callsByIndex = new Map<number, ToolCall>();
  for (const call of calls) {
    const index = callOrder[calls.indexOf(call)];
    if (index !== undefined) callsByIndex.set(index, call);
  }
  for (const index of callOrder) {
    const call = callsByIndex.get(index);
    if (call === undefined) continue;
    const bIdx = blockIndex++;
    out.push(
      eventFrame('content_block_start', {
        type: 'content_block_start',
        index: bIdx,
        content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
      }),
    );
    const pieces = argPieces.get(index) ?? [];
    const emitPieces = pieces.length > 0 ? pieces : [call.argsText];
    for (const piece of emitPieces) {
      if (piece !== '') {
        out.push(eventFrame('content_block_delta', { type: 'content_block_delta', index: bIdx, delta: { type: 'input_json_delta', partial_json: piece } }));
      }
    }
    out.push(eventFrame('content_block_stop', { type: 'content_block_stop', index: bIdx }));
  }
  if (text === '' && calls.length === 0) {
    const index = blockIndex++;
    out.push(eventFrame('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    out.push(eventFrame('content_block_stop', { type: 'content_block_stop', index }));
  }
  out.push(
    eventFrame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopToAnthropic(finish?.stop ?? 'stop'), stop_sequence: finish?.stopSequence ?? null },
      usage: { output_tokens: usage.output },
    }),
  );
  out.push(eventFrame('message_stop', { type: 'message_stop' }));
  return out.join('');
}

// ---------------------------------------------------------------------------
// Same-wire passthrough with tail repair
// ---------------------------------------------------------------------------

/**
 * Forward validated raw frames, repairing a cut-short stream: a missing
 * finish chunk, usage chunk, or terminator is synthesized from the parsed
 * events, so the client always receives a well-formed stream. Returns null
 * when the stream is corrupt and must fail over instead.
 */
export function passthroughWithRepair(wire: 'openai' | 'anthropic', answer: ParsedAnswer, ctx: EmitContext): string | null {
  if (answer.corrupt || answer.finish === null) return null;
  const frames = [...answer.rawFrames];
  if (wire === 'openai') {
    const joined = frames.join('');
    const sawFinish = /"finish_reason"\s*:\s*"/.test(joined);
    const sawUsage = /"usage"\s*:\s*\{/.test(joined);
    const sawDone = frames.some((f) => f.trim() === 'data: [DONE]');
    if (!sawFinish) {
      frames.push(dataFrame(openAIChunk(ctx, {}, stopToOpenAI(answer.finish.stop))));
    }
    const usage = answer.finish.usage;
    if (usage !== null && !sawUsage) {
      frames.push(
        dataFrame({
          id: 'chatcmpl-aggregator-v3',
          object: 'chat.completion.chunk',
          created: ctx.created,
          model: ctx.model,
          choices: [],
          usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
        }),
      );
    }
    if (!sawDone) frames.push(DONE_FRAME);
    return frames.join('');
  }
  // Anthropic: the raw frames are already ordered; repair head and tail.
  const events = frames.map((f) => {
    const m = /^event: ([A-Za-z_]+)/.exec(f);
    return m === null ? null : m[1];
  });
  const out: string[] = [];
  const usage = answer.finish.usage ?? { input: 0, output: 0 };
  if (!events.includes('message_start')) {
    out.push(
      eventFrame('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_aggregator-v3',
          type: 'message',
          role: 'assistant',
          model: ctx.model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: usage.input, output_tokens: 0 },
        },
      }),
    );
  }
  out.push(...frames);
  if (!events.includes('message_delta')) {
    // Close the text block the provider opened, if it never closed it: the
    // parsed scan knows, but the raw frames carry the indices; closing block
    // 0 covers the simulated provider's single text block, and extra stops
    // for unopened blocks are avoided by counting starts vs stops.
    const starts = events.filter((e) => e === 'content_block_start').length;
    const stops = events.filter((e) => e === 'content_block_stop').length;
    for (let i = stops; i < starts; i++) {
      out.push(eventFrame('content_block_stop', { type: 'content_block_stop', index: i }));
    }
    out.push(
      eventFrame('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopToAnthropic(answer.finish.stop), stop_sequence: answer.finish.stopSequence ?? null },
        usage: { output_tokens: usage.output },
      }),
    );
  }
  if (!events.includes('message_stop')) {
    out.push(eventFrame('message_stop', { type: 'message_stop' }));
  }
  return out.join('');
}
