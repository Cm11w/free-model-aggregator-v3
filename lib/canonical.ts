// lib/canonical.ts — the canonical request: one lossless form for all three
// wires. Every field the providers echo back (system text, user text, tool
// names, max_tokens, stop sequences, image bytes, JSON mode) is carried
// exactly, so translating a request to another wire never changes what the
// provider parses from it.
//
// Image bytes are sacred: a base64 payload is carried as the original string,
// never decoded and re-encoded, so the hash of the bytes is identical on
// every wire. A remote image URL is carried as the original string.

import { isRecord, numOrNull, strOrNull, stringList } from './util.ts';

export type ImageRef =
  | { kind: 'base64'; mime: string; data: string }
  | { kind: 'url'; url: string };

export type ContentPart =
  | { kind: 'text'; text: string }
  | { kind: 'image'; image: ImageRef }
  | { kind: 'toolCall'; id: string; name: string; argsText: string; argsJson: unknown }
  | { kind: 'toolResult'; id: string; name: string | null; content: string }
  | { kind: 'other'; wire: string; raw: unknown };

export interface CanonicalMessage {
  role: 'user' | 'assistant' | 'tool';
  parts: ContentPart[];
}

export interface CanonicalTool {
  name: string;
  description: string | null;
  /** The parameters/input schema, verbatim JSON from the client. */
  parameters: unknown;
}

export type ToolChoiceMode = 'auto' | 'any' | 'none' | 'specific' | 'unspecified';

export interface CanonicalRequest {
  sourceWire: 'openai' | 'anthropic';
  /** Joined system text ("" when the request carries no system prompt). */
  systemText: string;
  messages: CanonicalMessage[];
  tools: CanonicalTool[];
  toolChoice: { mode: ToolChoiceMode; name: string | null };
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  jsonMode: 'none' | 'json_object' | 'json_schema';
  /** The client's response_format object, verbatim, for json_schema. */
  jsonFormat: unknown;
  stream: boolean;
  /** The client's anthropic-version header, forwarded to anthropic-wire providers. */
  anthropicVersion: string | null;
}

/** Split a data: URL into mime + base64 payload, keeping the payload verbatim. */
function imageFromDataUrl(url: string): ImageRef | null {
  const m = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(url);
  if (m === null) return null;
  if (m[2] === ';base64') return { kind: 'base64', mime: m[1] === '' ? 'image/png' : m[1], data: m[3] };
  return null;
}

function imageFromUrl(url: string): ImageRef {
  return imageFromDataUrl(url) ?? { kind: 'url', url };
}

/** Text of a content value: a string, or the text blocks joined with "\n". */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function parseToolArgs(args: unknown): { argsText: string; argsJson: unknown } {
  if (typeof args === 'string') {
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(args);
    } catch {
      parsed = null;
    }
    return { argsText: args, argsJson: parsed };
  }
  if (args !== undefined) return { argsText: JSON.stringify(args), argsJson: args };
  return { argsText: '{}', argsJson: {} };
}

/** Parse OpenAI content parts (text / image_url / input_audio / file / refusal). */
function openAIParts(content: unknown): ContentPart[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (content === null) return [];
  if (!Array.isArray(content)) return [];
  const parts: ContentPart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      parts.push({ kind: 'text', text: block.text });
    } else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      parts.push({ kind: 'image', image: imageFromUrl(block.image_url.url) });
    } else if (block.type === 'input_audio' && isRecord(block.input_audio) && typeof block.input_audio.data === 'string') {
      parts.push({ kind: 'other', wire: 'openai', raw: block });
    } else {
      parts.push({ kind: 'other', wire: 'openai', raw: block });
    }
  }
  return parts;
}

function openAIToolChoice(raw: unknown): { mode: ToolChoiceMode; name: string | null } {
  if (raw === 'auto' || raw === undefined) return { mode: 'auto', name: null };
  if (raw === 'required') return { mode: 'any', name: null };
  if (raw === 'none') return { mode: 'none', name: null };
  if (isRecord(raw) && raw.type === 'function' && isRecord(raw.function) && typeof raw.function.name === 'string') {
    return { mode: 'specific', name: raw.function.name };
  }
  return { mode: 'auto', name: null };
}

export function parseOpenAIRequest(body: Record<string, unknown>, anthropicVersion: string | null): CanonicalRequest {
  const systemParts: string[] = [];
  const messages: CanonicalMessage[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system' || role === 'developer') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'tool',
        parts: [
          {
            kind: 'toolResult',
            id: strOrNull(raw.tool_call_id) ?? '',
            name: null,
            content: typeof raw.content === 'string' ? raw.content : blocksText(raw.content),
          },
        ],
      });
      continue;
    }
    if (role === 'assistant') {
      const parts = openAIParts(raw.content);
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
          const args = parseToolArgs(call.function.arguments);
          parts.push({ kind: 'toolCall', id: strOrNull(call.id) ?? '', name: call.function.name, argsText: args.argsText, argsJson: args.argsJson });
        }
      }
      messages.push({ role: 'assistant', parts });
      continue;
    }
    // 'user' and anything else defaults to a user turn.
    messages.push({ role: 'user', parts: openAIParts(raw.content) });
  }

  const tools: CanonicalTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: strOrNull(raw.function.description),
        parameters: raw.function.parameters ?? null,
      });
    }
  }

  let jsonMode: CanonicalRequest['jsonMode'] = 'none';
  let jsonFormat: unknown = null;
  if (isRecord(body.response_format)) {
    if (body.response_format.type === 'json_object') jsonMode = 'json_object';
    else if (body.response_format.type === 'json_schema') jsonMode = 'json_schema';
    jsonFormat = body.response_format;
  }

  return {
    sourceWire: 'openai',
    systemText: systemParts.join('\n'),
    messages,
    tools,
    toolChoice: openAIToolChoice(body.tool_choice),
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    jsonMode,
    jsonFormat,
    stream: body.stream === true,
    anthropicVersion,
  };
}

/** Parse Anthropic content blocks (text / image / tool_use / tool_result / others). */
function anthropicParts(content: unknown): ContentPart[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const parts: ContentPart[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      parts.push({ kind: 'text', text: block.text });
    } else if (block.type === 'image' && isRecord(block.source)) {
      const src = block.source;
      if (src.type === 'base64' && typeof src.data === 'string') {
        parts.push({ kind: 'image', image: { kind: 'base64', mime: strOrNull(src.media_type) ?? 'image/png', data: src.data } });
      } else if (src.type === 'url' && typeof src.url === 'string') {
        parts.push({ kind: 'image', image: { kind: 'url', url: src.url } });
      } else {
        parts.push({ kind: 'other', wire: 'anthropic', raw: block });
      }
    } else if (block.type === 'tool_use' && typeof block.name === 'string') {
      const args = parseToolArgs(block.input);
      parts.push({ kind: 'toolCall', id: strOrNull(block.id) ?? '', name: block.name, argsText: args.argsText, argsJson: args.argsJson });
    } else if (block.type === 'tool_result') {
      parts.push({
        kind: 'toolResult',
        id: strOrNull(block.tool_use_id) ?? '',
        name: null,
        content: typeof block.content === 'string' ? block.content : blocksText(block.content),
      });
    } else {
      parts.push({ kind: 'other', wire: 'anthropic', raw: block });
    }
  }
  return parts;
}

function anthropicToolChoice(raw: unknown): { mode: ToolChoiceMode; name: string | null } {
  if (!isRecord(raw)) return { mode: 'auto', name: null };
  if (raw.type === 'auto') return { mode: 'auto', name: null };
  if (raw.type === 'any') return { mode: 'any', name: null };
  if (raw.type === 'none') return { mode: 'none', name: null };
  if (raw.type === 'tool' && typeof raw.name === 'string') return { mode: 'specific', name: raw.name };
  return { mode: 'auto', name: null };
}

export function parseAnthropicRequest(body: Record<string, unknown>, anthropicVersion: string | null): CanonicalRequest {
  const systemText = blocksText(body.system);
  const messages: CanonicalMessage[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const parts = anthropicParts(raw.content);
    if (raw.role === 'assistant') {
      messages.push({ role: 'assistant', parts });
    } else {
      // Anthropic user turns may carry tool_result blocks; keep the role and
      // let the translators place tool results where each wire wants them.
      const hasOnlyResults = parts.length > 0 && parts.every((p) => p.kind === 'toolResult');
      messages.push({ role: hasOnlyResults ? 'tool' : 'user', parts });
    }
  }

  const tools: CanonicalTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: strOrNull(raw.description),
        parameters: raw.input_schema ?? null,
      });
    }
  }

  return {
    sourceWire: 'anthropic',
    systemText,
    messages,
    tools,
    toolChoice: anthropicToolChoice(body.tool_choice),
    maxTokens: numOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    jsonMode: 'none',
    jsonFormat: null,
    stream: body.stream === true,
    anthropicVersion,
  };
}

/** The request's requirements, derived from what it actually carries. */
export function requirementsOf(req: CanonicalRequest): { tools: boolean; vision: boolean; json: boolean } {
  let vision = false;
  for (const m of req.messages) {
    for (const p of m.parts) if (p.kind === 'image') vision = true;
  }
  return { tools: req.tools.length > 0, vision, json: req.jsonMode !== 'none' };
}
