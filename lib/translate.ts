// lib/translate.ts — canonical request to provider wire bodies. The rule
// throughout: the provider must parse exactly what the client sent. System
// text joins identically, the last user turn's text and image bytes survive
// byte for byte, tool names keep their order, max_tokens and stop sequences
// cross unchanged, and JSON mode is carried only on wires that can express
// it (OpenAI response_format, Gemini responseMimeType — never Anthropic).

import type { CanonicalRequest, CanonicalTool, ContentPart, ImageRef } from './canonical.ts';
import { cloneJson } from './util.ts';

function imageToOpenAI(image: ImageRef): Record<string, unknown> {
  if (image.kind === 'base64') {
    return { type: 'image_url', image_url: { url: `data:${image.mime};base64,${image.data}` } };
  }
  return { type: 'image_url', image_url: { url: image.url } };
}

function imageToAnthropic(image: ImageRef): Record<string, unknown> {
  if (image.kind === 'base64') {
    return { type: 'image', source: { type: 'base64', media_type: image.mime, data: image.data } };
  }
  return { type: 'image', source: { type: 'url', url: image.url } };
}

function imageToGemini(image: ImageRef): Record<string, unknown> {
  if (image.kind === 'base64') {
    return { inlineData: { mimeType: image.mime, data: image.data } };
  }
  return { fileData: { fileUri: image.url } };
}

/** Plain text of a message's text parts, joined as the wires join them. */
function messageText(parts: ContentPart[]): string {
  return parts.filter((p): p is Extract<ContentPart, { kind: 'text' }> => p.kind === 'text').map((p) => p.text).join('\n');
}

// ---------------------------------------------------------------------------
// OpenAI wire
// ---------------------------------------------------------------------------

function openAIMessage(role: string, parts: ContentPart[]): Record<string, unknown> {
  const content: unknown[] = [];
  const toolCalls: unknown[] = [];
  for (const part of parts) {
    if (part.kind === 'text') content.push({ type: 'text', text: part.text });
    else if (part.kind === 'image') content.push(imageToOpenAI(part.image));
    else if (part.kind === 'toolCall') {
      toolCalls.push({ id: part.id, type: 'function', function: { name: part.name, arguments: part.argsText } });
    } else if (part.kind === 'other' && part.wire === 'openai') {
      content.push(cloneJson(part.raw));
    }
  }
  const msg: Record<string, unknown> = { role };
  if (role === 'assistant' && toolCalls.length > 0) {
    const text = messageText(parts);
    msg.content = text === '' ? null : text;
    msg.tool_calls = toolCalls;
  } else if (content.length === 1 && typeof content[0] === 'object' && content[0] !== null && (content[0] as Record<string, unknown>).type === 'text') {
    msg.content = (content[0] as Record<string, unknown>).text as string;
  } else {
    msg.content = content;
  }
  return msg;
}

function openAITools(tools: CanonicalTool[]): unknown[] {
  return tools.map((t) => {
    const fn: Record<string, unknown> = { name: t.name };
    if (t.description !== null) fn.description = t.description;
    if (t.parameters !== null && t.parameters !== undefined) fn.parameters = cloneJson(t.parameters);
    return { type: 'function', function: fn };
  });
}

function openAIToolChoice(req: CanonicalRequest): unknown {
  const tc = req.toolChoice;
  if (tc.mode === 'any') return 'required';
  if (tc.mode === 'none') return 'none';
  if (tc.mode === 'specific' && tc.name !== null) return { type: 'function', function: { name: tc.name } };
  return 'auto';
}

export function toOpenAI(req: CanonicalRequest, model: string, stream: boolean, includeUsage: boolean): Record<string, unknown> {
  const messages: unknown[] = [];
  if (req.systemText !== '') messages.push({ role: 'system', content: req.systemText });
  for (const m of req.messages) {
    if (m.role === 'tool') {
      for (const part of m.parts) {
        if (part.kind === 'toolResult') {
          messages.push({ role: 'tool', tool_call_id: part.id, content: part.content });
        }
      }
      continue;
    }
    messages.push(openAIMessage(m.role, m.parts));
  }
  const body: Record<string, unknown> = { model, messages, stream };
  if (stream && includeUsage) body.stream_options = { include_usage: true };
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.temperature !== null) body.temperature = req.temperature;
  if (req.topP !== null) body.top_p = req.topP;
  if (req.stops.length > 0) body.stop = [...req.stops];
  if (req.tools.length > 0) {
    body.tools = openAITools(req.tools);
    body.tool_choice = openAIToolChoice(req);
  }
  if (req.jsonMode === 'json_object') body.response_format = { type: 'json_object' };
  else if (req.jsonMode === 'json_schema' && req.jsonFormat !== null) body.response_format = cloneJson(req.jsonFormat);
  return body;
}

// ---------------------------------------------------------------------------
// Anthropic wire
// ---------------------------------------------------------------------------

function anthropicBlocks(parts: ContentPart[]): unknown[] {
  const blocks: unknown[] = [];
  for (const part of parts) {
    if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
    else if (part.kind === 'image') blocks.push(imageToAnthropic(part.image));
    else if (part.kind === 'toolCall') {
      blocks.push({ type: 'tool_use', id: part.id, name: part.name, input: part.argsJson ?? {} });
    } else if (part.kind === 'toolResult') {
      blocks.push({ type: 'tool_result', tool_use_id: part.id, content: part.content });
    } else if (part.kind === 'other' && part.wire === 'anthropic') {
      blocks.push(cloneJson(part.raw));
    }
  }
  return blocks;
}

function anthropicTools(tools: CanonicalTool[]): unknown[] {
  return tools.map((t) => {
    const tool: Record<string, unknown> = { name: t.name };
    if (t.description !== null) tool.description = t.description;
    tool.input_schema = t.parameters !== null && t.parameters !== undefined ? cloneJson(t.parameters) : { type: 'object' };
    return tool;
  });
}

function anthropicToolChoice(req: CanonicalRequest): unknown {
  const tc = req.toolChoice;
  if (tc.mode === 'any') return { type: 'any' };
  if (tc.mode === 'none') return { type: 'none' };
  if (tc.mode === 'specific' && tc.name !== null) return { type: 'tool', name: tc.name };
  return { type: 'auto' };
}

export function toAnthropic(req: CanonicalRequest, model: string, stream: boolean): Record<string, unknown> {
  const messages: unknown[] = [];
  for (const m of req.messages) {
    if (m.role === 'tool') {
      // Tool results travel in a user turn on this wire.
      messages.push({ role: 'user', content: anthropicBlocks(m.parts) });
      continue;
    }
    const blocks = anthropicBlocks(m.parts);
    if (blocks.length === 1 && typeof blocks[0] === 'object' && blocks[0] !== null && (blocks[0] as Record<string, unknown>).type === 'text') {
      messages.push({ role: m.role, content: (blocks[0] as Record<string, unknown>).text as string });
    } else {
      messages.push({ role: m.role, content: blocks });
    }
  }
  const body: Record<string, unknown> = { model, messages, stream };
  if (req.systemText !== '') body.system = req.systemText;
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.temperature !== null) body.temperature = req.temperature;
  if (req.topP !== null) body.top_p = req.topP;
  if (req.stops.length > 0) body.stop_sequences = [...req.stops];
  if (req.tools.length > 0) {
    body.tools = anthropicTools(req.tools);
    body.tool_choice = anthropicToolChoice(req);
  }
  // No JSON mode on this wire: the router never sends a JSON-mode request here.
  return body;
}

// ---------------------------------------------------------------------------
// Gemini wire
// ---------------------------------------------------------------------------

/** Strip schema keys Gemini's restricted subset rejects, recursively. */
function geminiSchema(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(geminiSchema);
  if (value !== null && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if (k === 'additionalProperties' || k === '$schema' || k === 'exclusiveMinimum' || k === 'exclusiveMaximum') continue;
      out[k] = geminiSchema(v);
    }
    return out;
  }
  return value;
}

function geminiParts(parts: ContentPart[], callNames: Map<string, string>): unknown[] {
  const out: unknown[] = [];
  for (const part of parts) {
    if (part.kind === 'text') out.push({ text: part.text });
    else if (part.kind === 'image') out.push(imageToGemini(part.image));
    else if (part.kind === 'toolCall') {
      out.push({ functionCall: { name: part.name, args: part.argsJson ?? {} } });
    } else if (part.kind === 'toolResult') {
      let response: unknown = { result: part.content };
      try {
        const parsed: unknown = JSON.parse(part.content);
        if (parsed !== null && typeof parsed === 'object') response = parsed;
      } catch {
        // Keep the string form.
      }
      // OpenAI tool messages carry no name; resolve it from the call
      // with the same id earlier in the conversation.
      out.push({ functionResponse: { name: part.name ?? callNames.get(part.id) ?? 'tool', response } });
    }
  }
  return out;
}

export function toGemini(req: CanonicalRequest, _model: string): Record<string, unknown> {
  const callNames = new Map<string, string>();
  for (const m of req.messages) {
    for (const part of m.parts) {
      if (part.kind === 'toolCall' && !callNames.has(part.id)) callNames.set(part.id, part.name);
    }
  }
  const contents: unknown[] = [];
  for (const m of req.messages) {
    const parts = geminiParts(m.parts, callNames);
    if (parts.length === 0) continue;
    // Assistant turns are "model"; user and tool turns are "user".
    contents.push({ role: m.role === 'assistant' ? 'model' : 'user', parts });
  }
  const body: Record<string, unknown> = { contents };
  if (req.systemText !== '') body.systemInstruction = { parts: [{ text: req.systemText }] };
  if (req.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: req.tools.map((t) => {
          const decl: Record<string, unknown> = { name: t.name };
          if (t.description !== null) decl.description = t.description;
          if (t.parameters !== null && t.parameters !== undefined) decl.parameters = geminiSchema(t.parameters);
          return decl;
        }),
      },
    ];
    const tc = req.toolChoice;
    const mode = tc.mode === 'any' ? 'ANY' : tc.mode === 'none' ? 'NONE' : 'AUTO';
    const config: Record<string, unknown> = { mode };
    if (tc.mode === 'specific' && tc.name !== null) config.allowedFunctionNames = [tc.name];
    body.toolConfig = { functionCallingConfig: config };
  }
  const config: Record<string, unknown> = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.stops.length > 0) config.stopSequences = [...req.stops];
  if (req.jsonMode !== 'none') {
    config.responseMimeType = 'application/json';
    if (req.jsonMode === 'json_schema' && req.jsonFormat !== null && typeof req.jsonFormat === 'object' && (req.jsonFormat as Record<string, unknown>).json_schema !== undefined) {
      const schema = (req.jsonFormat as Record<string, unknown>).json_schema;
      if (schema !== null && typeof schema === 'object' && (schema as Record<string, unknown>).schema !== undefined) {
        config.responseSchema = geminiSchema((schema as Record<string, unknown>).schema);
      }
    }
  }
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}
