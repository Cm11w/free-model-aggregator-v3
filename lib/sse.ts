// lib/sse.ts — a hardened incremental SSE parser and the frame emitters.
//
// The parser follows the SSE framing rules the way a real event stream
// needs: CRLF and LF both terminate lines, blank lines terminate events,
// `data:` lines join with "\n" (one leading space stripped), `event:` names
// the event, `:` lines are comments, and a final event without its trailing
// blank line is still dispatched when the stream ends (flush). It is fed
// decoded text in arbitrary chunk splits — the caller decodes bytes with a
// streaming TextDecoder so multi-byte characters split across chunks survive.
//
// Every frame keeps its raw text, so a same-wire stream can be forwarded
// byte for byte after validation.

export interface SseFrame {
  /** The event name, or null for a bare `data:` frame. */
  event: string | null;
  /** The data payload: data lines joined with "\n". */
  data: string;
  /** The frame's raw text, exactly as received (line endings normalized to \n). */
  raw: string;
}

const MAX_FRAME_BYTES = 4 * 1024 * 1024;

export class SseParser {
  private buffer = '';
  private lineBuffer = '';
  private pending: { event: string | null; data: string[]; rawLines: string[] } | null = null;
  private overflowed = false;

  /** Feed decoded text; returns the complete frames it finishes. */
  feed(chunk: string): SseFrame[] {
    if (this.overflowed) return [];
    this.buffer += chunk;
    if (this.buffer.length > MAX_FRAME_BYTES * 2) {
      // A single event larger than the cap: drop the stream's remainder
      // rather than grow memory without bound.
      this.overflowed = true;
      this.buffer = '';
      return [];
    }
    const frames: SseFrame[] = [];
    let nl: number;
    while ((nl = this.buffer.indexOf('\n')) !== -1) {
      let line = this.buffer.slice(0, nl);
      this.buffer = this.buffer.slice(nl + 1);
      if (line.endsWith('\r')) line = line.slice(0, -1);
      const frame = this.consumeLine(line);
      if (frame !== null) frames.push(frame);
    }
    return frames;
  }

  /** End of stream: dispatch a trailing partial event, if one is open. */
  flush(): SseFrame[] {
    const frames: SseFrame[] = [];
    if (this.buffer !== '') {
      let line = this.buffer;
      this.buffer = '';
      if (line.endsWith('\r')) line = line.slice(0, -1);
      const frame = this.consumeLine(line);
      if (frame !== null) frames.push(frame);
    }
    const pending = this.finishEvent();
    if (pending !== null) frames.push(pending);
    this.lineBuffer = '';
    return frames;
  }

  private consumeLine(line: string): SseFrame | null {
    if (line === '') return this.finishEvent();
    if (this.pending === null) this.pending = { event: null, data: [], rawLines: [] };
    this.pending.rawLines.push(line);
    if (line.startsWith(':')) return null; // Comment.
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'data') this.pending.data.push(value);
    else if (field === 'event') this.pending.event = value;
    // id:, retry: and unknown fields are accepted and ignored.
    return null;
  }

  private finishEvent(): SseFrame | null {
    const pending = this.pending;
    this.pending = null;
    if (pending === null) return null;
    if (pending.data.length === 0 && pending.event === null) return null;
    return {
      event: pending.event,
      data: pending.data.join('\n'),
      raw: `${pending.rawLines.join('\n')}\n\n`,
    };
  }
}

/** True when a frame is the OpenAI stream terminator. */
export function isDoneFrame(frame: SseFrame): boolean {
  return frame.data.trim() === '[DONE]';
}

/** Parse a frame's data as JSON; null when it is not one JSON value. */
export function frameJson(frame: SseFrame): unknown {
  if (frame.data.trim() === '') return null;
  try {
    return JSON.parse(frame.data);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Emitters. Every emitter terminates its frame with a blank line.
// ---------------------------------------------------------------------------

/** A bare `data:` frame (OpenAI and Gemini wires). */
export function dataFrame(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

/** A named-event frame (Anthropic wire). */
export function eventFrame(event: string, payload: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
}

/** The OpenAI stream terminator. */
export const DONE_FRAME = 'data: [DONE]\n\n';

/**
 * Normalize raw frame text for forwarding: exactly one trailing blank line,
 * LF endings. The frame's inner bytes are untouched.
 */
export function normalizeRaw(raw: string): string {
  const body = raw.replace(/\r\n/g, '\n').replace(/\n+$/, '');
  return body === '' ? '' : `${body}\n\n`;
}
