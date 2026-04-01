# SYSTEM WALKTHROUGH — J.A.R.V.I.S / E.D.I.T.H

> **Just A Rather Very Intelligent System** — an async, streaming AI assistant backend built on FastAPI, LangChain, Groq, FAISS, and edge-tts.

---

## Table of Contents

1. [Startup Sequence](#1-startup-sequence)
2. [Request Routing Overview](#2-request-routing-overview)
3. [Chat Service — The Core Orchestrator](#3-chat-service--the-core-orchestrator)
4. [The Streaming Pipeline](#4-the-streaming-pipeline)
5. [The Thread → Async Queue Bridge](#5-the-thread--async-queue-bridge)
6. [TTS Pipeline](#6-tts-pipeline)
7. [Client Disconnect & Cancellation](#7-client-disconnect--cancellation)
8. [FAISS Vector Store](#8-faiss-vector-store)
9. [Session Memory & Thread Safety](#9-session-memory--thread-safety)
10. [API Key Rotation & Fallback](#10-api-key-rotation--fallback)
11. [Brain Service — Query Classifier](#11-brain-service--query-classifier)
12. [Error Handling](#12-error-handling)
13. [Concurrency Model Summary](#13-concurrency-model-summary)
14. [Recent System Upgrades & Fixes](#14-recent-system-upgrades--fixes)

---

## 1. Startup Sequence

When the server starts, the FastAPI **lifespan** context manager initialises every service in dependency order:

```
uvicorn starts
  └─ lifespan() begins
       ├─ VectorStoreService()   → loads / builds FAISS index from disk
       ├─ GroqService()          → creates one ChatGroq client per API key
       ├─ RealtimeGroqService()  → same, also inits Tavily search tool
       ├─ BrainService()         → mini Groq classifier (llama-3.1-8b-instant)
       ├─ AnthropicService()     → optional deep-reasoning provider (skipped if not configured)
       └─ ChatService()          → wires all services; owns in-memory session store
```

On **shutdown**, `_tts_pool.shutdown(wait=True)` is called and every open session is flushed to disk.

---

## 2. Request Routing Overview

The system exposes three chat streaming endpoints plus non-streaming variants:

| Endpoint | Mode | LLM Used |
|---|---|---|
| `POST /chat` | Non-stream, general | Groq (or Anthropic for deep queries) |
| `POST /chat/stream` | **SSE stream**, general | Groq / Anthropic |
| `POST /chat/realtime` | Non-stream + web search | Groq + Tavily |
| `POST /chat/realtime/stream` | **SSE stream** + web search | Groq + Tavily |
| `POST /chat/jarvis/stream` | **SSE stream**, brain-routed | Brain → Groq or Groq+Tavily |
| `POST /tts` | Audio response | edge-tts (no API key) |
| `GET /chat/history/{id}` | Chat history | — |
| `GET /health` | Status check | — |

Every streaming endpoint follows the same internal funnel through `_stream_generator`.

---

## 3. Chat Service — The Core Orchestrator

`ChatService` is the single object that owns:

- **Session store** — `Dict[str, List[ChatMessage]]` in RAM
- **Per-session `threading.RLock`** — one lock per `session_id`, created lazily
- **Service references** — `GroqService`, `RealtimeGroqService`, `BrainService`, `AnthropicService`

### Session Lifecycle

```
get_or_create_session(session_id)
  ├─ session in memory?   → return immediately
  ├─ session on disk?     → load JSON → lock → write to self.sessions
  └─ new session?         → lock → self.sessions[id] = []
```

Every session access (`add_message`, `get_chat_history`, `save_chat_session`) acquires `_get_session_lock(session_id)` — an `RLock` — before touching `self.sessions[session_id]`. Unrelated sessions are **never blocked by each other**.

### Anthropic Routing (Deep-Query Detection)

Before calling any LLM, `ChatService._should_use_anthropic()` decides the provider using zero-latency rule-based logic:

- **Keyword match** (`analyze`, `summarise`, `compare`, `deep dive`, …)
- **Length threshold** (> 300 chars)

If Anthropic is unavailable, the system falls back transparently to Groq.

---

## 4. The Streaming Pipeline

This is the most complex part of the system. Every streaming route follows this exact sequence:

```
Client POST /chat/jarvis/stream
│
├─ Endpoint calls chat_service.process_jarvis_message_stream(session_id, msg)
│    └─ Returns: AsyncIterator[str | dict]  (an async generator)
│
├─ Endpoint wraps it:
│    StreamingResponse(_stream_generator(session_id, chunk_iter, ..., request=request))
│
└─ _stream_generator runs in the event loop:
     ├─ Yields initial handshake: {"session_id": ..., "chunk": "", "done": false}
     ├─ Starts _watcher() asyncio.Task (polls request.is_disconnected())
     │
     ├─ async for chunk in chunk_iter:        ← awaits each item
     │    ├─ Check cancel_event.is_set()
     │    ├─ Drain ready TTS futures
     │    ├─ Route chunk type:
     │    │    ├─ dict + "_activity"     → SSE activity event
     │    │    ├─ dict + "_search_results" → SSE sources event
     │    │    └─ str                    → SSE text chunk
     │    └─ TTS sentence accumulation + submit
     │
     └─ Stream ends → flush remaining TTS → yield {"done": true}
```

### SSE Event Types Clients Receive

```json
{"session_id": "abc123", "chunk": "", "done": false}   // handshake
{"activity": {"event": "routing", "route": "realtime"}} // activity
{"type": "sources", "sources": [...]}                   // search results
{"chunk": "Hello ", "done": false}                      // text chunk
{"audio": "<base64>", "sentence": "Hello world."}       // TTS audio
{"chunk": "", "done": true, "session_id": "abc123"}     // completion
```

---

## 5. The Thread → Async Queue Bridge

LangChain's `chain.stream()` is **synchronous** — it blocks the calling thread. The event loop cannot `await` it directly without blocking all concurrent requests.

### Solution: Daemon Thread + asyncio.Queue

Each streaming generator in `ChatService` sets up this bridge internally:

```python
aq: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=256)
loop = asyncio.get_running_loop()

def _run_sync_stream() -> None:
    try:
        for chunk in sync_stream:                         # blocks this thread only
            loop.call_soon_threadsafe(aq.put_nowait, ("chunk", chunk))
        loop.call_soon_threadsafe(aq.put_nowait, ("done", None))
    except Exception as exc:
        loop.call_soon_threadsafe(aq.put_nowait, ("error", exc))

threading.Thread(target=_run_sync_stream, daemon=True).start()

# Back in the event loop — non-blocking consumption:
while True:
    item_type, data = await aq.get()
    if item_type == "done": break
    if item_type == "error": raise data
    yield data
```

**Why `maxsize=256`?** If the consumer stops (`aclose()` on disconnect), the producer blocks after 256 items instead of running unbounded. It self-terminates within seconds — no manual interrupt needed.

**Why `call_soon_threadsafe`?** Direct `asyncio.Queue` methods are not thread-safe. `call_soon_threadsafe` schedules the put on the event loop thread safely.

---

## 6. TTS Pipeline

TTS runs in a `ThreadPoolExecutor` (`_tts_pool`) with 4 workers, completely decoupled from the LLM stream:

```
Text chunk arrives
  └─ TTS sentence detector accumulates chunks
       └─ sentence boundary found (_split_sentences)
            └─ _submit_tts(sentence)
                 └─ _tts_pool.submit(_generate_tts_sync, text, voice, rate)
                      └─ asyncio.wrap_future(concurrent_fut)
                           └─ stored in audio_queue: list[(Future, str)]

Between/after LLM chunks:
  └─ _drain_ready() → non-blocking poll of audio_queue[0].done()
       └─ if done: fut.result() → base64 encode → yield SSE audio event

After LLM stream ends:
  └─ _drain_all_remaining() → await each future with 15s timeout
```

`_generate_tts_sync` runs in an isolated event loop (not the main loop) using `edge-tts`, accumulating audio bytes into a `bytearray` for O(1) appends.

**Result:** TTS audio is streamed to the client as sentences complete — not after the full response.

---

## 7. Client Disconnect & Cancellation

### The Problem
When a client closes the connection, the LangChain daemon thread keeps yielding into the queue. TTS jobs keep running in `_tts_pool`. Resources leak.

### The Solution: threading.Event + Request Polling

```
_watcher() asyncio.Task
  └─ every 0.5s: await request.is_disconnected()
       └─ if True: cancel_event.set()

Main loop (every chunk):
  └─ if cancel_event.is_set():
       ├─ await chunk_iter.aclose()    → GeneratorExit into chat_service gen
       │    └─ finally block runs → chunk_parts joined → session saved
       ├─ for fut in audio_queue: fut.cancel()   → TTS pool slots freed
       ├─ audio_queue.clear()
       └─ return                        → generator exits, watcher cancelled in finally
```

**Why `threading.Event` not `asyncio.Event`?** `cancel_event.is_set()` runs on the hot path (every chunk) without needing `await`. Zero overhead — just a flag read.

**Why `aclose()` not force-kill?** `aclose()` sends `GeneratorExit` cleanly into the chat_service async generator, triggering its `finally` block — session data is committed and saved even on disconnect.

---

## 8. FAISS Vector Store

### What It Does

Stores user-uploaded text as dense vector embeddings. At query time, similarity search retrieves the most relevant chunks and injects them into the LLM prompt as context ("memory").

### Thread Safety

FAISS's C++ index is **not thread-safe**. Every read and write is wrapped in `threading.RLock`:

```python
with self._faiss_lock:
    results = self.vector_store.similarity_search(query, k=5)
```

CPU-bound work (embedding computation, text chunking) happens **outside** the lock to minimise holding time. Only the actual FAISS index mutation is locked.

### File Ingestion Flow

```
POST /upload → UploadFile
  └─ text extracted (PDF/TXT)
       └─ RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
            └─ HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2) [outside lock]
                 └─ with _faiss_lock: faiss.add_documents()
                      └─ faiss.save_local()
```

---

## 9. Session Memory & Thread Safety

### Storage Model

```python
self.sessions: Dict[str, List[ChatMessage]]   # in-memory
self._session_locks: Dict[str, threading.RLock]  # per-session locks
self._locks_guard: threading.Lock               # guards lock creation only
```

### Double-Checked Lock Creation

```python
def _get_session_lock(session_id):
    lock = self._session_locks.get(session_id)   # fast path — no guard needed
    if lock is not None:
        return lock
    with self._locks_guard:                       # slow path — create once
        if session_id not in self._session_locks:
            self._session_locks[session_id] = threading.RLock()
        return self._session_locks[session_id]
```

`_locks_guard` is held for < 1 microsecond (dict insert). It is **never held during I/O, LLM calls, or streaming**. Cannot deadlock.

### Why RLock (not Lock)?

`save_chat_session` may be called from inside a stack frame that already holds the session lock (e.g., the streaming `finally` block). A plain `Lock` would deadlock on re-entry; `RLock` allows the same thread to re-acquire.

### Persistence

Every N chunks during streaming: `await asyncio.to_thread(save_chat_session, session_id, False)` — offloads JSON disk write to a thread pool, keeping the event loop free. On stream completion: final save. On server shutdown: all sessions flushed.

---

## 10. API Key Rotation & Fallback

### Multi-Key Registration

```
GROQ_API_KEY       → llms[0]   (primary)
GROQ_API_KEY_2     → llms[1]
GROQ_API_KEY_3     → llms[2]
...unlimited keys
```

### Rotation Strategy

`get_next_key_pair()` returns a starting index based on round-robin state. Each request starts at a different key to distribute load. If a key fails (rate limit, timeout), the loop increments to the next key:

```python
for j in range(n):
    i = (key_start_index + j) % n
    def _invoke_with_key(idx: int = i):    # capture by value — no closure bug
        chain = prompt | self.llms[idx]
        return chain.invoke(...)
    with_retry(_invoke_with_key, ...)
```

**Streaming fallback rule:** If any chunk has already been yielded, fallback is **disabled**. Yielding a partial response then starting over with a new key would produce duplicate/corrupted output. Instead, the error propagates immediately.

---

## 11. Brain Service — Query Classifier

The Brain classifies each JARVIS query as `"general"` (no web search) or `"realtime"` (needs web search):

```
User sends: "What happened in the news today?"
  └─ BrainService.classify(question, chat_history, key_index)
       └─ Groq (llama-3.1-8b-instant) → returns: "realtime"
            └─ ChatService routes to RealtimeGroqService (Tavily + Groq)

User sends: "What is the capital of France?"
  └─ Brain → "general"
       └─ ChatService routes to GroqService (no search)
```

Brain and Chat use **different** API keys simultaneously (key pair rotation separates them), preventing rate limit contention from brain calls starving the main LLM.

The web search (`prefetch_web_search`) runs **in parallel** with brain classification via `ThreadPoolExecutor(max_workers=2)` — by the time the brain returns, search results are often already ready.

---

## 12. Error Handling

| Layer | Handling |
|---|---|
| **All Groq keys fail** | `AllGroqApisFailedError` → HTTP 503 |
| **Rate limit (429)** | Detected by `_is_rate_limit_error()` → HTTP 429 + user message |
| **Brain timeout** | Logs warning, defaults to `"realtime"` route |
| **Search prefetch timeout** | Logs warning, proceeds with empty context |
| **Mid-stream LLM error** | SSE `{"done": true, "error": "..."}` sent to client |
| **TTS timeout** | Per-sentence 15s timeout; sentence skipped; stream continues |
| **TTS failure** | Logged, skipped silently; text still sent |
| **Client disconnect** | `cancel_event` → generator closes → TTS futures cancelled |
| **Session load failure** | Logs warning, returns empty session |
| **File save failure** | 3 retries with exponential backoff (0.1s, 0.2s) |

---

## 13. Concurrency Model Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                      uvicorn event loop                         │
│                                                                 │
│  FastAPI handler ──► _stream_generator (async generator)        │
│         │                    │                                  │
│         │            async for chunk in chunk_iter             │
│         │                    │                                  │
│         │            ┌───────▼────────┐                        │
│         │            │  asyncio.Queue │  ◄── call_soon_threadsafe │
│         │            └───────▲────────┘                        │
│         │                    │                                  │
│  _watcher Task               │                                  │
│  (polls disconnect)   ┌──────┴──────┐                          │
│                        │ Daemon Thread│  (LangChain sync stream) │
│                        └──────────────┘                         │
│                                                                 │
│         │        ┌────────────────────────┐                     │
│         └──────► │  ThreadPoolExecutor    │  (_tts_pool × 4)   │
│                   │  _generate_tts_sync   │                     │
│                   └────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘

Per-session RLock   →  guards self.sessions[id]
FAISS RLock         →  guards FAISS index (all reads + writes)
_locks_guard Lock   →  guards lock dict insertion only (< 1 µs)
```

**The rule:** no synchronous blocking code ever runs on the event loop thread. All blocking operations (LangChain, FAISS writes, disk I/O) are offloaded — either to daemon threads (streaming) or `asyncio.to_thread` (saves).

---

## 14. Recent System Upgrades & Fixes

### Frontend: E.D.I.T.H Mode State Synchronization
- **ID Resolution:** Fixed a critical bug where `id="btn-jarvis"` in HTML mismatched `btn-edith` in JavaScript, preventing the core mode listener from attaching.
- **Overlay Rendering Order:** Added `pointer-events: none` to the `.mode-slider` background pill. Previously, the animated element was intercepting click events, breaking mode switches unexpectedly.
- **Re-initialization (`newChat`):** Enhanced the frontend UI state manager to forcibly call `setMode(currentMode)` upon clearing the chat, restoring the button highlights, toggle visibility, and correct slider transform coordinates.

### Backend: Core Engine Stability
- **Streaming Limit Precision:** Corrected an off-by-one logic flaw (`>` to `>=`) for `MAX_STREAM_CHARS` in the `_stream_generator`, preventing any possibility of data truncation overflowing the frontend UI thresholds.
- **File Context Pipeline (FAISS RAG):** Debugged and restored the context ingestion flow. The LLM now correctly uses the extracted text chunks pushed through the FAISS index during prompt execution, preventing E.D.I.T.H from ignoring previously uploaded PDF/TXT content.
- **Environment Bootstrapping:** Repaired Python `python-dotenv` import errors in `config.py` so keys are reliably injected to the process at startup before lifespan runs.
