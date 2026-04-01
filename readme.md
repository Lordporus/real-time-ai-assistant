# E.D.I.T.H — Every Day I'm Thinking of Humanity

> A production-grade, streaming AI assistant backend with real-time web search, voice synthesis, vector memory, and full async concurrency hardening.

---

## Project Overview

E.D.I.T.H is a personal AI assistant backend designed for low-latency, high-fidelity conversational AI. It streams responses token-by-token, synthesises speech sentence-by-sentence, retrieves personal context from a local vector store, and performs live web searches — all concurrently without blocking.

This is not a minimal prototype. The system is built with production-grade concurrency, cancellation propagation, thread safety, and resource management from the ground up.

---

## Features

- **Multi-mode streaming** — general chat, real-time (web search), and brain-routed (auto-classify) via SSE
- **Token-level streaming** — responses appear word-by-word in the client
- **Real-time web search** — Tavily integration for live query augmentation
- **Text-to-Speech** — edge-tts sentence streaming; audio delivered as sentences complete
- **Vector memory** — FAISS + sentence-transformers; upload files, retrieve relevant context per query
- **Multi-key API fallback** — unlimited Groq API keys with automatic rotation and rate-limit recovery
- **Deep reasoning routing** — Anthropic Claude used automatically for long or complex queries
- **Session persistence** — in-memory sessions with JSON disk backup; survives restarts
- **Client disconnect propagation** — backend stops processing immediately when client disconnects
- **Thread-safe sessions** — per-session RLock; concurrent requests for the same session cannot corrupt data
- **Stream safety limits** — configurable max chars and max chunks per response; prevents runaway generation

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI + Uvicorn |
| **LLM Orchestration** | LangChain (ChatGroq, ChatAnthropic) |
| **LLM Provider** | Groq API (`llama-3.3-70b-versatile`) |
| **Deep Reasoning** | Anthropic Claude (optional) |
| **Web Search** | Tavily |
| **Embeddings** | sentence-transformers `all-MiniLM-L6-v2` (local, no API) |
| **Vector Store** | FAISS (CPU) |
| **Text-to-Speech** | edge-tts (Microsoft Edge TTS, no API key) |
| **Streaming Protocol** | Server-Sent Events (SSE) |
| **Concurrency** | asyncio + ThreadPoolExecutor hybrid |
| **Config** | python-dotenv |

---

## Architecture Summary

```
Client (browser / app)
       │  SSE stream
       ▼
FastAPI endpoint  ──►  _stream_generator()
                              │
                    ┌─────────┴──────────┐
                    │                    │
             ChatService          AsyncIterator
           (orchestrator)       (async generator)
                    │                    │
                    │         asyncio.Queue (maxsize=256)
                    │                    ▲
                    │           Daemon thread
                    │         (LangChain sync stream)
                    │
          ┌─────────┼───────────────┐
          │         │               │
     GroqService  RealtimeService  BrainService
      (General)  (Tavily + Groq)  (Classifier)
          │
     VectorStoreService
       (FAISS + embeddings)

TTS: _tts_pool (ThreadPoolExecutor × 4)
     edge-tts per-sentence, async-wrapped
```

Full detail: see [SYSTEM_WALKTHROUGH.md](./SYSTEM_WALKTHROUGH.md)

---

## Key Engineering Highlights

### 1. Thread → Async Queue Bridge

LangChain's `chain.stream()` is synchronous. Running it on the event loop thread would block all concurrent requests. Instead, each streaming generator spawns a daemon thread that pushes chunks into an `asyncio.Queue` via `loop.call_soon_threadsafe`. The async generator `await`s from this queue — zero blocking on the event loop.

### 2. Per-Session RLock (Not Global Lock)

Each `session_id` gets its own `threading.RLock`, created lazily using double-checked locking. Concurrent requests to **different sessions** never block each other. `RLock` is used instead of `Lock` to allow safe re-entry from within a streaming `finally` block.

### 3. Client Disconnect Propagation

A background `asyncio.Task` polls `request.is_disconnected()` every 500ms. On disconnect, it sets a `threading.Event`. The main streaming loop checks this event before processing each chunk. On cancel: `chunk_iter.aclose()` is called (triggering the generator's `finally` block for clean session save), and all pending TTS futures are immediately cancelled.

### 4. Streaming Memory Optimisation

Response text is accumulated using `list.append(chunk)` + a single `"".join(chunk_parts)` in the `finally` block — not `+=` per chunk. This reduces O(n²) string allocation to O(n) total work for long responses.

### 5. Stream Safety Limits

Max chars per response (`MAX_STREAM_CHARS`, default 20,000) and max chunks (`MAX_STREAM_CHUNKS`, default 2,000) are enforced in every streaming loop. A user-visible `[response truncated]` notice is appended when the limit is hit. Both are configurable via environment variables.

### 6. API Key Rotation Without Closure Bugs

`_invoke_with_key` uses a default-argument capture (`idx: int = i`) to capture the loop variable by value, not by reference. This prevents the classic Python closure bug where retries silently use the wrong API key.

### 7. FAISS Thread Safety

FAISS's C++ core is not thread-safe. All index reads and writes are serialised with a `threading.RLock`. Embedding computation (CPU-intensive) happens outside the lock — only the index mutation is locked, minimising contention time.

### 8. Parallel Brain + Search

In JARVIS mode, the brain classification and Tavily web search run simultaneously in a `ThreadPoolExecutor(max_workers=2)`. By the time the brain returns its routing decision, search results are often already available — saving one full sequential round-trip.

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10+
- A Groq API key ([console.groq.com](https://console.groq.com))
- Optional: Tavily API key ([tavily.com](https://tavily.com))
- Optional: Anthropic API key

### 2. Clone & Install

```bash
git clone <your-repo>
cd jarvis
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=gsk_...

# Optional — additional keys for fallback & rotation
GROQ_API_KEY_2=gsk_...
GROQ_API_KEY_3=gsk_...

# Optional — enables web search (realtime mode)
TAVILY_API_KEY=tvly-...

# Optional — enables deep reasoning routing
ANTHROPIC_API_KEY=sk-ant-...

# Optional — personalisation
ASSISTANT_NAME=EDITH
JARVIS_USER_TITLE=Boss
JARVIS_OWNER_NAME=YourName

# Optional — model overrides
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_BRAIN_MODEL=llama-3.1-8b-instant

# Optional — TTS voice (run `edge-tts --list-voices`)
TTS_VOICE=en-GB-RyanNeural
TTS_RATE=+22%

# Optional — stream safety limits
MAX_STREAM_CHARS=20000
MAX_STREAM_CHUNKS=2000

# Optional — deployment mode
ENVIRONMENT=development   # or: production
SESSION_TTL=86400
```

### 4. Run

```bash
# Development (auto-reload, 1 worker)
python run.py

# Or explicitly set environment
ENVIRONMENT=development python run.py

# Production (no reload, 4 workers)
ENVIRONMENT=production python run.py
```

The API is available at `http://localhost:8000`  
The frontend (if present) at `http://localhost:8000/app/`

---

## API Endpoints

### Chat

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Non-streaming general chat |
| `POST` | `/chat/stream` | **Streaming** general chat (SSE) |
| `POST` | `/chat/realtime` | Non-streaming + Tavily web search |
| `POST` | `/chat/realtime/stream` | **Streaming** + web search (SSE) |
| `POST` | `/chat/jarvis/stream` | **Streaming**, brain auto-routes (recommended) |
| `GET` | `/chat/history/{session_id}` | Retrieve chat history |

**Request body (all chat endpoints):**

```json
{
  "message": "What's the latest news on AI?",
  "session_id": "optional-existing-session-id",
  "tts": false
}
```

### Files & Memory

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload a file (PDF/TXT) into the vector store |
| `GET` | `/files` | List all ingested files |
| `DELETE` | `/files/{filename}` | Remove a file from memory |

### Voice

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/tts` | Generate TTS audio for a text string (returns MP3) |

### System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/api` | Endpoint index |

---

## SSE Event Format

Streaming endpoints yield Server-Sent Events. Clients should parse `data:` lines as JSON:

```js
// Handshake (first event)
{ "session_id": "abc123", "chunk": "", "done": false }

// Activity event (routing information)
{ "activity": { "event": "routing", "route": "realtime", "provider": "groq" } }

// Search sources (realtime mode only)
{ "type": "sources", "sources": [...] }

// Text chunk
{ "chunk": "Hello, ", "done": false }

// TTS audio (if tts: true)
{ "audio": "<base64-mp3>", "sentence": "Hello, how can I help?" }

// Stream complete
{ "chunk": "", "done": true, "session_id": "abc123" }

// Error
{ "chunk": "", "done": true, "error": "..." }
```

---

## Before vs After — Production Hardening

| Problem (Before) | Fix (After) |
|---|---|
| `asyncio.run()` inside ThreadPoolExecutor | Per-call `new_event_loop()` with full cleanup |
| Global `reload=True` in production | Env-based toggle; `reload=False` + 4 workers in prod |
| Mid-stream LLM fallback → duplicate output | Fallback blocked if any chunk already yielded |
| Loop variable `i` captured by reference | Default-arg capture: `_invoke_with_key(idx: int = i)` |
| `save_chat_session` blocking event loop | `await asyncio.to_thread(save_chat_session, id)` |
| `self.sessions[-1]` race condition | Fixed index `assistant_index` captured at insert time |
| FAISS accessed concurrently → SIGSEGV | `threading.RLock` on all reads and writes |
| No locking on session store | Per-session `RLock`, double-checked lock creation |
| `content += chunk` O(n²) concatenation | `list.append` + `"".join()` in `finally` |
| No response size limit | `MAX_STREAM_CHARS` / `MAX_STREAM_CHUNKS` guards |
| Client disconnect → zombie threads & wasted quota | `threading.Event` watcher + `aclose()` propagation |

---

## Limitations

- **Single-process session store** — sessions live in RAM. Multiple Uvicorn workers (production mode) cannot share sessions. Use a single worker or migrate to Redis for multi-worker deployments.
- **FAISS is single-node** — no distributed vector search.
- **edge-tts requires internet** — it proxies Microsoft's cloud TTS. Offline deployments need a different TTS backend.
- **Groq rate limits** — even with multi-key rotation, sustained high-concurrency workloads will hit rate limits. The system handles this gracefully but cannot bypass Groq's quotas.
- **Anthropic routing is rule-based** — keyword/length heuristics are fast but imperfect. Complex short queries may still route to Groq.

---

## Future Improvements

- **Redis session store** — enables horizontal scaling with multiple workers
- **Streaming retry with exponential backoff** — per-connection resilience
- **WebSocket transport** — bidirectional for voice-input + voice-output scenarios
- **Authentication** — per-user sessions with JWT or API key auth
- **Structured output** — function calling / tool use for triggerable actions
- **Observability** — OpenTelemetry tracing across the LLM → stream → TTS pipeline
- **Model selection per request** — allow clients to specify Groq model at runtime

---

## Project Structure

```
jarvis/
├── app/
│   ├── main.py                  # FastAPI app, all endpoints, stream generator
│   ├── models.py                # Pydantic request/response models
│   ├── auth.py                  # Session authentication middleware
│   └── services/
│       ├── chat_service.py      # Core orchestrator, session management, streaming
│       ├── groq_service.py      # General LLM (Groq, multi-key, FAISS context)
│       ├── realtime_service.py  # Realtime LLM (Groq + Tavily search)
│       ├── brain_service.py     # Query classifier (routes general vs realtime)
│       ├── vector_store.py      # FAISS ingestion and similarity search
│       └── anthropic_service.py # Deep reasoning provider (optional)
├── config.py                    # Central configuration (env vars, prompts, paths)
├── run.py                       # Uvicorn entry point with env-based config
├── requirements.txt
├── SYSTEM_WALKTHROUGH.md        # Full architecture documentation
├── database/
│   ├── chats_data/              # JSON chat history files
│   ├── learning_data/           # Plain-text user knowledge files
│   └── vector_store/            # FAISS index files
└── frontend/                    # Static frontend (served at /app/)
```

---

## License

Private — internal / personal use. Contact the maintainer for licensing enquiries.