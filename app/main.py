from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from contextlib import asynccontextmanager
import os
import uvicorn
import logging
import json
import time
import re
import base64
import asyncio
import typing
import threading
from concurrent.futures import ThreadPoolExecutor
import edge_tts
from app.models import ChatRequest, ChatResponse, TTSRequest

RATE_LIMIT_MESSAGE = (
    "You've reached your daily API limit for this assistant. "
    "Your credits will reset in a few hours, or you can upgrade your plan for more. "
    "Please try again later."
)

def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "tokens per day" in msg

from app.services.vector_store import VectorStoreService
from app.services.groq_service import GroqService, AllGroqApisFailedError
from app.services.realtime_service import RealtimeGroqService
from app.services.chat_service import ChatService
from app.services.brain_service import BrainService
try:
    from app.services.anthropic_service import AnthropicService as _AnthropicService
except ImportError:
    _AnthropicService = None  # type: ignore[assignment,misc]
from config import (
    VECTOR_STORE_DIR, GROQ_API_KEYS, GROQ_MODEL, TAVILY_API_KEY,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHAT_HISTORY_TURNS,
    ASSISTANT_NAME, TTS_VOICE, TTS_RATE,
    MAX_STREAM_CHARS, MAX_STREAM_CHUNKS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("E.D.I.T.H")

vector_store_service: VectorStoreService = None
groq_service: GroqService = None
realtime_service: RealtimeGroqService = None
brain_service: BrainService = None
chat_service: ChatService = None
anthropic_service: _AnthropicService = None

# ── startup guard ───────────────────────────────────────────────────────────
# Secondary process-level guard. The primary fix is workers=1 in run.py.
# Prevents the lifespan from running twice in the same Python process
# (e.g. test frameworks, edge-case reload scenarios).
_SERVICES_INITIALIZED: bool = False
_INIT_LOCK = threading.Lock()

def print_title():
    title = """

        ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
        ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
        ██║███████║██████╔╝██║   ██║██║███████╗
   ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
   ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
    ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝

        Every Day I'm Thinking of Humanity
    """
    print(title)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store_service, groq_service, realtime_service, brain_service, chat_service
    global _SERVICES_INITIALIZED

    with _INIT_LOCK:
        if _SERVICES_INITIALIZED:
            logger.warning(
                "[STARTUP] lifespan called again in the same process — skipping duplicate init. "
                "If you see this in production, ensure workers=1 in run.py."
            )
            yield
            return
        _SERVICES_INITIALIZED = True

    print_title()
    logger.info("=" * 60)
    logger.info("E.D.I.T.H - Starting Up...")
    logger.info("=" * 60)
    logger.info("[CONFIG] Assistant name: %s", ASSISTANT_NAME)
    logger.info("[CONFIG] Groq model: %s", GROQ_MODEL)
    logger.info("[CONFIG] Groq API keys loaded: %d", len(GROQ_API_KEYS))
    logger.info("[CONFIG] Tavily API key: %s", "configured" if TAVILY_API_KEY else "NOT SET")
    logger.info("[CONFIG] Embedding model: %s", EMBEDDING_MODEL)
    logger.info("[CONFIG] Chunk size: %d | Overlap: %d | Max history turns: %d",
                CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHAT_HISTORY_TURNS)

    try:
        logger.info("Initializing vector store service...")
        t0 = time.perf_counter()
        vector_store_service = VectorStoreService()
        vector_store_service.create_vector_store()
        logger.info("[TIMING] startup_vector_store: %.3fs", time.perf_counter() - t0)

        logger.info("Initializing Groq service (general queries)...")
        groq_service = GroqService(vector_store_service)
        logger.info("Groq service initialized successfully")

        logger.info("Initializing Realtime Groq service (with Tavily search)...")
        realtime_service = RealtimeGroqService(vector_store_service)
        logger.info("Realtime Groq service initialized successfully")

        logger.info("Initializing Brain service (Groq query classification)...")
        brain_service = BrainService()
        logger.info("Brain service initialized successfully")

        logger.info("Initializing Anthropic service (deep reasoning)...")
        if _AnthropicService is not None:
            try:
                anthropic_service = _AnthropicService(vector_store_service)
                logger.info("Anthropic service initialized successfully (deep routing enabled)")
            except Exception as _ae:
                anthropic_service = None
                logger.warning("Anthropic service unavailable: %s — deep routing disabled", _ae)
        else:
            anthropic_service = None
            logger.warning("AnthropicService module not found — deep routing disabled")

        logger.info("Initializing chat service...")
        chat_service = ChatService(
            groq_service, realtime_service, brain_service,
            anthropic_service=anthropic_service,
        )
        logger.info("Chat service initialized successfully")

        logger.info("=" * 60)
        logger.info("Service Status:")
        logger.info("  Vector Store: Ready")
        logger.info("  Groq AI (General): Ready")
        logger.info("  Groq AI (Realtime): Ready")
        logger.info("  Brain (Groq): Ready")
        logger.info("  Chat Service: Ready")
        logger.info("=" * 60)

        logger.info("E.D.I.T.H is online and ready!")
        logger.info("API: http://localhost:8000")
        logger.info("Frontend: http://localhost:8000/app/ (open in browser)")
        logger.info("=" * 60)

        yield

        logger.info("\nShutting down E.D.I.T.H...")
        _tts_pool.shutdown(wait=True)
        if chat_service:
            for session_id in list(chat_service.sessions.keys()):
                chat_service.save_chat_session(session_id)
        logger.info("All sessions saved. Goodbye!")

    except Exception as e:
        logger.error(f"Fatal error during startup: {e}", exc_info=True)
        raise


app = FastAPI(
    title="E.D.I.T.H API",
    description="Just A Rather Very Intelligent System",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

ALLOWED_ORIGINS = [
    o.strip() for o in
    os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - t0
        path = request.url.path
        logger.info("[REQUEST] %s %s -> %s (%.3fs)", request.method, path, response.status_code, elapsed)
        return response

app.add_middleware(TimingMiddleware)

@app.get("/api")
async def api_info():
    return {
        "message": "E.D.I.T.H API",
        "endpoints": {
            "/chat": "General chat (non-streaming)",
            "/chat/stream": "General chat (streaming chunks)",
            "/chat/realtime": "Realtime chat (non-streaming)",
            "/chat/realtime/stream": "Realtime chat (streaming chunks)",
            "/chat/jarvis/stream": "Jarvis unified route (brain classifies, streams)",
            "/chat/history/{session_id}": "Get chat history",
            "/health": "System health check",
            "/tts": "Text-to-speech (POST text, returns streamed MP3)"
        }
    }

@app.get("/health")
async def health():
    try:
        return {
            "status": "healthy",
            "vector_store": vector_store_service is not None,
            "groq_service": groq_service is not None,
            "realtime_service": realtime_service is not None,
            "brain_service": brain_service is not None,
            "chat_service": chat_service is not None
        }
    except Exception as e:
        logger.warning("[API /health] Error: %s", e)
        return {"status": "degraded", "error": str(e)}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    logger.info("[API /chat] Incoming | session_id=%s | message_len=%d | message=%.100s",
                request.session_id or "new", len(request.message), request.message)

    try:
        session_id = chat_service.get_or_create_session(request.session_id)
        response_text = chat_service.process_message(session_id, request.message)
        chat_service.save_chat_session(session_id)

        logger.info("[API /chat] Done | session_id=%s | response_len=%d",
                    session_id[:12], len(response_text))

        return ChatResponse(response_text=response_text, session_id=session_id)

    except ValueError as e:
        logger.warning("[API /chat] Invalid session_id: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    except AllGroqApisFailedError as e:
        logger.error("[API /chat] All Groq APIs failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        if _is_rate_limit_error(e):
            logger.warning("[API /chat] Rate limit hit: %s", e)
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)

        logger.error("[API /chat] Error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")
    
_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+")
_MIN_WORDS_FIRST = 2
_MIN_WORDS = 3
_MERGE_IF_WORDS = 2

def _split_sentences(buf: str):
    parts = _SPLIT_RE.split(buf)
    if len(parts) <= 1:
        return [], buf
    raw: list[str] = [p.strip() for p in typing.cast(list[str], parts[:-1]) if p.strip()]
    sentences: list[str] = []
    pending: str = ""
    for s in raw:
        if pending:
            s = (pending + " " + s).strip()
            pending = ""
        min_req = _MIN_WORDS_FIRST if not sentences else _MIN_WORDS
        if len(s.split()) < min_req:
            pending = s
            continue
        sentences.append(s)
    remaining = (pending + " " + parts[-1].strip()).strip() if pending else parts[-1].strip()
    return sentences, remaining


def _merge_short(sentences):
    if not sentences:
        return []
    merged: list[str] = []
    i: int = 0
    while i < len(sentences):
        cur: str = sentences[i]
        j: int = i + 1
        while j < len(sentences) and len(sentences[j].split()) <= _MERGE_IF_WORDS:
            cur = (cur + " " + sentences[j]).strip()
            j += 1  # type: ignore
        merged.append(cur)
        i = j
    return merged


def _generate_tts_sync(text: str, voice: str, rate: str) -> bytes:
    """Run edge-tts in a dedicated event loop synchronously.
    
    Hardened for production:
      - Uses a fresh event loop per call (thread-safe, avoids asyncio.run() clashes).
      - Sets the event loop so underlying libraries don't fail missing it.
      - Implements strict 30s timeout to prevent thread pool starvation.
      - Gracefully shuts down async generators to prevent memory leaks / websockets left open.
      - Cancels dangling tasks cleanly before loop closure.
    """
    if not text.strip():
        return b""

    async def _inner() -> bytes:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
        parts = bytearray()
        
        async def _accumulate() -> bytes:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    parts.extend(chunk["data"])
            return bytes(parts)
            
        # 30-second absolute timeout to prevent ThreadPoolExecutor thread starvation
        return await asyncio.wait_for(_accumulate(), timeout=30.0)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_inner())
    except asyncio.TimeoutError:
        logger.error("[TTS-WORKER] Timeout generating audio for text: %s", text[:50])
        return b""
    except Exception as e:
        logger.error("[TTS-WORKER] Generation failed: %s", e)
        return b""
    finally:
        try:
            # 1. Shutdown async generators (closes edge-tts websockets cleanly)
            loop.run_until_complete(loop.shutdown_asyncgens())
            
            # 2. Cancel any dangling tasks (prevents memory leaks over 100+ calls)
            pending = asyncio.all_tasks(loop)
            if pending:
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as cleanup_err:
            logger.error("[TTS-WORKER] Error during loop cleanup: %s", cleanup_err)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    return b""  # satisfy static analysis for unhandled BaseExceptions


_tts_pool = ThreadPoolExecutor(max_workers=4)

async def _stream_generator(
    session_id: str,
    chunk_iter,
    is_realtime: bool,
    tts_enabled: bool = False,
    request: typing.Optional[typing.Any] = None,   # starlette.requests.Request
):
    """Fully async SSE generator — zero blocking calls inside the event loop.

    Architecture
    ------------
    * chunk_iter is an async generator produced by ChatService.  It internally
      manages a daemon thread that bridges the synchronous LangChain/Groq stream
      into asyncio.Queue, so this generator never calls any blocking code.
    * TTS futures run in _tts_pool (ThreadPoolExecutor).
    * A lightweight _watcher task polls request.is_disconnected() every 0.5 s
      and sets cancel_event when the client disconnects.  The main loop checks
      cancel_event before each chunk so it exits promptly.
    * On cancel: chunk_iter is aclose()'d (the internal LangChain thread will
      drain naturally once the queue fills); TTS futures are cancelled.
    """
    # ── cancellation signal ──────────────────────────────────────────────────
    # threading.Event so it can be read from both async and sync contexts.
    cancel_event = threading.Event()

    # ── watcher task ─────────────────────────────────────────────────────────
    # Always create a concrete Task (even when request is None) so the finally
    # block can always call watcher_task.cancel() without an AttributeError.
    async def _watcher() -> None:
        """Poll is_disconnected() every 0.5 s; set cancel_event on disconnect.

        Hardening:
        - Full try/except around is_disconnected() — a closed ASGI transport
          raises RuntimeError/AttributeError; we must never let those crash the
          watcher and leave cancel_event unset.
        - Exits immediately if cancel_event is already set (stream ended first).
        - asyncio.CancelledError is caught and silently swallowed so the watcher
          shuts down cleanly when the main generator cancels it in finally.
        """
        if request is None:
            return
        try:
            while not cancel_event.is_set():
                await asyncio.sleep(0.5)
                if cancel_event.is_set():   # stream may have ended while sleeping
                    return
                try:
                    disconnected = await request.is_disconnected()
                except Exception as _conn_err:
                    # Transport closed / ASGI layer gone — treat as disconnect
                    logger.debug("[STREAM-WATCHER] is_disconnected check failed (%s) — treating as disconnect", _conn_err)
                    disconnected = True
                if disconnected:
                    cancel_event.set()
                    logger.info("[STREAM] Client disconnected — cancel_event set")
                    return
        except asyncio.CancelledError:
            pass  # cancelled by the finally block — expected, clean exit
        except Exception as _w_err:
            # Any other unexpected watcher error: log and set cancel_event so
            # the main loop can exit cleanly rather than streaming into the void.
            logger.warning("[STREAM-WATCHER] Unexpected error: %s — setting cancel_event", _w_err)
            cancel_event.set()

    watcher_task = asyncio.ensure_future(_watcher())

    # ── initial session_id handshake ────────────────────────────────────────
    yield f"data: {json.dumps({'session_id': session_id, 'chunk': '', 'done': False})}\n\n"

    # ── shared state ────────────────────────────────────────────────────────
    buffer_parts: list[str] = []   # TTS sentence accumulator — list-append avoids O(n²) string copies
    held: typing.Optional[str] = None
    is_first: bool = True
    # List of (asyncio.Future[bytes], sentence_text) pairs waiting for audio
    audio_queue: list[tuple[asyncio.Future, str]] = []
    _stream_completed_normally: bool = False  # track whether we hit the done sentinel
    # ── SSE-level stream bounding (second defence layer) ─────────────────────
    # chat_service already enforces these limits internally; this is a safety
    # net in case chunks arrive from a source that bypasses chat_service, or
    # if the upstream limits are relaxed.  Both constants come from config so
    # they share the same env-variable knobs as the chat_service limits.
    chunk_count: int = 0
    total_chars: int = 0

    # ── TTS helpers ─────────────────────────────────────────────────────────
    def _submit_tts(text: str) -> None:
        """Submit a TTS job and store an asyncio-wrapped future."""
        if not text or not text.strip():
            return
        concurrent_fut = _tts_pool.submit(_generate_tts_sync, text, TTS_VOICE, TTS_RATE)
        async_fut = asyncio.wrap_future(concurrent_fut)
        audio_queue.append((async_fut, text))

    async def _drain_ready() -> typing.AsyncIterator[str]:
        """Yield SSE audio events for every TTS future that is already done
        (non-blocking check — does not await anything that isn't ready yet)."""
        while audio_queue and audio_queue[0][0].done():
            fut: asyncio.Future  # type: ignore[type-arg]
            sent: str
            fut, sent = audio_queue.pop(0)
            try:
                audio = fut.result()  # safe: already done
                b64 = base64.b64encode(audio).decode("ascii")
                yield f"data: {json.dumps({'audio': b64, 'sentence': sent})}\n\n"
            except Exception as exc:
                logger.warning("[TTS-INLINE] Failed for '%s': %s", sent[:40], exc)

    async def _drain_all_remaining() -> typing.AsyncIterator[str]:
        """Await every remaining TTS future in order, respecting a per-chunk
        timeout so a hung TTS job cannot stall the connection indefinitely."""
        TTS_CHUNK_TIMEOUT = 15.0
        for item in audio_queue:
            fut2: asyncio.Future  # type: ignore[type-arg]
            sent2: str
            fut2, sent2 = item
            try:
                audio = await asyncio.wait_for(fut2, timeout=TTS_CHUNK_TIMEOUT)
                b64 = base64.b64encode(audio).decode("ascii")
                yield f"data: {json.dumps({'audio': b64, 'sentence': sent2})}\n\n"
            except asyncio.TimeoutError:
                logger.warning("[TTS-INLINE] Timeout for '%s' (%.0fs)", sent2[:40], TTS_CHUNK_TIMEOUT)
            except Exception as exc:
                logger.warning("[TTS-INLINE] Failed for '%s': %s", sent2[:40], exc)

    def _cancel_tts() -> None:
        """Cancel all pending TTS futures and clear the queue."""
        for fut, _ in audio_queue:
            fut.cancel()
        audio_queue.clear()

    # ── main consumer loop ──────────────────────────────────────────────────
    try:
        async for chunk in chunk_iter:
            # ── disconnect check ─────────────────────────────────────────────
            if cancel_event.is_set():
                logger.info("[STREAM] Cancelling mid-stream (client disconnected)")
                _cancel_tts()
                # Wrap aclose() — if chat_service's finally block raises, we
                # don't want it to propagate and skip our own finally block.
                try:
                    await chunk_iter.aclose()
                except Exception as _close_err:
                    logger.debug("[STREAM] chunk_iter.aclose() raised (ignored): %s", _close_err)
                return

            # Drain any TTS futures that finished while we were awaiting
            if tts_enabled:
                async for ev in _drain_ready():
                    yield ev

            # ── activity / metadata side-channel ────────────────────────────
            if isinstance(chunk, dict) and "_activity" in chunk:
                yield f"data: {json.dumps({'activity': chunk['_activity']})}\n\n"
                continue

            if isinstance(chunk, dict) and "_search_results" in chunk:
                yield f"data: {json.dumps({'type': 'sources', 'sources': chunk['_search_results']})}\n\n"
                continue

            if not chunk:
                continue

            # ── text chunk → SSE ────────────────────────────────────────────
            # ── SSE-level hard limits ────────────────────────────────────────
            chunk_text: str = str(chunk) if not isinstance(chunk, str) else chunk
            chunk_count += 1
            total_chars += len(chunk_text)

            if total_chars >= MAX_STREAM_CHARS:
                trunc_notice = "\n[response truncated]"
                yield f"data: {json.dumps({'chunk': trunc_notice, 'done': False})}\n\n"
                logger.warning(
                    "[STREAM] SSE layer: response exceeded MAX_STREAM_CHARS=%d — stopping",
                    MAX_STREAM_CHARS,
                )
                _stream_completed_normally = True  # clean stop — allow TTS flush
                break

            if chunk_count >= MAX_STREAM_CHUNKS:
                trunc_notice = "\n[response truncated]"
                yield f"data: {json.dumps({'chunk': trunc_notice, 'done': False})}\n\n"
                logger.warning(
                    "[STREAM] SSE layer: hit MAX_STREAM_CHUNKS=%d failsafe — stopping",
                    MAX_STREAM_CHUNKS,
                )
                _stream_completed_normally = True
                break

            yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"

            if not tts_enabled:
                continue

            # Drain again after yielding (TTS jobs may have completed)
            async for ev in _drain_ready():
                yield ev

            # ── TTS sentence detection ──────────────────────────────────────
            # buffer_parts uses list-append to avoid O(n²) string copies;
            # we pass the joined string to _split_sentences which may pop
            # sentences out — we then rebuild buffer_parts from the remainder.
            buffer_parts.append(chunk_text)
            full_buffer: str = "".join(buffer_parts)
            sentences, remaining_buf = _split_sentences(full_buffer)
            sentences = _merge_short(sentences)
            buffer_parts = [str(remaining_buf)]  # reset to unconsumed remainder only

            if held and sentences and len(sentences[0].split()) <= _MERGE_IF_WORDS:
                held_str: str = str(held)
                held = (held_str + " " + sentences[0]).strip()
                sentences = typing.cast(list[str], sentences[1:])

            for i, sent in enumerate(sentences):
                min_w = _MIN_WORDS_FIRST if is_first else _MIN_WORDS
                if len(sent.split()) < min_w:
                    continue
                is_last = (i == len(sentences) - 1)
                if held:
                    _submit_tts(str(held))  # narrow Optional[str] → str
                    held = None
                    is_first = False
                if is_last:
                    held = sent
                else:
                    _submit_tts(sent)
                    is_first = False

        _stream_completed_normally = True

    except asyncio.CancelledError:
        # Generator was cancelled (e.g. server shutdown or ASGI disconnect)
        logger.info("[STREAM] Generator CancelledError — cleaning up")
        _cancel_tts()
        # Do NOT re-raise: we want the finally to run, and returning here is
        # equivalent to StopAsyncIteration from the consumer's perspective.
        return

    except Exception as e:
        # Cancel any pending TTS futures to free thread-pool slots
        _cancel_tts()
        yield f"data: {json.dumps({'chunk': '', 'done': True, 'error': str(e)})}\n\n"
        return

    finally:
        # ── watcher shutdown ─────────────────────────────────────────────────
        # Set cancel_event first so the watcher's while-loop exits on its own
        # without needing an await.  This makes the shutdown safe even if we
        # are here due to GeneratorExit (you cannot await inside GeneratorExit
        # handling in a regular coroutine, but async generators handle it via
        # aclose() which runs the finally synchronously before the event loop
        # resumes — the key is NOT to block on watcher_task here).
        cancel_event.set()          # tell watcher: no need to keep polling
        watcher_task.cancel()       # cancel the asyncio task itself
        # Best-effort await with a short timeout — if the watcher is already
        # done or gets cancelled, both CancelledError and TimeoutError are fine.
        try:
            await asyncio.wait_for(asyncio.shield(watcher_task), timeout=0.3)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass  # watcher is gone or will be gone shortly — that's the goal

    # ── flush remaining TTS after stream ends (only on clean completion) ─────
    if _stream_completed_normally and tts_enabled:
        remaining = "".join(buffer_parts).strip()
        if held:
            held_str: str = str(held)
            if remaining and len(remaining.split()) <= _MERGE_IF_WORDS:
                _submit_tts(held_str + " " + remaining.strip())
            else:
                _submit_tts(held_str)
                if remaining:
                    _submit_tts(remaining)
        elif remaining:
            _submit_tts(remaining)

        async for ev in _drain_all_remaining():
            yield ev

    yield f"data: {json.dumps({'chunk': '', 'done': True, 'session_id': session_id})}\n\n"


@app.post("/chat/stream")
async def chat_stream(http_request: Request, body: ChatRequest):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    logger.info("[API /chat/stream] Incoming | session_id=%s | message_len=%d | message=%.100s",
                body.session_id or "new", len(body.message), body.message)

    try:
        assert chat_service is not None
        session_id = chat_service.get_or_create_session(body.session_id)
        chunk_iter = chat_service.process_message_stream(session_id, body.message)

        return StreamingResponse(
            _stream_generator(session_id, chunk_iter, is_realtime=False,
                              tts_enabled=body.tts, request=http_request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AllGroqApisFailedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        if _is_rate_limit_error(e):
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
        logger.error("[API /chat/stream] Error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/realtime", response_model=ChatResponse)
async def chat_realtime(request: ChatRequest):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    if not realtime_service:
        raise HTTPException(status_code=503, detail="Realtime service not initialized")

    logger.info("[API /chat/realtime] Incoming | session_id=%s | message_len=%d | message=%.100s",
                request.session_id or "new", len(request.message), request.message)

    try:
        assert chat_service is not None
        session_id = chat_service.get_or_create_session(request.session_id)
        response_text = chat_service.process_realtime_message(session_id, request.message)
        chat_service.save_chat_session(session_id)

        logger.info("[API /chat/realtime] Done | session_id=%s | response_len=%d",
                    session_id[:12], len(response_text))

        return ChatResponse(response_text=response_text, session_id=session_id)

    except ValueError as e:
        logger.warning("[API /chat/realtime] Invalid session_id: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except AllGroqApisFailedError as e:
        logger.error("[API /chat/realtime] All Groq APIs failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        if _is_rate_limit_error(e):
            logger.warning("[API /chat/realtime] Rate limit hit: %s", e)
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
        logger.error("[API /chat/realtime] Error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.post("/chat/realtime/stream")
async def chat_realtime_stream(http_request: Request, body: ChatRequest):
    if not chat_service or not realtime_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    logger.info("[API /chat/realtime/stream] Incoming | session_id=%s | message_len=%d | message=%.100s",
                body.session_id or "new", len(body.message), body.message)

    try:
        assert chat_service is not None
        session_id = chat_service.get_or_create_session(body.session_id)
        chunk_iter = chat_service.process_realtime_message_stream(session_id, body.message)

        return StreamingResponse(
            _stream_generator(session_id, chunk_iter, is_realtime=True,
                              tts_enabled=body.tts, request=http_request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AllGroqApisFailedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        if _is_rate_limit_error(e):
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
        logger.error("[API /chat/realtime/stream] Error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/jarvis/stream")
async def chat_jarvis_stream(http_request: Request, body: ChatRequest):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    logger.info("[API /chat/jarvis/stream] Incoming | session_id=%s | message_len=%d | message=%.100s",
                body.session_id or "new", len(body.message), body.message)

    try:
        assert chat_service is not None
        session_id = chat_service.get_or_create_session(body.session_id)
        chunk_iter = chat_service.process_jarvis_message_stream(session_id, body.message)

        return StreamingResponse(
            _stream_generator(session_id, chunk_iter, is_realtime=True,
                              tts_enabled=body.tts, request=http_request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AllGroqApisFailedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        if _is_rate_limit_error(e):
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
        logger.error("[API /chat/jarvis/stream] Error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    if not chat_service.validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    try:
        messages = chat_service.get_chat_history(session_id)
        return {
            "session_id": session_id,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages]
        }

    except Exception as e:
        logger.error("Error retrieving history: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")


# ── Upload ingestion ──────────────────────────────────────────────────────────

_UPLOAD_MAX_BYTES = 5 * 1024 * 1024  # 5 MB


def process_uploaded_text(text: str, session_id: str) -> None:
    """
    Background ingestion pipeline: TEXT → CHUNKS → EMBEDDINGS → FAISS.

    Uses the app-level vector_store_service (initialised at startup) so no
    extra setup is needed. All errors are caught — server is never affected.
    """
    from datetime import datetime, timezone
    from langchain_core.documents import Document

    if not text or not text.strip():
        logger.warning("[UPLOAD] process_uploaded_text: empty text for session=%s", session_id)
        return

    if vector_store_service is None:
        logger.error("[UPLOAD] vector_store_service is not ready — cannot index upload")
        return

    # Determine a human-readable source label
    source_label = f"upload_{session_id[:8]}"
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(
        "[UPLOAD] Starting ingestion | session_id=%s | chars=%d | source=%s",
        session_id,
        len(text),
        source_label,
    )

    try:
        doc = Document(
            page_content=text,
            metadata={
                "source": source_label,
                "session_id": session_id,
                "timestamp": timestamp,
                "type": "upload",
            },
        )

        added = vector_store_service.add_documents([doc])

        if added > 0:
            logger.info(
                "[UPLOAD] Ingestion complete | session_id=%s | chunks_added=%d | source=%s",
                session_id,
                added,
                source_label,
            )
        else:
            logger.warning(
                "[UPLOAD] Ingestion produced 0 chunks | session_id=%s | source=%s",
                session_id,
                source_label,
            )

    except Exception as exc:
        logger.error(
            "[UPLOAD] Ingestion failed | session_id=%s | error=%s",
            session_id,
            exc,
            exc_info=True,
        )


def _clean_text(raw: str) -> str:
    """Normalise line endings and strip excessive whitespace."""
    # Unify CRLF / CR → LF
    normalised = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of blank lines to a single blank line
    normalised = re.sub(r"\n{3,}", "\n\n", normalised)
    return normalised.strip()


@app.post("/upload")
async def upload_text(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
    session_id: Optional[str] = Form(default=None),
):
    """
    Accept user content via:
      - multipart/form-data  { file: <.txt file>, session_id: (optional) }
      - multipart/form-data  { text: "pasted content", session_id: (optional) }

    Returns immediately; actual processing runs in the background.
    """
    import uuid

    # ── resolve session id ────────────────────────────────────────────────────
    sid = (session_id or "").strip() or str(uuid.uuid4())

    # ── branch: file upload ───────────────────────────────────────────────────
    if file is not None:
        filename = file.filename or ""
        if not filename.lower().endswith(".txt"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type '{filename}'. Only .txt files are accepted.",
            )

        raw_bytes = await file.read()

        if len(raw_bytes) > _UPLOAD_MAX_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({len(raw_bytes):,} bytes). Maximum allowed size is 5 MB.",
            )

        try:
            extracted = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                extracted = raw_bytes.decode("latin-1")
                logger.warning("[UPLOAD] UTF-8 decode failed, fell back to latin-1 for '%s'", filename)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="File could not be decoded. Ensure it is a plain-text UTF-8 file.",
                )

        logger.info(
            "[UPLOAD] File received | name=%s | size=%d bytes | session_id=%s",
            filename,
            len(raw_bytes),
            sid,
        )

    # ── branch: pasted text ───────────────────────────────────────────────────
    elif text is not None:
        extracted = text
        logger.info(
            "[UPLOAD] Text received | len=%d chars | session_id=%s",
            len(text),
            sid,
        )

    # ── neither provided ──────────────────────────────────────────────────────
    else:
        raise HTTPException(
            status_code=400,
            detail="No input provided. Send a .txt file as 'file' or pasted content as 'text'.",
        )

    # ── clean & validate content ──────────────────────────────────────────────
    content = _clean_text(extracted)
    if not content:
        raise HTTPException(status_code=400, detail="Input is empty after cleaning.")

    # ── schedule background processing ───────────────────────────────────────
    background_tasks.add_task(asyncio.to_thread, process_uploaded_text, content, sid)

    return {
        "status": "processing",
        "message": "Input received",
        "session_id": sid,
        "chars": len(content),
    }


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    async def generate():
        try:
            communicate = edge_tts.Communicate(text=text, voice=TTS_VOICE, rate=TTS_RATE)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        except Exception as e:
            logger.error("[TTS] Error generating speech: %s", e)

    return StreamingResponse(
        generate(),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-cache"},
    )


_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/app/", status_code=302)

