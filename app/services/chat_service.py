import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import List, Optional, Dict, Iterator, Any, Union, AsyncIterator
import uuid
import threading

from config import (
    CHATS_DATA_DIR, MAX_CHAT_HISTORY_TURNS, GROQ_API_KEYS,
    MAX_STREAM_CHARS, MAX_STREAM_CHUNKS,
)
from app.models import ChatMessage, ChatHistory
from app.services.groq_service import GroqService, AllGroqApisFailedError
from app.services.realtime_service import RealtimeGroqService
from app.services.brain_service import BrainService
from app.utils.key_rotation import get_next_key_pair

try:
    from app.services.anthropic_service import AnthropicService, AnthropicServiceError
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    AnthropicService = None          # type: ignore[assignment,misc]
    AnthropicServiceError = Exception  # type: ignore[assignment,misc]

logger = logging.getLogger("E.D.I.T.H")

JARVIS_BRAIN_SEARCH_TIMEOUT = 15
SAVE_EVERY_N_CHUNKS = 5


class ChatService:
    # Keywords that signal a query needs deep reasoning (Anthropic)
    _DEEP_KEYWORDS = frozenset([
        "analyze", "analyse", "explain deeply", "summarize", "summarise",
        "compare", "in detail", "step by step", "break down", "break it down",
        "evaluate", "critique", "contrast", "elaborate", "deep dive",
        "pros and cons", "advantages and disadvantages",
    ])
    _DEEP_LENGTH_THRESHOLD = 300  # chars

    def __init__(
        self,
        groq_service: GroqService,
        realtime_service: Optional[RealtimeGroqService] = None,
        brain_service: Optional[BrainService] = None,
        anthropic_service=None,
    ):
        self.groq_service = groq_service
        self.realtime_service = realtime_service
        self.brain_service = brain_service
        self.anthropic_service = anthropic_service
        self.sessions: Dict[str, List[ChatMessage]] = {}

        # ── per-session locking ───────────────────────────────────────────────
        # One RLock per session_id.  Unrelated sessions never block each other.
        # _locks_guard is a plain Lock used ONLY to safely initialise new per-
        # session RLocks — it is held for microseconds and never during I/O or
        # LLM calls, so it cannot deadlock or become a bottleneck.
        self._session_locks: Dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    # ── session lock helpers ──────────────────────────────────────────────────

    def _get_session_lock(self, session_id: str) -> threading.RLock:
        """Return the RLock for session_id, creating it lazily and thread-safely.

        _locks_guard is held ONLY long enough to insert a new RLock — never
        during session reads or writes — so contention is negligible.
        Uses double-checked locking to skip the guard on the hot path.
        """
        lock = self._session_locks.get(session_id)
        if lock is not None:
            return lock
        with self._locks_guard:
            # Re-check: another thread may have created it between get() and here
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.RLock()
            return self._session_locks[session_id]

    def delete_session(self, session_id: str) -> None:
        """Remove session data AND its lock to prevent unbounded memory growth."""
        with self._get_session_lock(session_id):
            self.sessions.pop(session_id, None)
        with self._locks_guard:
            self._session_locks.pop(session_id, None)

    def load_session_from_disk(self, session_id: str) -> bool:
        safe_session_id = session_id.replace("..", "").replace("/", "_")
        filename = f"chat_{safe_session_id}.json"
        filepath = CHATS_DATA_DIR / filename

        if not filepath.exists():
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                chat_dict = json.load(f)

            messages = []
            for msg in chat_dict.get("messages", []):
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                role = role if role in ("user", "assistant") else "user"
                content = msg.get("content")
                content = content if isinstance(content, str) else str(content or "")
                messages.append(ChatMessage(role=role, content=content))

            with self._get_session_lock(session_id):
                self.sessions[session_id] = messages
            return True

        except Exception as e:
            logger.warning("Failed to load session %s from disk: %s", session_id, e)
            return False

    def validate_session_id(self, session_id: str) -> bool:
        if not session_id or not session_id.strip():
            return False
        if "\0" in session_id:
            return False
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            return False
        if len(session_id) > 255:
            return False
        return True

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        t0 = time.perf_counter()

        if not session_id:
            new_session_id = str(uuid.uuid4())
            with self._get_session_lock(new_session_id):
                self.sessions[new_session_id] = []
            logger.info("[TIMING] session_get_or_create: %.3fs (new)", time.perf_counter() - t0)
            return new_session_id

        if not self.validate_session_id(session_id):
            raise ValueError(
                f"Invalid session_id format: {session_id}. Session ID must be non-empty, "
                "not contain path traversal characters, and be under 255 characters."
            )

        if session_id in self.sessions:
            logger.info("[TIMING] session_get_or_create: %.3fs (memory)", time.perf_counter() - t0)
            return session_id

        if self.load_session_from_disk(session_id):
            logger.info("[TIMING] session_get_or_create: %.3fs (disk)", time.perf_counter() - t0)
            return session_id

        with self._get_session_lock(session_id):
            # Re-check after acquiring lock: concurrent request may have loaded it
            if session_id not in self.sessions:
                self.sessions[session_id] = []
        logger.info("[TIMING] session_get_or_create: %.3fs (new_id)", time.perf_counter() - t0)
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        with self._get_session_lock(session_id):
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            self.sessions[session_id].append(ChatMessage(role=role, content=content))

    def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        with self._get_session_lock(session_id):
            # Return a shallow copy so callers can't mutate the held list
            return list(self.sessions.get(session_id, []))

    def format_history_for_llm(self, session_id: str, exclude_last: bool = False) -> List[tuple]:
        messages = self.get_chat_history(session_id)
        history = []

        messages_to_process = messages[:-1] if exclude_last and messages else messages

        i = 0
        while i < len(messages_to_process) - 1:
            user_msg = messages_to_process[i]
            ai_msg = messages_to_process[i + 1]

            if user_msg.role == "user" and ai_msg.role == "assistant":
                u_content = user_msg.content if isinstance(user_msg.content, str) else str(user_msg.content or "")
                a_content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content or "")
                history.append((u_content, a_content))
                i += 2
            else:
                i += 1

        if len(history) > MAX_CHAT_HISTORY_TURNS:
            history = history[-MAX_CHAT_HISTORY_TURNS:]

        return history

    # ── Anthropic routing helpers ──────────────────────────────────────────────

    def _should_use_anthropic(self, question: str) -> tuple:
        """
        Returns (use_anthropic: bool, reason: str).
        Rule-based, zero-latency — no extra LLM call.
        """
        if self.anthropic_service is None:
            return False, "anthropic_service not available"

        q_lower = question.lower()

        if len(question) > self._DEEP_LENGTH_THRESHOLD:
            return True, f"length={len(question)} > {self._DEEP_LENGTH_THRESHOLD}"

        for kw in self._DEEP_KEYWORDS:
            if kw in q_lower:
                return True, f"keyword='{kw}'"

        return False, "short/simple query"

    def _stream_with_anthropic_fallback(
        self,
        question: str,
        chat_history: list,
    ) -> Iterator[Union[str, Dict[str, Any]]]:
        """
        Try Anthropic first; fall back to Groq on any failure.

        Uses a peek pattern: we advance the Anthropic generator once to flush
        any immediate connection/auth errors BEFORE yielding to the client.
        If Anthropic fails upfront, Groq fallback fires cleanly.
        Mid-stream errors (network drop after first chunk) are logged — chunks
        are already in-flight so a provider swap is not possible at that point.
        """
        yield {"_activity": {"event": "provider", "provider": "anthropic"}}

        def _groq_fallback(reason: str):
            logger.warning("[ROUTER] Anthropic failed (%s) — falling back to Groq", reason)
            yield {"_activity": {"event": "provider", "provider": "groq-fallback"}}
            _, chat_idx = get_next_key_pair(len(GROQ_API_KEYS), need_brain=False)
            yield from self.groq_service.stream_response(
                question=question,
                chat_history=chat_history,
                key_start_index=chat_idx,
            )

        # Build Anthropic generator without starting to iterate
        try:
            anthropic_gen = self.anthropic_service.stream_response(  # type: ignore[union-attr]
                question=question,
                chat_history=chat_history,
            )
        except Exception as exc:
            yield from _groq_fallback(str(exc))
            return

        # Peek: pull first item to catch auth/connection errors early
        try:
            first = next(anthropic_gen)
        except StopIteration:
            logger.info("[ROUTER] Anthropic returned empty stream")
            return
        except Exception as exc:
            yield from _groq_fallback(str(exc))
            return

        # First chunk received — commit and stream the rest
        logger.info("[ROUTER] Anthropic stream started — committing")
        yield first
        try:
            yield from anthropic_gen
            logger.info("[ROUTER] Anthropic stream completed")
        except Exception as exc:
            # Mid-stream: chunks already sent, cannot swap provider
            logger.error("[ROUTER] Anthropic mid-stream failure (partial response): %s", exc)



    def process_message(self, session_id: str, user_message: str) -> str:
        self.add_message(session_id, "user", user_message)

        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        logger.info("[GENERAL] History pairs sent to LLM: %d", len(chat_history))

        use_anthropic, reason = self._should_use_anthropic(user_message)

        if use_anthropic:
            logger.info("[ROUTER] Deep query → Anthropic (%s)", reason)
            try:
                response = self.anthropic_service.get_response(  # type: ignore[union-attr]
                    question=user_message,
                    chat_history=chat_history,
                )
                logger.info("[ROUTER] Anthropic response | length: %d", len(response))
            except Exception as exc:
                logger.warning("[ROUTER] Anthropic failed (%s) — falling back to Groq", exc)
                _, chat_idx = get_next_key_pair(len(GROQ_API_KEYS), need_brain=False)
                response = self.groq_service.get_response(
                    question=user_message,
                    chat_history=chat_history,
                    key_start_index=chat_idx,
                )
        else:
            logger.info("[ROUTER] Short query → Groq (%s)", reason)
            _, chat_idx = get_next_key_pair(len(GROQ_API_KEYS), need_brain=False)
            response = self.groq_service.get_response(
                question=user_message,
                chat_history=chat_history,
                key_start_index=chat_idx,
            )

        self.add_message(session_id, "assistant", response)
        logger.info("[GENERAL] Response length: %d chars | Preview: %.120s", len(response), response)
        return response

    def process_realtime_message(self, session_id: str, user_message: str) -> str:
        if not self.realtime_service:
            raise ValueError("Realtime service is not initialized. Cannot process realtime queries.")

        logger.info("[REALTIME] Session: %s | User: %.200s", session_id[:12], user_message)
        self.add_message(session_id, "user", user_message)

        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        logger.info("[REALTIME] History pairs sent to LLM: %d", len(chat_history))

        _, chat_idx = get_next_key_pair(len(GROQ_API_KEYS), need_brain=False)

        response = self.realtime_service.get_response(
            question=user_message,
            chat_history=chat_history,
            key_start_index=chat_idx
        )

        self.add_message(session_id, "assistant", response)
        logger.info("[REALTIME] Response length: %d chars | Preview: %.120s", len(response), response)
        return response

    async def process_message_stream(
        self,
        session_id: str,
        user_message: str,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:

        logger.info("[GENERAL-STREAM] Session: %s | User: %.200s", session_id[:12], user_message)
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", "")
        assistant_index = len(self.sessions[session_id]) - 1

        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        logger.info("[GENERAL-STREAM] History pairs sent to LLM: %d", len(chat_history))

        use_anthropic, reason = self._should_use_anthropic(user_message)
        provider_label = "anthropic" if use_anthropic else "groq"

        yield {"_activity": {"event": "query_detected", "message": user_message}}
        yield {"_activity": {"event": "routing", "route": "general", "provider": provider_label}}

        if use_anthropic:
            logger.info("[ROUTER] Deep query → Anthropic (%s)", reason)
            sync_stream = self._stream_with_anthropic_fallback(user_message, chat_history)
        else:
            logger.info("[ROUTER] Short query → Groq (%s)", reason)
            _, chat_idx = get_next_key_pair(len(GROQ_API_KEYS), need_brain=False)
            sync_stream = self.groq_service.stream_response(
                question=user_message,
                chat_history=chat_history,
                key_start_index=chat_idx,
            )

        yield {"_activity": {"event": "streaming_started", "route": "general", "provider": provider_label}}

        # Bridge the synchronous LangChain iterator into the async world.
        aq: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=256)
        loop = asyncio.get_running_loop()

        def _run_sync_stream() -> None:
            try:
                for c in sync_stream:
                    asyncio.run_coroutine_threadsafe(aq.put(("chunk", c)), loop).result()
                asyncio.run_coroutine_threadsafe(aq.put(("done", None)), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(aq.put(("error", exc)), loop).result()

        threading.Thread(target=_run_sync_stream, daemon=True, name="llm-stream-general").start()

        chunk_count = 0
        char_total = 0
        chunk_parts: list[str] = []   # ✅ O(1) amortised append; join once at end
        truncated = False
        t0 = time.perf_counter()

        try:
            while True:
                item_type, data = await aq.get()

                if item_type == "done":
                    break
                if item_type == "error":
                    raise data  # type: ignore[misc]

                chunk = data
                if isinstance(chunk, dict):
                    yield chunk
                    continue

                if chunk_count == 0:
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    yield {"_activity": {"event": "first_chunk", "route": "general",
                                        "provider": provider_label, "elapsed_ms": elapsed_ms}}

                # ── response-size guard ───────────────────────────────────────
                char_total += len(chunk)
                if char_total > MAX_STREAM_CHARS:
                    notice = " [response truncated]"  # user-visible signal
                    chunk_parts.append(notice)
                    yield notice
                    truncated = True
                    logger.warning(
                        "[GENERAL-STREAM] Response exceeded MAX_STREAM_CHARS=%d — truncating",
                        MAX_STREAM_CHARS,
                    )
                    break

                chunk_parts.append(chunk)
                chunk_count += 1

                # ── chunk-count failsafe ──────────────────────────────────────
                if chunk_count >= MAX_STREAM_CHUNKS:
                    logger.warning(
                        "[GENERAL-STREAM] Hit MAX_STREAM_CHUNKS=%d failsafe — stopping",
                        MAX_STREAM_CHUNKS,
                    )
                    break

                if chunk_count % SAVE_EVERY_N_CHUNKS == 0:
                    await asyncio.to_thread(self.save_chat_session, session_id, False)

                yield chunk

        finally:
            # ── commit accumulated content in one shot (no quadratic growth) ─
            self.sessions[session_id][assistant_index].content = "".join(chunk_parts)
            final_response = self.sessions[session_id][assistant_index].content
            logger.info(
                "[GENERAL-STREAM] Completed | Provider: %s | Chunks: %d | Response: %d chars%s",
                provider_label, chunk_count, len(final_response),
                " (truncated)" if truncated else "",
            )
            await asyncio.to_thread(self.save_chat_session, session_id)

    async def process_realtime_message_stream(
        self,
        session_id: str,
        user_message: str,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:

        if not self.realtime_service:
            raise ValueError("Realtime service is not initialized.")

        logger.info("[REALTIME-STREAM] Session: %s | User: %.200s", session_id[:12], user_message)
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", "")
        assistant_index = len(self.sessions[session_id]) - 1

        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        logger.info("[REALTIME-STREAM] History pairs sent to LLM: %d", len(chat_history))

        yield {"_activity": {"event": "query_detected", "message": user_message}}
        yield {"_activity": {"event": "routing", "route": "realtime"}}
        yield {"_activity": {"event": "streaming_started", "route": "realtime"}}

        _, chat_idx = get_next_key_pair(len(GROQ_API_KEYS), need_brain=False)

        sync_stream = self.realtime_service.stream_response(
            question=user_message,
            chat_history=chat_history,
            key_start_index=chat_idx,
        )

        aq: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=256)
        loop = asyncio.get_running_loop()

        def _run_sync_stream() -> None:
            try:
                for c in sync_stream:
                    asyncio.run_coroutine_threadsafe(aq.put(("chunk", c)), loop).result()
                asyncio.run_coroutine_threadsafe(aq.put(("done", None)), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(aq.put(("error", exc)), loop).result()

        threading.Thread(target=_run_sync_stream, daemon=True, name="llm-stream-realtime").start()

        chunk_count = 0
        char_total = 0
        chunk_parts: list[str] = []   # ✅ list-append to avoid O(n²) string concat
        truncated = False
        t0 = time.perf_counter()

        try:
            while True:
                item_type, data = await aq.get()

                if item_type == "done":
                    break
                if item_type == "error":
                    raise data  # type: ignore[misc]

                chunk = data
                if isinstance(chunk, dict):
                    yield chunk
                    continue

                if chunk_count == 0:
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    yield {"_activity": {"event": "first_chunk", "route": "realtime", "elapsed_ms": elapsed_ms}}

                # ── response-size guard ───────────────────────────────────────
                char_total += len(chunk)
                if char_total > MAX_STREAM_CHARS:
                    notice = " [response truncated]"
                    chunk_parts.append(notice)
                    yield notice
                    truncated = True
                    logger.warning(
                        "[REALTIME-STREAM] Response exceeded MAX_STREAM_CHARS=%d — truncating",
                        MAX_STREAM_CHARS,
                    )
                    break

                chunk_parts.append(chunk)
                chunk_count += 1

                # ── chunk-count failsafe ──────────────────────────────────────
                if chunk_count >= MAX_STREAM_CHUNKS:
                    logger.warning(
                        "[REALTIME-STREAM] Hit MAX_STREAM_CHUNKS=%d failsafe — stopping",
                        MAX_STREAM_CHUNKS,
                    )
                    break

                if chunk_count % SAVE_EVERY_N_CHUNKS == 0:
                    await asyncio.to_thread(self.save_chat_session, session_id, False)

                yield chunk

        finally:
            self.sessions[session_id][assistant_index].content = "".join(chunk_parts)
            final_response = self.sessions[session_id][assistant_index].content
            logger.info(
                "[REALTIME-STREAM] Completed | Chunks: %d | Response: %d chars%s",
                chunk_count, len(final_response), " (truncated)" if truncated else "",
            )
            await asyncio.to_thread(self.save_chat_session, session_id)

    async def process_jarvis_message_stream(
        self,
        session_id: str,
        user_message: str,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:

        logger.info("[JARVIS-STREAM] Session: %s | User: %.200s", session_id[:12], user_message)
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", "")
        assistant_index = len(self.sessions[session_id]) - 1

        chat_history = self.format_history_for_llm(session_id, exclude_last=True)

        yield {"_activity": {"event": "query_detected", "message": user_message}}

        brain_idx, chat_idx = get_next_key_pair(len(GROQ_API_KEYS), need_brain=True)

        query_type = "realtime"
        reasoning = "Defaulting to realtime"
        brain_elapsed_ms = 0
        formatted_results = ""
        search_payload = None

        def _run_brain():
            if self.brain_service and brain_idx is not None:
                qt, rs, ms = self.brain_service.classify(user_message, chat_history, key_index=brain_idx)
                return (qt, rs, ms)
            return ("realtime", "No brain service", 0)

        def _run_search():
            if self.realtime_service:
                return self.realtime_service.prefetch_web_search(user_message, chat_history)
            return ("", None)

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_brain = executor.submit(_run_brain)
            future_search = executor.submit(_run_search)

            try:
                query_type, reasoning, brain_elapsed_ms = future_brain.result(timeout=JARVIS_BRAIN_SEARCH_TIMEOUT)
            except FuturesTimeoutError:
                logger.warning("[JARVIS] Brain classification timed out after %ds, defaulting to realtime",
                               JARVIS_BRAIN_SEARCH_TIMEOUT)
                query_type, reasoning, brain_elapsed_ms = "realtime", "Brain timeout, defaulting to realtime", 0

            if query_type == "general":
                formatted_results, search_payload = "", None
            else:
                try:
                    formatted_results, search_payload = future_search.result(timeout=JARVIS_BRAIN_SEARCH_TIMEOUT)
                except FuturesTimeoutError:
                    logger.warning("[JARVIS] Web search prefetch timed out after %ds", JARVIS_BRAIN_SEARCH_TIMEOUT)
                    formatted_results, search_payload = "", None

        logger.info("[JARVIS] Brain: %s in %d ms — %s", query_type, brain_elapsed_ms, reasoning)

        yield {"_activity": {"event": "decision", "query_type": query_type, "reasoning": reasoning, "elapsed_ms": brain_elapsed_ms}}
        yield {"_activity": {"event": "routing", "route": query_type}}

        if query_type == "realtime" and search_payload:
            yield {"_search_results": search_payload}

        yield {"_activity": {"event": "streaming_started", "route": query_type}}

        if query_type == "general":
            sync_stream = self.groq_service.stream_response(
                question=user_message,
                chat_history=chat_history,
                key_start_index=chat_idx,
            )
        else:
            if not self.realtime_service:
                raise ValueError("Realtime service not initialized.")
            sync_stream = self.realtime_service.stream_response_with_prefetched(
                question=user_message,
                chat_history=chat_history,
                formatted_results=formatted_results,
                payload=search_payload,
                key_start_index=chat_idx,
            )

        aq: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=256)
        loop = asyncio.get_running_loop()

        def _run_sync_stream() -> None:
            try:
                for c in sync_stream:
                    asyncio.run_coroutine_threadsafe(aq.put(("chunk", c)), loop).result()
                asyncio.run_coroutine_threadsafe(aq.put(("done", None)), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(aq.put(("error", exc)), loop).result()

        threading.Thread(target=_run_sync_stream, daemon=True, name="llm-stream-jarvis").start()

        chunk_count = 0
        char_total = 0
        chunk_parts: list[str] = []   # ✅ list-append to avoid O(n²) string concat
        truncated = False
        t0 = time.perf_counter()

        try:
            while True:
                item_type, data = await aq.get()

                if item_type == "done":
                    break
                if item_type == "error":
                    raise data  # type: ignore[misc]

                chunk = data
                if isinstance(chunk, dict):
                    yield chunk
                    continue

                if chunk_count == 0:
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    yield {"_activity": {"event": "first_chunk", "route": query_type, "elapsed_ms": elapsed_ms}}

                # ── response-size guard ───────────────────────────────────────
                char_total += len(chunk)
                if char_total > MAX_STREAM_CHARS:
                    notice = " [response truncated]"
                    chunk_parts.append(notice)
                    yield notice
                    truncated = True
                    logger.warning(
                        "[JARVIS-STREAM] Response exceeded MAX_STREAM_CHARS=%d — truncating",
                        MAX_STREAM_CHARS,
                    )
                    break

                chunk_parts.append(chunk)
                chunk_count += 1

                # ── chunk-count failsafe ──────────────────────────────────────
                if chunk_count >= MAX_STREAM_CHUNKS:
                    logger.warning(
                        "[JARVIS-STREAM] Hit MAX_STREAM_CHUNKS=%d failsafe — stopping",
                        MAX_STREAM_CHUNKS,
                    )
                    break

                if chunk_count % SAVE_EVERY_N_CHUNKS == 0:
                    await asyncio.to_thread(self.save_chat_session, session_id, False)

                yield chunk

        finally:
            self.sessions[session_id][assistant_index].content = "".join(chunk_parts)
            final_response = self.sessions[session_id][assistant_index].content
            logger.info(
                "[JARVIS-STREAM] Completed | Route: %s | Chunks: %d | Response: %d chars%s",
                query_type, chunk_count, len(final_response), " (truncated)" if truncated else "",
            )
            await asyncio.to_thread(self.save_chat_session, session_id)

    def save_chat_session(self, session_id: str, log_timing: bool = True):
        # ── snapshot under lock ───────────────────────────────────────────────
        with self._get_session_lock(session_id):
            if session_id not in self.sessions or not self.sessions[session_id]:
                return
            chat_dict = {
                "session_id": session_id,
                "messages": [{"role": msg.role, "content": msg.content} for msg in self.sessions[session_id]]
            }
        # ── file write outside lock ───────────────────────────────────────────
        safe_session_id = session_id.replace("..", "").replace("/", "_")
        filepath = CHATS_DATA_DIR / f"chat_{safe_session_id}.json"

        max_retries = 3
        last_exc = None

        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter() if log_timing else 0
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(chat_dict, f, indent=2, ensure_ascii=False)
                if log_timing:
                    logger.info("[TIMING] save_session_json: %.3fs", time.perf_counter() - t0)
                return

            except OSError as e:
                last_exc = e
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))

            except Exception as e:
                logger.error("Failed to save chat session %s to disk: %s", session_id, e)
                return

        logger.error("Failed to save chat session %s after %d retries: %s", session_id, max_retries, last_exc)