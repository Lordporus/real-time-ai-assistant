from typing import List, Optional, Iterator, Union, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import logging
import time
from langchain_community.tools.tavily_search import TavilySearchResults

from config import (
    GROQ_API_KEYS,
    GROQ_MODEL,
    JARVIS_SYSTEM_PROMPT,
    GENERAL_CHAT_ADDENDUM,
    TAVILY_API_KEY,
)

from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information
from app.utils.retry import with_retry

logger = logging.getLogger("E.D.I.T.H")

GROQ_REQUEST_TIMEOUT = 60
ALL_APIS_FAILED_MESSAGE = (
    "I'm unable to process your request at the moment. All API services are "
    "temporarily unavailable. Please try again in a few minutes."
)

_MEMORY_USAGE_ADDENDUM = """\
You have access to retrieved memory from uploaded documents and stored data.

If relevant context is provided:
- You MUST use it to answer
- You MUST assume it is correct
- You MUST NOT say you don't have access to files
- You MUST NOT ignore the provided context

If the user asks about uploaded files, documents, or data:
- Assume the answer exists in the provided context
- Base your answer on that context

Never say:
- 'I cannot access files'
- 'I don't have that information'

Instead:
- Extract the answer from the provided context"""


class AllGroqApisFailedError(Exception):
    pass


def escape_curly_braces(text: str) -> str:
    if not text:
        return text
    return text.replace("{", "{{").replace("}", "}}")


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "tokens per day" in msg


def _log_timing(label: str, elapsed: float, extra: str = ""):
    msg = f"[TIMING] {label}: {elapsed:.3f}s"
    if extra:
        msg += f" ({extra})"
    logger.info(msg)


def _mask_api_key(key: str) -> str:
    if not key or len(key) < 12:
        return "***masked***"
    s_key = str(key)
    return f"{s_key[0:8]}...{s_key[-4:]}"  # type: ignore


class RealtimeGroqService:

    @staticmethod
    def _expand_query(question: str) -> str:
        q = question.lower()
        if any(word in q for word in ["file", "upload", "document", "data"]):
            return question + " uploaded file content knowledge document"
        return question

    def __init__(self, vector_store_service: VectorStoreService):
        if not GROQ_API_KEYS:
            raise ValueError(
                "No Groq API keys configured. Set GROQ_API_KEY (and optionally GROQ_API_KEY_2, GROQ_API_KEY_3, ...) in .env"
            )

        self.llms = [
            ChatGroq(
                groq_api_key=key,
                model_name=GROQ_MODEL,
                temperature=0.5,
                request_timeout=GROQ_REQUEST_TIMEOUT,
            )
            for key in GROQ_API_KEYS
        ]

        self.vector_store_service = vector_store_service
        
        self.search_tool = None
        if TAVILY_API_KEY:
            self.search_tool = TavilySearchResults(api_key=TAVILY_API_KEY, k=5)
            logger.info("Realtime search enabled (Tavily)")
        else:
            logger.warning("TAVILY_API_KEY not found; realtime search will be disabled")

        logger.info(f"Initialized RealtimeGroqService with {len(GROQ_API_KEYS)} API key(s)")

    def _invoke_llm(
        self,
        prompt: ChatPromptTemplate,
        messages: list,
        question: str,
        key_start_index: int = 0,
    ) -> str:

        n = len(self.llms)
        last_exc = None
        keys_tried = []

        for j in range(n):
            i = (key_start_index + j) % n
            keys_tried.append(i)

            masked_key = _mask_api_key(GROQ_API_KEYS[i])
            logger.info(f"Trying API key #{i + 1}/{n}: ({masked_key})")

            def _invoke_with_key(idx: int = i):
                chain = prompt | self.llms[idx]
                return chain.invoke({"history": messages, "question": question})

            try:
                response = with_retry(
                    _invoke_with_key,
                    max_retries=2,
                    initial_delay=0.5,
                )

                if j > 0:
                    logger.info(f"Fallback successful: API key #{i + 1}/{n} succeeded: ({masked_key})")

                return response.content

            except Exception as e:
                last_exc = e

                if _is_rate_limit_error(e):
                    logger.warning(f"API key #{i + 1}/{n} rate limited: {masked_key}")
                else:
                    logger.warning(f"API key #{i + 1}/{n} failed: {masked_key} - {str(e)[:100]}")  # type: ignore

                if j < n - 1:
                    logger.info("Falling back to next API key...")
                    continue

                break

        masked_all = ", ".join([_mask_api_key(GROQ_API_KEYS[j]) for j in keys_tried])
        logger.error(f"All ({n}) API key(s) failed. Tried: {masked_all}")
        raise AllGroqApisFailedError(ALL_APIS_FAILED_MESSAGE) from last_exc  # type: ignore

    def _stream_llm(
        self,
        prompt: ChatPromptTemplate,
        messages: list,
        question: str,
        key_start_index: int = 0,
    ) -> Iterator[str]:
        """
        Strict sequential failover across API keys.

        Rules:
          - Pre-flight failure (0 chunks yielded) → try next key.
          - Mid-stream failure (≥1 chunks already yielded) → abort immediately;
            never attempt another key (client has partial output, a second LLM
            stream would duplicate content).
          - Success → return immediately; remaining keys are never touched.
        """
        n = len(self.llms)
        last_exc = None
        # True once the FIRST chunk has been yielded to the caller.
        # Prevents the exception handler from ever falling through to the next
        # key after partial output has been sent to the client.
        _stream_exhausted: bool = False

        for j in range(n):
            i = (key_start_index + j) % n

            masked_key = _mask_api_key(GROQ_API_KEYS[i])
            logger.info(f"Streaming with API key #{i + 1}/{n}: ({masked_key})")

            chunk_count = 0
            try:
                chain = prompt | self.llms[i]

                first_chunk_time = None
                stream_start = time.perf_counter()

                for chunk in chain.stream({"history": messages, "question": question}):
                    content = ""

                    if hasattr(chunk, "content"):
                        content = chunk.content or ""
                    elif isinstance(chunk, dict) and "content" in chunk:
                        content = chunk.get("content", "") or ""

                    if isinstance(content, str) and content:
                        if first_chunk_time is None:
                            first_chunk_time = float(time.perf_counter()) - float(stream_start)
                            _log_timing("first_chunk", first_chunk_time)

                        chunk_count += 1
                        _stream_exhausted = True  # partial output is now in-flight
                        yield content

                total_stream = float(time.perf_counter()) - float(stream_start)
                _log_timing("groq_stream_total", total_stream, f"chunks: {chunk_count}")

                if j > 0 and chunk_count > 0:
                    logger.info(f"Fallback successful: API key #{i + 1}/{n} streamed: ({masked_key})")

                return  # ✅ success — stop immediately, no other key runs

            except Exception as e:
                last_exc = e

                if _stream_exhausted:
                    # Partial output already sent to the client — swapping
                    # providers NOW would duplicate content.  Abort hard.
                    logger.error(
                        f"[STREAM] Mid-stream error after {chunk_count} chunk(s) with "
                        f"API key #{i + 1}/{n} ({masked_key}). "
                        "Cannot fall back — partial response already sent."
                    )
                    raise AllGroqApisFailedError("Stream interrupted mid-response") from e

                if _is_rate_limit_error(e):
                    logger.warning(f"API key #{i + 1}/{n} rate limited: {masked_key}")
                else:
                    logger.warning(f"API key #{i + 1}/{n} failed: {masked_key} - {str(e)[:100]}")

                if j < n - 1:
                    logger.info("Falling back to next API key for stream...")
                    continue

                break

        logger.error("All API key(s) failed during stream.")
        raise AllGroqApisFailedError(ALL_APIS_FAILED_MESSAGE) from last_exc  # type: ignore

    def _build_prompt_and_messages(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        extra_system_parts: Optional[List[str]] = None,
        mode_addendum: str = "",
    ):

        context = ""
        context_sources = []

        t0 = time.perf_counter()

        try:
            expanded_query = self._expand_query(question)
            docs_and_scores = self.vector_store_service.vector_store.similarity_search_with_relevance_scores(
                expanded_query, k=10
            )

            # ── Debug: log every score so we can tune the threshold ──────────
            for doc, score in docs_and_scores:
                logger.info(
                    "[SCORE] source=%s score=%.4f",
                    doc.metadata.get("source", "unknown"),
                    score,
                )

            # ── Step 1: filter low-relevance docs (threshold 0.2) ────────────
            all_docs = [doc for doc, score in docs_and_scores if score > 0.2]

            if not all_docs:
                all_docs = [doc for doc, _ in docs_and_scores]

            # ── Step 2: prioritise upload_* > learning > chat ────────────────
            upload_docs   = [d for d in all_docs if str(d.metadata.get("source", "")).startswith("upload_")]
            learning_docs = [d for d in all_docs if not str(d.metadata.get("source", "")).startswith("chat_")
                             and not str(d.metadata.get("source", "")).startswith("upload_")]
            chat_docs     = [d for d in all_docs if str(d.metadata.get("source", "")).startswith("chat_")]

            if upload_docs:
                context_docs = upload_docs + learning_docs
            elif learning_docs:
                context_docs = learning_docs
            else:
                context_docs = chat_docs
                if chat_docs:
                    logger.info("[CONTEXT] Only chat-history chunks available — using them as fallback")

            if context_docs:
                context = "\n".join([doc.page_content for doc in context_docs])
                context_sources = [doc.metadata.get("source", "unknown") for doc in context_docs]
                logger.info("[FINAL_CONTEXT] %d chunks | sources: %s", len(context_docs), context_sources)
            else:
                logger.info("[CONTEXT] No relevant chunks found for query")

        except Exception as retrieval_err:
            logger.warning(f"Vector store retrieval failed, using empty context: {retrieval_err}")

        finally:
            _log_timing("vector_db", time.perf_counter() - t0)

        time_info = get_time_information()

        system_message = JARVIS_SYSTEM_PROMPT
        system_message += f"\n\nCurrent time and date: {time_info}"

        if context:
            system_message += f"\n\nRelevant context from your learning data and past conversations:\n{escape_curly_braces(context)}"
            system_message += """

--- MEMORY PRIORITY OVERRIDE ---

If retrieved context is present:
- Treat it as factual and authoritative
- Answer strictly using this context

You MUST NOT say:
- "I have not received any file"
- "I cannot access uploaded files"
- "I don't have that information"

If context exists, it means the system HAS access.

Always prioritize retrieved context over:
- prior assumptions
- general knowledge
- default responses

--------------------------------
"""

        system_message += f"\n\n{_MEMORY_USAGE_ADDENDUM}"

        if extra_system_parts:
            system_message += "\n\n" + "\n\n".join(extra_system_parts)

        if mode_addendum:
            system_message += f"\n\n{mode_addendum}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        messages = []

        if chat_history:
            for human_msg, ai_msg in chat_history:
                messages.append(HumanMessage(content=human_msg))
                messages.append(AIMessage(content=ai_msg))

        logger.info(
            "[PROMPT] System message length: %d chars | History pairs: %d | Question: %.100s",
            len(system_message),
            len(chat_history) if chat_history else 0,
            question,
        )

        return prompt, messages

    def get_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        key_start_index: int = 0,
    ) -> str:

        try:
            prompt, messages = self._build_prompt_and_messages(
                question,
                chat_history,
                mode_addendum=GENERAL_CHAT_ADDENDUM,
            )

            t0 = time.perf_counter()

            result = self._invoke_llm(
                prompt,
                messages,
                question,
                key_start_index=key_start_index,
            )

            _log_timing("groq_api", time.perf_counter() - t0)

            logger.info(
                "[RESPONSE] General chat | Length: %d chars | Preview: %.120s",
                len(result),
                result,
            )

            return result

        except AllGroqApisFailedError:
            raise
        except Exception as e:
            raise Exception(f"Error getting response from Groq: {str(e)}") from e

    def stream_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        key_start_index: int = 0,
    ) -> Iterator[Union[str, dict]]:

        try:
            results_text, payload = self.prefetch_web_search(question, chat_history)
            yield from self.stream_response_with_prefetched(
                question,
                chat_history,
                results_text,
                payload,
                key_start_index
            )
        except Exception as e:
            raise Exception(f"Error in realtime stream: {str(e)}") from e

    def prefetch_web_search(self, question: str, chat_history: Optional[List[tuple]] = None):
        if not self.search_tool:
            return "", None

        t0 = time.perf_counter()
        try:
            results = self.search_tool.invoke(question)
            _log_timing("tavily_search", time.perf_counter() - t0)
            
            if not results:
                return "No search results found.", []

            formatted = "\n\n".join([
                f"Source: {r.get('url', 'Unknown')}\nContent: {r.get('content', '')}"
                for r in results
            ])
            return formatted, results
        except Exception as e:
            logger.error(f"Search prefetch failed: {e}")
            return f"Search failed: {str(e)}", None

    def stream_response_with_prefetched(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        formatted_results: str = "",
        payload: Any = None,
        key_start_index: int = 0,
    ) -> Iterator[Union[str, dict]]:

        try:
            extra_system = []
            if formatted_results:
                extra_system.append(f"RELEVANT WEB SEARCH RESULTS:\n{formatted_results}")
            
            prompt, messages = self._build_prompt_and_messages(
                question,
                chat_history,
                extra_system_parts=extra_system,
                mode_addendum="You are in REALTIME mode. Use the search results provided to answer accurately."
            )

            if payload:
                yield {"_search_results": payload}

            yield from self._stream_llm(
                prompt,
                messages,
                question,
                key_start_index=key_start_index,
            )

        except Exception as e:
            raise Exception(f"Error in stream_with_prefetched: {str(e)}") from e