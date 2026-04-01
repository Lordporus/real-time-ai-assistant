"""
AnthropicService — Claude provider for EDITH.

Drop-in parallel to GroqService. Interface is intentionally identical so it
can be wired into ChatService / main.py with minimal changes later.

NOT connected to any chat flow yet. Safe to import without side effects.
"""

from typing import List, Optional, Iterator, Union
import logging
import time
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from config import JARVIS_SYSTEM_PROMPT, GENERAL_CHAT_ADDENDUM
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information

logger = logging.getLogger("EDITH.anthropic")

ANTHROPIC_MODEL      = "claude-3-5-sonnet-20241022"
ANTHROPIC_TIMEOUT    = 60
ANTHROPIC_MAX_TOKENS = 8192   # Claude requires an explicit max_tokens

ANTHROPIC_UNAVAILABLE = (
    "I'm unable to process your request at the moment. "
    "The Anthropic service is temporarily unavailable. Please try again in a few minutes."
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

# ── helpers (mirrored from groq_service for consistency) ─────────────────────

def _escape_curly_braces(text: str) -> str:
    """Escape { } so LangChain doesn't treat them as template variables."""
    if not text:
        return text
    return text.replace("{", "{{").replace("}", "}}")


def _log_timing(label: str, elapsed: float, extra: str = "") -> None:
    msg = f"[TIMING][ANTHROPIC] {label}: {elapsed:.3f}s"
    if extra:
        msg += f" ({extra})"
    logger.info(msg)


# ── service ───────────────────────────────────────────────────────────────────

class AnthropicServiceError(Exception):
    """Raised when Anthropic API fails."""


class AnthropicService:
    """
    Claude provider — mirrors the public interface of GroqService:

        get_response(question, chat_history)  → str
        stream_response(question, chat_history) → Iterator[str | dict]

    Usage:
        service = AnthropicService(vector_store_service)
        for chunk in service.stream_response("Hello"):
            print(chunk, end="", flush=True)
    """

    def __init__(self, vector_store_service: VectorStoreService) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-..."
            )

        self.llm = ChatAnthropic(
            api_key=api_key,                   # type: ignore[arg-type]
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            temperature=0.5,
            timeout=ANTHROPIC_TIMEOUT,
            streaming=True,
        )

        self.vector_store_service = vector_store_service
        logger.info(
            "AnthropicService initialized — model: %s | key: %s...%s",
            ANTHROPIC_MODEL,
            api_key[:8],
            api_key[-4:],
        )

    # ── internal: build prompt ────────────────────────────────────────────────

    def _build_prompt_and_messages(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        mode_addendum: str = "",
    ):
        """
        Assemble system prompt + history messages, identical logic to GroqService.
        Returns (ChatPromptTemplate, list[BaseMessage]).
        """

        # --- Vector store retrieval ------------------------------------------
        context = ""
        t0 = time.perf_counter()
        try:
            expanded_query = question
            if any(w in question.lower() for w in ["file", "upload", "document", "data"]):
                expanded_query = question + " uploaded file content knowledge document"

            docs_and_scores = self.vector_store_service.vector_store.similarity_search_with_relevance_scores(
                expanded_query, k=10
            )

            # ── Debug: log every score ────────────────────────────────────────
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
                context = "\n".join(d.page_content for d in context_docs)
                sources = [d.metadata.get("source", "unknown") for d in context_docs]
                logger.info("[FINAL_CONTEXT] %d chunks | sources: %s", len(context_docs), sources)
            else:
                logger.info("[CONTEXT] No relevant chunks found for query")
        except Exception as err:
            logger.warning("Vector store retrieval failed (Anthropic), using empty context: %s", err)
        finally:
            _log_timing("vector_db", time.perf_counter() - t0)

        # --- Build system message -------------------------------------------
        time_info = get_time_information()
        system_msg = JARVIS_SYSTEM_PROMPT
        system_msg += f"\n\nCurrent time and date: {time_info}"

        if context:
            system_msg += (
                f"\n\nRelevant context from your learning data and past conversations:\n"
                f"{_escape_curly_braces(context)}"
            )
            system_msg += """

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

        system_msg += f"\n\n{_MEMORY_USAGE_ADDENDUM}"

        if mode_addendum:
            system_msg += f"\n\n{mode_addendum}"

        # --- Build prompt template ------------------------------------------
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        messages: list = []
        if chat_history:
            for human_msg, ai_msg in chat_history:
                messages.append(HumanMessage(content=human_msg))
                messages.append(AIMessage(content=ai_msg))

        logger.info(
            "[PROMPT] system: %d chars | history_pairs: %d | question: %.100s",
            len(system_msg),
            len(chat_history) if chat_history else 0,
            question,
        )

        return prompt, messages

    # ── public: non-streaming ─────────────────────────────────────────────────

    def get_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
    ) -> str:
        """
        Non-streaming response. Returns the full text as a single string.
        Mirrors GroqService.get_response().
        """
        try:
            prompt, messages = self._build_prompt_and_messages(
                question,
                chat_history,
                mode_addendum=GENERAL_CHAT_ADDENDUM,
            )

            t0 = time.perf_counter()
            chain = prompt | self.llm
            result = chain.invoke({"history": messages, "question": question})
            _log_timing("anthropic_api", time.perf_counter() - t0)

            content = result.content if hasattr(result, "content") else str(result)
            logger.info(
                "[RESPONSE] Anthropic | length: %d chars | preview: %.120s",
                len(content),
                content,
            )
            return content

        except AnthropicServiceError:
            raise
        except Exception as exc:
            logger.error("Anthropic get_response failed: %s", exc)
            raise AnthropicServiceError(ANTHROPIC_UNAVAILABLE) from exc

    # ── public: streaming ─────────────────────────────────────────────────────

    def stream_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
    ) -> Iterator[Union[str, dict]]:
        """
        Streaming response — yields str tokens and optionally dict activity events.
        Mirrors GroqService.stream_response() so main.py's _stream_generator
        can consume it without any changes.
        """
        try:
            prompt, messages = self._build_prompt_and_messages(
                question,
                chat_history,
                mode_addendum=GENERAL_CHAT_ADDENDUM,
            )

            # Activity event — same pattern as GroqService
            yield {
                "_activity": {
                    "event": "context_retrieved",
                    "message": "Retrieved relevant context from knowledge base (Anthropic)",
                }
            }

            chain = prompt | self.llm

            chunk_count = 0
            first_chunk_time: Optional[float] = None
            stream_start = time.perf_counter()

            for chunk in chain.stream({"history": messages, "question": question}):
                content = ""

                if hasattr(chunk, "content"):
                    content = chunk.content or ""
                elif isinstance(chunk, dict) and "content" in chunk:
                    content = chunk.get("content", "") or ""

                if isinstance(content, str) and content:
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter() - stream_start
                        _log_timing("first_chunk", first_chunk_time)
                    chunk_count += 1
                    yield content

            _log_timing(
                "anthropic_stream_total",
                time.perf_counter() - stream_start,
                f"chunks: {chunk_count}",
            )

        except AnthropicServiceError:
            raise
        except Exception as exc:
            logger.error("Anthropic stream_response failed: %s", exc)
            raise AnthropicServiceError(ANTHROPIC_UNAVAILABLE) from exc
