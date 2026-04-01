import json
import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.vectorstores import FAISS
#from langchain_core.documents import Document

from config import (
    LEARNING_DATA_DIR,
    CHATS_DATA_DIR,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logger = logging.getLogger("E.D.I.T.H")


class VectorStoreService:
    def __init__(self):
        #self.embeddings = HuggingFaceEmbeddings(
        #    model_name=EMBEDDING_MODEL,
        #    model_kwargs={"device": "cpu"},
        #)

        self.embeddings = None

########################################################

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        self.vector_store: Optional[FAISS] = None

        # cache now includes (k, source_filter)
        self._retriever_cache: dict[Tuple[int, Optional[str]], any] = {}

        # ── concurrency guard ─────────────────────────────────────────────
        # FAISS's underlying C++ index is NOT thread-safe: concurrent reads
        # and writes can produce segfaults, corrupted results, or silent data
        # loss.  A single RLock serialises every FAISS operation while still
        # being re-entrant so that methods that call each other (e.g.
        # create_vector_store → save_vector_store) don't deadlock.
        self._faiss_lock = threading.RLock()

    # ─────────────────────────────────────────────
    # LOAD DATA  (pure Python / disk I/O — no FAISS, no lock needed)
    # ─────────────────────────────────────────────

    def load_learning_data(self) -> List[Document]:
        documents = []

        for file_path in sorted(LEARNING_DATA_DIR.glob("*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if content:
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": str(file_path.name), "type": "learning"},
                        )
                    )
                    logger.info(
                        "[VECTOR] Loaded learning data: %s (%d chars)",
                        file_path.name,
                        len(content),
                    )

            except Exception as e:
                logger.warning("Could not load learning data file %s: %s", file_path, e)

        logger.info("[VECTOR] Total learning data files loaded: %d", len(documents))
        return documents

    def load_chat_history(self) -> List[Document]:
        documents = []

        for file_path in sorted(CHATS_DATA_DIR.glob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chat_data = json.load(f)

                messages = chat_data.get("messages", [])

                if not isinstance(messages, list):
                    continue

                lines = []

                for msg in messages:
                    if not isinstance(msg, dict):
                        continue

                    role = msg.get("role") or "assistant"
                    content = msg.get("content") or ""

                    prefix = "User: " if role == "user" else "Assistant: "
                    lines.append(prefix + content)

                chat_content = "\n".join(lines)

                if chat_content.strip():
                    documents.append(
                        Document(
                            page_content=chat_content,
                            metadata={"source": f"chat_{file_path.stem}", "type": "chat"},
                        )
                    )

                    logger.info(
                        "[VECTOR] Loaded chat history: %s (%d messages)",
                        file_path.name,
                        len(messages),
                    )

            except Exception as e:
                logger.warning("Could not load chat history file %s: %s", file_path, e)

        logger.info("[VECTOR] Total chat history files loaded: %d", len(documents))
        return documents

    # ─────────────────────────────────────────────
    # CREATE STORE  (write — full lock for entire build)
    # ─────────────────────────────────────────────

    def create_vector_store(self) -> FAISS:
        # Load documents outside the lock — pure disk I/O, no FAISS touch.
        learning_docs = self.load_learning_data()
        chat_docs = self.load_chat_history()
        all_documents = learning_docs + chat_docs

        logger.info(
            "[VECTOR] Total documents to index: %d (learning: %d, chat: %d)",
            len(all_documents),
            len(learning_docs),
            len(chat_docs),
        )
####################################################################
        # Embedding + FAISS index build — CPU-heavy but must be atomic.
        #if not all_documents:
        #    new_store = FAISS.from_texts(["No data available yet."], self.embeddings)
        if self.embeddings is None:
            logger.warning("[VECTOR] No embeddings — skipping store creation")
            return None
            logger.info("[VECTOR] No documents found, created placeholder index")
        else:
            chunks = self.text_splitter.split_documents(all_documents)
            logger.info(
                "[VECTOR] Split into %d chunks (chunk_size=%d, overlap=%d)",
                len(chunks),
                CHUNK_SIZE,
                CHUNK_OVERLAP,
            )
##################################################################
        if self.embeddings is None:
            logger.warning("[VECTOR] Skipped FAISS build (embeddings disabled)")
            return None

        new_store = FAISS.from_documents(chunks, self.embeddings)
        logger.info(
            "[VECTOR] FAISS index built successfully with %d vectors",
            len(chunks),
            )

        # Swap in the new store and persist — hold lock for the swap + save.
        with self._faiss_lock:
            self.vector_store = new_store
            self._retriever_cache.clear()
            self.save_vector_store()   # re-entrant: save_vector_store also acquires

        return self.vector_store  # type: ignore[return-value]

    def save_vector_store(self):
        with self._faiss_lock:
            if self.vector_store:
                try:
                    self.vector_store.save_local(str(VECTOR_STORE_DIR))
                except Exception as e:
                    logger.error("Failed to save vector store to disk: %s", e)

    # ─────────────────────────────────────────────
    # 🔥 FIXED RETRIEVER (MAIN BUG FIX)
    # ─────────────────────────────────────────────

    def get_retriever(self, k: int = 10, source_filter: Optional[str] = None):
        with self._faiss_lock:
            if not self.vector_store:
                raise RuntimeError("Vector store not initialized.")

            cache_key = (k, source_filter)

            if cache_key not in self._retriever_cache:
                search_kwargs = {"k": k}

                # ✅ CRITICAL FIX: only apply filter if valid
                if source_filter:
                    search_kwargs["filter"] = {"source": source_filter}
                    logger.info("[VECTOR] Using source filter: %s", source_filter)
                else:
                    logger.info("[VECTOR] No source filter applied")

                retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
                self._retriever_cache[cache_key] = retriever

            return self._retriever_cache[cache_key]

    # ─────────────────────────────────────────────
    # ADD DOCUMENTS (UPLOAD)  (write — full lock)
    # ─────────────────────────────────────────────

#    def add_documents(self, documents: List[Document]) -> int:
#        if not documents:
#            logger.warning("[VECTOR] add_documents called with empty list")
#            return 0
#
#        try:
#            # Chunking / embedding happens outside the lock (CPU-bound, no FAISS state).
#            chunks = self.text_splitter.split_documents(documents)
#
 #           if not chunks:
#                logger.warning("[VECTOR] No chunks created")
#                return 0
#
#            logger.info("[VECTOR] add_documents: %d → %d chunks", len(documents), len(chunks))
#
#            # Build a temporary index from the new chunks (no shared state yet).
#            new_index = FAISS.from_documents(chunks, self.embeddings)
#
#            # Merge into the live index under the lock — minimum critical section.
# #           with self._faiss_lock:
#                if self.vector_store is None:
#                    self.vector_store = new_index
#                else:
#                    self.vector_store.merge_from(new_index)
#
 #               self._retriever_cache.clear()
 #               self.save_vector_store()   # re-entrant
#
#            return len(chunks)
#
#        except Exception as exc:
#            logger.error("[VECTOR] add_documents failed: %s", exc, exc_info=True)
#            return 0
def add_documents(self, documents: List[Document]) -> int:
    if self.embeddings is None:
        logger.warning("[VECTOR] Skipping add_documents (embeddings disabled)")
        return 0

    if not documents:
        logger.warning("[VECTOR] add_documents called with empty list")
        return 0

    try:
        chunks = self.text_splitter.split_documents(documents)

        if not chunks:
            logger.warning("[VECTOR] No chunks created")
            return 0

        logger.info(
            "[VECTOR] add_documents: %d → %d chunks",
            len(documents),
            len(chunks),
        )

        new_index = FAISS.from_documents(chunks, self.embeddings)

        with self._faiss_lock:
            if self.vector_store is None:
                self.vector_store = new_index
            else:
                self.vector_store.merge_from(new_index)

            self._retriever_cache.clear()
            self.save_vector_store()

        return len(chunks)

    except Exception as exc:
        logger.error("[VECTOR] add_documents failed: %s", exc, exc_info=True)
        return 0