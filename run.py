"""
RUN SCRIPT - Start the E.D.I.T.H server
=========================================

PURPOSE:
Single entry point to start the backend. Run this once per user/machine;
the server then handles all chat and realtime requests for that instance.

WHAT IT DOES:
- Imports the FastAPI app from app.main.
- Runs it with uvicorn on host 0.0.0.0 (accept connections from any interface) and port 8000.
- ENVIRONMENT=development enables hot-reload (single worker, file watcher).
- ENVIRONMENT=production disables reload and uses a single worker (FAISS/sessions are in-memory
  and cannot be shared across OS processes — always use workers=1 with this architecture).

USAGE:
    python run.py                          # development mode (default)
    ENVIRONMENT=production python run.py   # production mode (no reload)

Then open http://localhost:8000 in the browser, or use the API from another app.

NOTE:
Before running, set GROQ_API_KEY (and optionally TAVILY_API_KEY for realtime search) in .env.

⚠️  MULTI-WORKER WARNING:
    This app uses in-memory FAISS and in-memory session storage.
    Running workers > 1 creates N independent in-memory stores — sessions and
    file uploads made on worker A are invisible to worker B.
    Keep workers=1 until you migrate to a shared store (e.g. Redis + Chroma/Qdrant).
"""

import os
import uvicorn


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

# Only run uvicorn when this file is executed directly (python run.py),
# not when it is imported by another module.
if __name__ == "__main__":
    # Default to 'development' — plain `python run.py` is always safe.
    # Set ENVIRONMENT=production explicitly when deploying to a server.
    is_dev = os.getenv("ENVIRONMENT", "development").lower() == "development"

    # IMPORTANT: reload=True and workers>1 are mutually exclusive in Uvicorn.
    #   reload=True  → spawns a file-watcher parent + exactly 1 worker child
    #   workers>1    → forks N independent processes, each loading FAISS/sessions
    # Never combine them — the result is N×FAISS loads and split session state.
    # This architecture requires workers=1 in ALL modes.
    uvicorn.run(
        "app.main:app",  # module:variable path to the FastAPI instance
        host="0.0.0.0",  # accept connections from any network interface
        port=8000,        # HTTP port; change if 8000 is already in use
        reload=is_dev,    # hot-reload in dev only
        workers=1,        # always 1 — in-memory state cannot be shared across processes
    )
