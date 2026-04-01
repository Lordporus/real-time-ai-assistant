"""
AUTHENTICATION MODULE
=====================
Session-based authentication for single-admin access.
Uses bcrypt for password hashing and in-memory session tokens with TTL.

Sessions are stored in a Python dict and cleared on server restart
(forcing re-login). Each session records the username and an expiry
timestamp; expired sessions are purged on every auth check so memory
stays bounded.

USAGE:
    from app.auth import login_required
    @app.get("/protected", dependencies=[Depends(login_required)])
    async def protected_route(): ...
"""

import os
import time
import secrets
import logging

import bcrypt
from fastapi import Request, HTTPException, Response
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("E.D.I.T.H")

# -----------------------------------------------------------------------------
# CREDENTIALS FROM ENVIRONMENT
# -----------------------------------------------------------------------------
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin").strip()
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "").strip()

# Cookie name used for the session token.
SESSION_COOKIE = "session_token"

# Session TTL imported from central config.
from config import SESSION_TTL

# -----------------------------------------------------------------------------
# IN-MEMORY SESSION STORE
# -----------------------------------------------------------------------------
# Dict of valid session tokens → metadata.  Cleared on restart (intentional).
# Structure: { token_str: {"username": str, "expires_at": float} }
_active_sessions: dict = {}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def create_session(username: str) -> str:
    """Generate a cryptographically secure session token, store it with TTL."""
    token = secrets.token_hex(32)
    _active_sessions[token] = {
        "username": username,
        "expires_at": time.time() + SESSION_TTL,
    }
    logger.info(
        "[AUTH] Session created for '%s' (active: %d, TTL: %ds)",
        username, len(_active_sessions), SESSION_TTL,
    )
    return token


def delete_session(token: str) -> None:
    """Remove a session token, effectively logging out."""
    _active_sessions.pop(token, None)
    logger.info("[AUTH] Session deleted (active: %d)", len(_active_sessions))


def _cleanup_expired_sessions() -> None:
    """Purge all sessions whose expires_at is in the past. Called on every auth check."""
    now = time.time()
    expired = [t for t, s in _active_sessions.items() if s["expires_at"] <= now]
    for t in expired:
        del _active_sessions[t]
    if expired:
        logger.info("[AUTH] Purged %d expired session(s) (remaining: %d)", len(expired), len(_active_sessions))


def is_valid_session(token: str) -> bool:
    """Check if a session token exists and has not expired."""
    session = _active_sessions.get(token)
    if not session:
        return False
    if session["expires_at"] <= time.time():
        # Expired — remove immediately.
        del _active_sessions[token]
        return False
    return True


async def login_required(request: Request) -> None:
    """
    FastAPI dependency that enforces authentication.

    On every call:
      1. Purges all globally expired sessions (memory safety).
      2. Reads the session_token cookie.
      3. If missing → 401.
      4. If expired or invalid → 401.

    Use with Depends():
        @app.get("/route", dependencies=[Depends(login_required)])
    """
    _cleanup_expired_sessions()

    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not is_valid_session(token):
        raise HTTPException(status_code=401, detail="Session expired or invalid")
