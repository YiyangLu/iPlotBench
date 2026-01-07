"""Session context management for MCP tools."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session_store import SessionStore

# Thread-local storage for session context (properly thread-safe)
_thread_local = threading.local()


def register_session(session_store: "SessionStore", session_id: str) -> None:
    """Set current session for this thread."""
    _thread_local.current_session = (session_store, session_id)
    # Keep env vars for backward compatibility (single-threaded use)
    os.environ["_AGENT_SESSION_ID"] = session_id
    os.environ["_AGENT_LOGS_ROOT"] = str(session_store.logs_root)


def get_current_session() -> tuple["SessionStore | None", str | None]:
    """Get current session from thread-local storage or env vars."""
    # Try thread-local first
    current = getattr(_thread_local, 'current_session', None)
    if current is not None:
        return current

    # Fallback to environment variables (for single-threaded use)
    session_id = os.environ.get("_AGENT_SESSION_ID")
    logs_root = os.environ.get("_AGENT_LOGS_ROOT")
    if session_id and logs_root:
        from .session_store import get_session_store

        store = get_session_store(Path(logs_root))
        return store, session_id

    return None, None


def unregister_session() -> None:
    """Clear current session for this thread."""
    _thread_local.current_session = None
    os.environ.pop("_AGENT_SESSION_ID", None)
    os.environ.pop("_AGENT_LOGS_ROOT", None)


@contextmanager
def session_context(session_store: "SessionStore", session_id: str):
    """Context manager for session scope."""
    prev = getattr(_thread_local, 'current_session', None)
    try:
        register_session(session_store, session_id)
        yield
    finally:
        _thread_local.current_session = prev
        if prev is None:
            unregister_session()
