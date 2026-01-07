"""Session storage with SQLite backend.

Implements session-per-folder architecture for storing plots, interactions,
and screenshots.
"""

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def _safe_folder_name(path: str) -> str:
    """Convert path to safe folder name."""
    return re.sub(r"[^a-zA-Z0-9_-]", "-", path.strip("/"))


class SessionStore:
    """SQLite-based session storage with session-per-folder architecture.

    Directory structure:
        logs/
            session_index.db          # Global registry
            sessions/
                {safe_cwd}/
                    {session_id}/
                        session.db    # Per-session data
                        plots/
                            1.json
                            2.json
                        screenshots/
                            1/
                                1.png
                                2.png
    """

    def __init__(self, logs_root: Path):
        self.logs_root = logs_root
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self._index_db = self.logs_root / "session_index.db"
        self._init_index_db()

    def _init_index_db(self) -> None:
        """Initialize global session index."""
        with sqlite3.connect(self._index_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    folder_path TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    cwd TEXT,
                    session_name TEXT
                )
            """)
            conn.commit()

    def _get_session_dir(self, session_id: str) -> Path | None:
        """Get session directory path (alias for _get_session_folder)."""
        return self._get_session_folder(session_id)

    def _get_session_folder(self, session_id: str) -> Path | None:
        """Get session folder path from index."""
        with sqlite3.connect(self._index_db) as conn:
            row = conn.execute(
                "SELECT folder_path FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row:
                return Path(row[0])
        return None

    def _get_session_db(self, session_id: str) -> Path:
        """Get session database path."""
        session_dir = self._get_session_dir(session_id)
        if session_dir:
            return session_dir / "session.db"
        return None

    def _init_session_db(self, db_path: Path) -> None:
        """Initialize per-session database."""
        with sqlite3.connect(db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS plots (
                    plot_id INTEGER PRIMARY KEY,
                    plotly_code TEXT,
                    description TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plot_id INTEGER,
                    timestamp TEXT,
                    event_type TEXT,
                    payload TEXT,
                    screenshot_path TEXT,
                    screenshot_size_kb INTEGER,
                    cumulative_state TEXT,
                    FOREIGN KEY (plot_id) REFERENCES plots(plot_id)
                );
            """)
            conn.commit()

    def create_session(
        self,
        session_id: str,
        cwd: Path,
        session_name: str | None = None,
    ) -> Path:
        """Create new session and return session directory."""
        safe_name = _safe_folder_name(str(cwd))
        session_dir = self.logs_root / "sessions" / safe_name / session_id

        # Create directories
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "plots").mkdir(exist_ok=True)
        (session_dir / "screenshots").mkdir(exist_ok=True)

        # Initialize session database
        db_path = session_dir / "session.db"
        self._init_session_db(db_path)

        # Register in index
        now = datetime.now().isoformat()
        with sqlite3.connect(self._index_db) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                (session_id, folder_path, created_at, updated_at, cwd, session_name)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, str(session_dir), now, now, str(cwd), session_name),
            )
            conn.commit()

        return session_dir

    def log_plot(
        self,
        session_id: str,
        plot_id: int,
        fig_json: dict,
        plotly_code: str | None = None,
        description: str | None = None,
    ) -> None:
        """Log a plot to the session."""
        session_dir = self._get_session_dir(session_id)
        if not session_dir:
            raise ValueError(f"Session not found: {session_id}")

        # Save figure JSON to file
        plots_dir = session_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        with open(plots_dir / f"{plot_id}.json", "w") as f:
            json.dump(fig_json, f)

        # Log to database
        db_path = session_dir / "session.db"
        now = datetime.now().isoformat()
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO plots (plot_id, plotly_code, description, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (plot_id, plotly_code, description, now),
            )
            conn.commit()

    def get_plot_json(self, session_id: str, plot_id: int) -> dict | None:
        """Get plot JSON from session."""
        session_dir = self._get_session_dir(session_id)
        if not session_dir:
            return None

        plot_path = session_dir / "plots" / f"{plot_id}.json"
        if plot_path.exists():
            with open(plot_path) as f:
                return json.load(f)
        return None

    def save_plot_json(self, session_id: str, plot_id: int, fig_json: dict) -> None:
        """Save updated plot JSON."""
        session_dir = self._get_session_dir(session_id)
        if not session_dir:
            raise ValueError(f"Session not found: {session_id}")

        plot_path = session_dir / "plots" / f"{plot_id}.json"
        with open(plot_path, "w") as f:
            json.dump(fig_json, f)

    def get_plots(self, session_id: str) -> list[dict]:
        """Get all plots for session."""
        db_path = self._get_session_db(session_id)
        if not db_path or not db_path.exists():
            return []

        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT plot_id, plotly_code, description, created_at FROM plots"
            ).fetchall()
            return [dict(row) for row in rows]

    def log_interaction(
        self,
        session_id: str,
        plot_id: int,
        event_type: str,
        payload: dict,
        screenshot_path: str | None = None,
        screenshot_size_kb: int | None = None,
    ) -> int:
        """Log an interaction and compute cumulative state."""
        db_path = self._get_session_db(session_id)
        if not db_path:
            raise ValueError(f"Session not found: {session_id}")

        # Get previous cumulative state
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT cumulative_state FROM interactions
                WHERE plot_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (plot_id,),
            ).fetchone()
            prev_state = json.loads(row[0]) if row and row[0] else None

        # Compute new cumulative state
        cumulative_state = self._compute_cumulative_state(
            prev_state, event_type, payload
        )

        # Insert interaction
        now = datetime.now().isoformat()
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO interactions
                (plot_id, timestamp, event_type, payload, screenshot_path,
                 screenshot_size_kb, cumulative_state)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plot_id,
                    now,
                    event_type,
                    json.dumps(payload),
                    screenshot_path,
                    screenshot_size_kb,
                    json.dumps(cumulative_state),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def update_interaction_screenshot(
        self,
        session_id: str,
        interaction_id: int,
        screenshot_path: str,
        screenshot_size_kb: int,
    ) -> None:
        """Update interaction with screenshot info."""
        db_path = self._get_session_db(session_id)
        if not db_path:
            raise ValueError(f"Session not found: {session_id}")

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                UPDATE interactions
                SET screenshot_path = ?, screenshot_size_kb = ?
                WHERE id = ?
                """,
                (screenshot_path, screenshot_size_kb, interaction_id),
            )
            conn.commit()

    def get_interactions(
        self,
        session_id: str,
        plot_id: int | None = None,
        event_type: str | None = None,
    ) -> list[dict]:
        """Get interactions for session/plot."""
        db_path = self._get_session_db(session_id)
        if not db_path or not db_path.exists():
            return []

        query = "SELECT * FROM interactions WHERE 1=1"
        params = []

        if plot_id is not None:
            query += " AND plot_id = ?"
            params.append(plot_id)
        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type)

        query += " ORDER BY id ASC"

        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                if d["payload"]:
                    d["payload"] = json.loads(d["payload"])
                if d["cumulative_state"]:
                    d["cumulative_state"] = json.loads(d["cumulative_state"])
                result.append(d)
            return result

    def get_plot_cumulative_state(
        self, session_id: str, plot_id: int
    ) -> dict | None:
        """Get latest cumulative state for plot."""
        db_path = self._get_session_db(session_id)
        if not db_path or not db_path.exists():
            return None

        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT cumulative_state FROM interactions
                WHERE plot_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (plot_id,),
            ).fetchone()
            if row and row[0]:
                return json.loads(row[0])
        return None

    def get_screenshot_dir(self, session_id: str) -> Path:
        """Get screenshot directory for session (matches original: screenshots/{interaction_id}.png)."""
        session_dir = self._get_session_dir(session_id)
        if not session_dir:
            raise ValueError(f"Session not found: {session_id}")

        screenshot_dir = session_dir / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        return screenshot_dir

    def get_session_info(self, session_id: str) -> "SessionInfo | None":
        """Get session info including folder path."""
        with sqlite3.connect(self._index_db) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row:
                return SessionInfo(
                    session_id=row["session_id"],
                    folder_path=Path(row["folder_path"]),
                    cwd=row["cwd"],
                    session_name=row["session_name"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
        return None


    def _compute_cumulative_state(
        self,
        prev_state: dict | None,
        event_type: str,
        payload: dict,
    ) -> dict:
        """Compute cumulative state from previous state and event."""
        state = (prev_state or {}).copy()

        if event_type == "init":
            # Reset to initial state
            state = {"trace_visibility": {}}

        elif event_type == "relayout":
            # Update axis ranges
            for key in ["xaxis.range[0]", "xaxis.range[1]",
                        "yaxis.range[0]", "yaxis.range[1]"]:
                if key in payload:
                    state[key] = payload[key]
            # Handle autorange
            if payload.get("xaxis.autorange"):
                state.pop("xaxis.range[0]", None)
                state.pop("xaxis.range[1]", None)
            if payload.get("yaxis.autorange"):
                state.pop("yaxis.range[0]", None)
                state.pop("yaxis.range[1]", None)

        elif event_type == "legendclick":
            curve_number = payload.get("curve_number")
            visible = payload.get("visible", True)
            if "trace_visibility" not in state:
                state["trace_visibility"] = {}
            state["trace_visibility"][str(curve_number)] = visible

        elif event_type == "selected":
            state["selection_x_range"] = payload.get("x_range")
            state["selection_y_range"] = payload.get("y_range")
            state["selected_points"] = payload.get("point_indices")

        elif event_type == "doubleclick":
            # Reset to autoscale
            state.pop("xaxis.range[0]", None)
            state.pop("xaxis.range[1]", None)
            state.pop("yaxis.range[0]", None)
            state.pop("yaxis.range[1]", None)

        return state


class SessionInfo:
    """Session information from index."""

    def __init__(
        self,
        session_id: str,
        folder_path: Path,
        cwd: str,
        session_name: str | None,
        created_at: str,
        updated_at: str,
    ):
        self.session_id = session_id
        self.folder_path = folder_path
        self.cwd = cwd
        self.session_name = session_name
        self.created_at = created_at
        self.updated_at = updated_at


# Singleton storage
_session_stores: dict[str, SessionStore] = {}


def get_session_store(logs_root: Path | None = None) -> SessionStore:
    """Get or create session store singleton."""
    if logs_root is None:
        logs_root = Path("localenv/logs")

    key = str(logs_root.absolute())
    if key not in _session_stores:
        _session_stores[key] = SessionStore(logs_root)
    return _session_stores[key]
