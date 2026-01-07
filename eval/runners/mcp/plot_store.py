"""In-memory plot storage with disk fallback.

Provides session-scoped plot IDs and caching.
"""

from typing import Any

from .context import get_current_session


class PlotStore:
    """In-memory plot storage with session-scoped IDs.

    Each session has its own plot counter starting at 1.
    Syncs with disk on first plot of session.
    """

    def __init__(self):
        # (session_id, plot_id) -> figure dict
        self._plots: dict[tuple[str, int], dict] = {}
        # session_id -> next plot_id
        self._session_counters: dict[str, int] = {}

    def add_plot(self, session_id: str, fig_json: dict) -> tuple[int, dict]:
        """Add plot and return (plot_id, figure).

        Args:
            session_id: Session identifier
            fig_json: Plotly figure as dict

        Returns:
            (plot_id, figure_dict)
        """
        # Sync counter with disk if first plot in session
        if session_id not in self._session_counters:
            store, _ = get_current_session()
            if store:
                existing_plots = store.get_plots(session_id)
                if existing_plots:
                    max_id = max(p["plot_id"] for p in existing_plots)
                    self._session_counters[session_id] = max_id + 1
                else:
                    self._session_counters[session_id] = 1
            else:
                self._session_counters[session_id] = 1

        # Get next plot ID
        plot_id = self._session_counters[session_id]
        self._session_counters[session_id] += 1

        # Store in memory
        self._plots[(session_id, plot_id)] = fig_json

        return plot_id, fig_json

    def get_plot(self, session_id: str, plot_id: int) -> dict | None:
        """Get plot from memory or disk.

        Args:
            session_id: Session identifier
            plot_id: Plot ID

        Returns:
            Figure dict or None
        """
        key = (session_id, plot_id)

        # Check memory first
        if key in self._plots:
            return self._plots[key]

        # Fall back to disk
        store, _ = get_current_session()
        if store:
            fig_json = store.get_plot_json(session_id, plot_id)
            if fig_json:
                # Cache to memory
                self._plots[key] = fig_json
                return fig_json

        return None

    def update_plot(self, session_id: str, plot_id: int, fig_json: dict) -> None:
        """Update plot in memory and disk.

        Args:
            session_id: Session identifier
            plot_id: Plot ID
            fig_json: Updated figure dict
        """
        key = (session_id, plot_id)
        self._plots[key] = fig_json

        # Also save to disk
        store, _ = get_current_session()
        if store:
            store.save_plot_json(session_id, plot_id, fig_json)

    def exists(self, session_id: str, plot_id: int) -> bool:
        """Check if plot exists."""
        return self.get_plot(session_id, plot_id) is not None

    def clear_session(self, session_id: str) -> None:
        """Clear all plots for a session from memory."""
        keys_to_remove = [k for k in self._plots if k[0] == session_id]
        for k in keys_to_remove:
            del self._plots[k]
        self._session_counters.pop(session_id, None)

    def clear_all(self) -> None:
        """Clear all plots from memory."""
        self._plots.clear()
        self._session_counters.clear()


# Global singleton
_plot_store: PlotStore | None = None


def get_plot_store() -> PlotStore:
    """Get global plot store instance."""
    global _plot_store
    if _plot_store is None:
        _plot_store = PlotStore()
    return _plot_store
