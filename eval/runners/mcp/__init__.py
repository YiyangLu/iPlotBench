"""MCP Plotly tools for local LLM evaluation.

Replicates the exact same MCP tools from the Claude agent for local LLM testing.
"""

from .tools import (
    PLOTLY_TOOLS,
    execute_tool,
    show_plot,
    query_interactions,
    get_plot_json,
    get_plot_image,
    relayout,
    legendclick,
    selected,
)
from .session_store import SessionStore, get_session_store
from .plot_store import PlotStore, get_plot_store
from .context import (
    register_session,
    get_current_session,
    unregister_session,
    session_context,
)

__all__ = [
    # Tools
    "PLOTLY_TOOLS",
    "execute_tool",
    "show_plot",
    "query_interactions",
    "get_plot_json",
    "get_plot_image",
    "relayout",
    "legendclick",
    "selected",
    # Storage
    "SessionStore",
    "get_session_store",
    "PlotStore",
    "get_plot_store",
    # Context
    "register_session",
    "get_current_session",
    "unregister_session",
    "session_context",
]
