"""Runner modules for iPlotBench evaluation."""

from .local_llm import (
    LocalLLMRunner,
    RunnerConfig,
    TaskResult,
    run_figure,
    AGENT_CONFIGS,
)
from .mcp import (
    PLOTLY_TOOLS,
    execute_tool,
    SessionStore,
    get_session_store,
    PlotStore,
    get_plot_store,
    register_session,
    get_current_session,
    unregister_session,
    session_context,
)

__all__ = [
    # Runner
    "LocalLLMRunner",
    "RunnerConfig",
    "TaskResult",
    "run_figure",
    "AGENT_CONFIGS",
    # MCP Tools
    "PLOTLY_TOOLS",
    "execute_tool",
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
