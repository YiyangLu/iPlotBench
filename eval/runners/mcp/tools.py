"""MCP Plotly tools implementation.

Implements all 7 MCP tools for local LLM evaluation:
1. show_plot - Create visualizations
2. query_interactions - Get interaction history
3. get_plot_json - Retrieve plot data
4. get_plot_image - Get screenshots
5. relayout - Zoom/Pan
6. legendclick - Toggle trace visibility
7. selected - Select data points
"""

import json
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

from .context import get_current_session
from .event_bus import get_event_bus
from .headless_handler import get_headless_handler
from .plot_store import get_plot_store


class PlotlyToolError(Exception):
    """Error in Plotly tool execution."""
    pass


# =============================================================================
# OpenAI-compatible tool definitions (matching original MCP tools exactly)
# =============================================================================

PLOTLY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "show_plot",
            "description": "Create Plotly figure from code. Returns {plot_id}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plotly_codes": {
                        "type": "string",
                        "description": "Python code defining 'fig' (Plotly figure). Include imports."
                    }
                },
                "required": ["plotly_codes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_interactions",
            "description": "Get interaction history. Returns [{id, event_type, payload, has_screenshot}].",
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_id": {
                        "type": "integer",
                        "description": "Plot ID from show_plot."
                    },
                    "event_type": {
                        "type": "string",
                        "description": "Optional filter: init, relayout, legendclick, selected."
                    }
                },
                "required": ["plot_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_plot_json",
            "description": "Get Plotly figure data. Returns {data, layout}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_id": {
                        "type": "integer",
                        "description": "Plot ID from show_plot."
                    }
                },
                "required": ["plot_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_plot_image",
            "description": "Get screenshot of a plot. Returns {image_path}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_id": {
                        "type": "integer",
                        "description": "Plot ID from show_plot."
                    },
                    "interaction_id": {
                        "type": "integer",
                        "description": "Optional. Specific interaction ID for past state."
                    }
                },
                "required": ["plot_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "relayout",
            "description": "Zoom/pan plot by setting axis ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_id": {
                        "type": "integer",
                        "description": "Plot ID from show_plot."
                    },
                    "x_min": {
                        "type": "number",
                        "description": "X-axis range minimum."
                    },
                    "x_max": {
                        "type": "number",
                        "description": "X-axis range maximum."
                    },
                    "y_min": {
                        "type": "number",
                        "description": "Y-axis range minimum."
                    },
                    "y_max": {
                        "type": "number",
                        "description": "Y-axis range maximum."
                    }
                },
                "required": ["plot_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "legendclick",
            "description": "Toggle trace visibility.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_id": {
                        "type": "integer",
                        "description": "Plot ID from show_plot."
                    },
                    "curve_number": {
                        "type": "integer",
                        "description": "Trace index (0-based)."
                    }
                },
                "required": ["plot_id", "curve_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "selected",
            "description": "Select data points by region.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_id": {
                        "type": "integer",
                        "description": "Plot ID from show_plot."
                    },
                    "x_min": {
                        "type": "number",
                        "description": "X-axis selection minimum."
                    },
                    "x_max": {
                        "type": "number",
                        "description": "X-axis selection maximum."
                    },
                    "y_min": {
                        "type": "number",
                        "description": "Y-axis selection minimum."
                    },
                    "y_max": {
                        "type": "number",
                        "description": "Y-axis selection maximum."
                    }
                },
                "required": ["plot_id"]
            }
        }
    }
]


# =============================================================================
# Tool implementations
# =============================================================================

def show_plot(plotly_codes: str) -> dict:
    """Create a Plotly figure from code.

    Args:
        plotly_codes: Python code defining 'fig' variable

    Returns:
        {"plot_id": int, "success": True}
    """
    store, session_id = get_current_session()
    if not store or not session_id:
        raise PlotlyToolError("No active session. Call register_session first.")

    # Execute code in isolated namespace
    exec_namespace = {}

    # Monkey-patch show to no-op
    original_show = pio.show
    pio.show = lambda *_args, **_kwargs: None

    try:
        exec(plotly_codes, exec_namespace)
    except SyntaxError as e:
        raise PlotlyToolError(f"Syntax error in code: {e}")
    except Exception as e:
        raise PlotlyToolError(f"Error executing code: {e}")
    finally:
        pio.show = original_show

    # Extract fig variable
    if "fig" not in exec_namespace:
        raise PlotlyToolError("Code must define a 'fig' variable")

    fig = exec_namespace["fig"]
    if not isinstance(fig, go.Figure):
        raise PlotlyToolError(f"'fig' must be a Plotly Figure, got {type(fig)}")

    # Convert to JSON dict
    fig_json = json.loads(fig.to_json())

    # Add to plot store
    plot_store = get_plot_store()
    plot_id, _ = plot_store.add_plot(session_id, fig_json)

    # Log to session store
    store.log_plot(
        session_id=session_id,
        plot_id=plot_id,
        fig_json=fig_json,
        plotly_code=plotly_codes,
    )

    # Generate initial screenshot via headless handler
    handler = get_headless_handler()
    handler.handle_init(session_id, plot_id)

    # Emit event
    get_event_bus().publish("plot_show", {
        "session_id": session_id,
        "plot_id": plot_id,
    })

    return {"plot_id": plot_id, "success": True}


def query_interactions(plot_id: int, event_type: str | None = None) -> dict:
    """Get interaction history for a plot.

    Args:
        plot_id: Plot ID
        event_type: Optional filter

    Returns:
        {"events": [{"id", "event_type", "payload", "has_screenshot"}]}
    """
    store, session_id = get_current_session()
    if not store or not session_id:
        raise PlotlyToolError("No active session")

    interactions = store.get_interactions(session_id, plot_id, event_type)

    events = []
    for interaction in interactions:
        events.append({
            "id": interaction["id"],
            "event_type": interaction["event_type"],
            "payload": interaction["payload"],
            "has_screenshot": bool(interaction.get("screenshot_path")),
        })

    return {"events": events}


def get_plot_json(plot_id: int) -> dict:
    """Get Plotly figure data as JSON.

    Args:
        plot_id: Plot ID

    Returns:
        {"data": [...], "layout": {...}, "plot_id": int}
    """
    store, session_id = get_current_session()
    if not store or not session_id:
        raise PlotlyToolError("No active session")

    plot_store = get_plot_store()
    fig_json = plot_store.get_plot(session_id, plot_id)

    if not fig_json:
        raise PlotlyToolError(f"Plot {plot_id} not found")

    # Get cumulative state
    cumulative_state = store.get_plot_cumulative_state(session_id, plot_id)

    return {
        "data": fig_json.get("data", []),
        "layout": fig_json.get("layout", {}),
        "plot_id": plot_id,
        "session_id": session_id,
        "cumulative_state": cumulative_state,
    }


def get_plot_image(plot_id: int, interaction_id: int | None = None) -> dict:
    """Get screenshot of a plot.

    Args:
        plot_id: Plot ID
        interaction_id: Optional specific interaction

    Returns:
        {"image_path": str, "image_base64": str} - path and base64-encoded PNG
    """
    import base64

    store, session_id = get_current_session()
    if not store or not session_id:
        raise PlotlyToolError("No active session")

    # Get interactions with screenshots
    interactions = store.get_interactions(session_id, plot_id)
    screenshots = [i for i in interactions if i.get("screenshot_path")]

    if not screenshots:
        raise PlotlyToolError(f"No screenshots available for plot {plot_id}")

    # Find specific or latest
    if interaction_id is not None:
        matching = [i for i in screenshots if i["id"] == interaction_id]
        if not matching:
            raise PlotlyToolError(f"Interaction {interaction_id} not found or has no screenshot")
        interaction = matching[0]
    else:
        interaction = screenshots[-1]  # Latest

    # Build absolute path
    session_dir = store._get_session_dir(session_id)
    screenshot_path = session_dir / interaction["screenshot_path"]

    if not screenshot_path.exists():
        raise PlotlyToolError(f"Screenshot file not found: {screenshot_path}")

    # Read and encode image for VLMs that need embedded images
    with open(screenshot_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()

    return {
        "image_path": str(screenshot_path.absolute()),
        "image_base64": image_base64,
    }


def relayout(
    plot_id: int,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> dict:
    """Zoom/pan plot by setting axis ranges.

    Args:
        plot_id: Plot ID
        x_min, x_max: X-axis range
        y_min, y_max: Y-axis range

    Returns:
        {"success": True, "plot_id": int}
    """
    store, session_id = get_current_session()
    if not store or not session_id:
        raise PlotlyToolError("No active session")

    # Build relayout args
    args = {}
    if x_min is not None:
        args["xaxis.range[0]"] = x_min
    if x_max is not None:
        args["xaxis.range[1]"] = x_max
    if y_min is not None:
        args["yaxis.range[0]"] = y_min
    if y_max is not None:
        args["yaxis.range[1]"] = y_max

    if not args:
        raise PlotlyToolError("Must provide at least one range parameter")

    # Handle via headless handler
    handler = get_headless_handler()
    result = handler.handle_command(session_id, plot_id, "relayout", args)

    if not result.get("success"):
        raise PlotlyToolError(result.get("error", "Relayout failed"))

    # Emit event
    get_event_bus().publish("plot_command", {
        "session_id": session_id,
        "plot_id": plot_id,
        "command": "relayout",
        "args": args,
    })

    return {
        "success": True,
        "plot_id": plot_id,
        "x_range": [x_min, x_max] if x_min is not None or x_max is not None else None,
        "y_range": [y_min, y_max] if y_min is not None or y_max is not None else None,
    }


def legendclick(plot_id: int, curve_number: int) -> dict:
    """Toggle trace visibility.

    Args:
        plot_id: Plot ID
        curve_number: Zero-based trace index

    Returns:
        {"success": True, "plot_id": int, "curve_number": int}
    """
    store, session_id = get_current_session()
    if not store or not session_id:
        raise PlotlyToolError("No active session")

    # Handle via headless handler
    handler = get_headless_handler()
    result = handler.handle_command(
        session_id, plot_id, "legendclick", {"curve_number": curve_number}
    )

    if not result.get("success"):
        raise PlotlyToolError(result.get("error", "Legendclick failed"))

    # Emit event
    get_event_bus().publish("plot_command", {
        "session_id": session_id,
        "plot_id": plot_id,
        "command": "legendclick",
        "args": {"curve_number": curve_number},
    })

    return {
        "success": True,
        "plot_id": plot_id,
        "curve_number": curve_number,
    }


def selected(
    plot_id: int,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> dict:
    """Select data points by region.

    Args:
        plot_id: Plot ID
        x_min: X-axis selection minimum
        x_max: X-axis selection maximum
        y_min: Y-axis selection minimum
        y_max: Y-axis selection maximum

    Returns:
        {"success": True, "plot_id": int, "selection": {...}}
    """
    store, session_id = get_current_session()
    if not store or not session_id:
        raise PlotlyToolError("No active session")

    # Build ranges from min/max params (matching original MCP wrapper)
    args = {}
    x_range = None
    y_range = None
    if x_min is not None and x_max is not None:
        x_range = [x_min, x_max]
        args["x_range"] = x_range
    if y_min is not None and y_max is not None:
        y_range = [y_min, y_max]
        args["y_range"] = y_range

    if not args:
        raise PlotlyToolError("Must provide x_min/x_max or y_min/y_max")

    # Handle via headless handler
    handler = get_headless_handler()
    result = handler.handle_command(session_id, plot_id, "selected", args)

    if not result.get("success"):
        raise PlotlyToolError(result.get("error", "Selection failed"))

    # Emit event
    get_event_bus().publish("plot_command", {
        "session_id": session_id,
        "plot_id": plot_id,
        "command": "selected",
        "args": args,
    })

    return {
        "success": True,
        "plot_id": plot_id,
        "selection": {
            "x_range": x_range,
            "y_range": y_range,
        },
    }


# =============================================================================
# Tool executor
# =============================================================================

def execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool and return the result.

    Args:
        tool_name: Name of the tool
        arguments: Tool arguments

    Returns:
        Tool result or error dict
    """
    try:
        if tool_name == "show_plot":
            return show_plot(arguments.get("plotly_codes", ""))
        elif tool_name == "query_interactions":
            return query_interactions(
                arguments.get("plot_id"),
                arguments.get("event_type"),
            )
        elif tool_name == "get_plot_json":
            return get_plot_json(arguments.get("plot_id"))
        elif tool_name == "get_plot_image":
            return get_plot_image(
                arguments.get("plot_id"),
                arguments.get("interaction_id"),
            )
        elif tool_name == "relayout":
            return relayout(
                arguments.get("plot_id"),
                arguments.get("x_min"),
                arguments.get("x_max"),
                arguments.get("y_min"),
                arguments.get("y_max"),
            )
        elif tool_name == "legendclick":
            return legendclick(
                arguments.get("plot_id"),
                arguments.get("curve_number"),
            )
        elif tool_name == "selected":
            return selected(
                arguments.get("plot_id"),
                arguments.get("x_min"),
                arguments.get("x_max"),
                arguments.get("y_min"),
                arguments.get("y_max"),
            )
        else:
            return {"error": f"Unknown tool: {tool_name}", "success": False}

    except PlotlyToolError as e:
        return {"error": str(e), "success": False, "error_type": "PlotlyToolError"}
    except Exception as e:
        return {"error": str(e), "success": False, "error_type": type(e).__name__}
