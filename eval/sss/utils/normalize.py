"""
Normalization utilities for SSS metrics.

Provides Plotly default values and alias normalization.
"""

from typing import Any, Dict

# Plotly default values for style attributes
PLOTLY_DEFAULTS: Dict[str, Any] = {
    # Marker defaults
    "marker.size": 6,
    "marker.symbol": "circle",
    "marker.opacity": 1.0,
    # Line defaults
    "line.width": 2,
    "line.dash": "solid",
    # Mode defaults
    "mode": "lines",  # For scatter traces
}

# Symbol alias map (all aliases -> canonical name)
SYMBOL_ALIASES: Dict[str, str] = {
    # Circle
    "circle": "circle",
    "circle-open": "circle",
    "o": "circle",
    "0": "circle",
    # Square
    "square": "square",
    "square-open": "square",
    "s": "square",
    "1": "square",
    # Diamond
    "diamond": "diamond",
    "diamond-open": "diamond",
    "d": "diamond",
    "2": "diamond",
    # Cross
    "cross": "cross",
    "cross-open": "cross",
    "x": "cross",
    "+": "cross",
    "3": "cross",
    # Triangle-up
    "triangle-up": "triangle-up",
    "triangle-up-open": "triangle-up",
    "^": "triangle-up",
    "4": "triangle-up",
    # Triangle-down
    "triangle-down": "triangle-down",
    "triangle-down-open": "triangle-down",
    "v": "triangle-down",
    "5": "triangle-down",
    # Star
    "star": "star",
    "star-open": "star",
    "*": "star",
}

# Dash alias map (all aliases -> canonical name)
DASH_ALIASES: Dict[str, str] = {
    # Solid
    "solid": "solid",
    "": "solid",
    # Dash
    "dash": "dash",
    "--": "dash",
    "dashed": "dash",
    # Dot
    "dot": "dot",
    ":": "dot",
    "dotted": "dot",
    # Dash-dot
    "dashdot": "dashdot",
    "-.": "dashdot",
    # Long dash
    "longdash": "longdash",
    # Long dash dot
    "longdashdot": "longdashdot",
}


def normalize_symbol(symbol: Any) -> str:
    """
    Normalize marker symbol to canonical name.

    Args:
        symbol: Symbol value (string, int, or other)

    Returns:
        Canonical symbol name
    """
    if symbol is None:
        return "circle"

    symbol_str = str(symbol).lower().strip()
    return SYMBOL_ALIASES.get(symbol_str, symbol_str)


def normalize_dash(dash: Any) -> str:
    """
    Normalize line dash pattern to canonical name.

    Args:
        dash: Dash value (string or other)

    Returns:
        Canonical dash name
    """
    if dash is None:
        return "solid"

    dash_str = str(dash).lower().strip()
    return DASH_ALIASES.get(dash_str, dash_str)


def get_with_default(obj: dict, path: str, default: Any = None) -> Any:
    """
    Get nested value from dict with dot notation, using Plotly defaults.

    Args:
        obj: Dictionary to query
        path: Dot-separated path (e.g., "marker.size")
        default: Override default (if None, uses PLOTLY_DEFAULTS)

    Returns:
        Value at path, or default
    """
    keys = path.split(".")
    current = obj

    for key in keys:
        if not isinstance(current, dict):
            break
        current = current.get(key)
        if current is None:
            break

    if current is not None:
        return current

    # Use explicit default if provided
    if default is not None:
        return default

    # Otherwise use Plotly defaults
    return PLOTLY_DEFAULTS.get(path)
