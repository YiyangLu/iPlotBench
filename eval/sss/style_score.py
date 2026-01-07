"""
S_Style: Visual style similarity score.

Compares colors, mode, symbols, dash patterns, and sizes.
"""

from typing import List, Tuple, Optional, Dict, Any

from .utils.color import color_similarity, color_emd
from .utils.normalize import normalize_symbol, normalize_dash, get_with_default, PLOTLY_DEFAULTS


def extract_colors(trace: dict) -> List[str]:
    """Extract all colors from a trace."""
    colors = []

    # Marker colors
    marker = trace.get("marker", {})
    if isinstance(marker, dict):
        mc = marker.get("color")
        if mc:
            if isinstance(mc, list):
                colors.extend(str(c) for c in mc if c)
            else:
                colors.append(str(mc))

        # Pie chart uses 'colors' instead of 'color'
        mcs = marker.get("colors")
        if mcs and isinstance(mcs, list):
            colors.extend(str(c) for c in mcs if c)

    # Line colors
    line = trace.get("line", {})
    if isinstance(line, dict):
        lc = line.get("color")
        if lc:
            colors.append(str(lc))

    return colors


def compute_color_score(gt_trace: dict, pred_trace: dict) -> float:
    """
    Compute color similarity between traces.

    Single color: CIELAB Delta E
    Color array: EMD-based comparison

    Returns:
        Similarity in [0, 1]
    """
    gt_colors = extract_colors(gt_trace)
    pred_colors = extract_colors(pred_trace)

    if not gt_colors and not pred_colors:
        return 1.0
    if not gt_colors or not pred_colors:
        return 0.0

    if len(gt_colors) == 1 and len(pred_colors) == 1:
        # Single color comparison
        return color_similarity(gt_colors[0], pred_colors[0])
    else:
        # Array comparison using EMD
        return color_emd(gt_colors, pred_colors)


def compute_mode_score(gt_trace: dict, pred_trace: dict) -> float:
    """
    Compute mode matching score.

    Exact match for scatter mode (lines, markers, lines+markers).

    Returns:
        1.0 if match, 0.0 otherwise
    """
    gt_mode = gt_trace.get("mode", "")
    pred_mode = pred_trace.get("mode", "")

    # For non-scatter traces, mode is not applicable
    gt_type = gt_trace.get("type", "scatter")
    pred_type = pred_trace.get("type", "scatter")

    if gt_type not in ("scatter",) or pred_type not in ("scatter",):
        return 1.0  # Not applicable

    # Normalize modes (sort components for comparison)
    def normalize_mode(m: str) -> str:
        if not m:
            return ""
        parts = sorted(m.replace("+", " ").split())
        return "+".join(parts)

    return 1.0 if normalize_mode(gt_mode) == normalize_mode(pred_mode) else 0.0


def compute_symbol_score(gt_trace: dict, pred_trace: dict) -> float:
    """
    Compute marker symbol matching score.

    Uses alias normalization for flexible matching.

    Returns:
        1.0 if match, 0.0 otherwise
    """
    gt_marker = gt_trace.get("marker", {})
    pred_marker = pred_trace.get("marker", {})

    if not isinstance(gt_marker, dict) or not isinstance(pred_marker, dict):
        return 1.0  # No markers

    gt_symbol = gt_marker.get("symbol", PLOTLY_DEFAULTS.get("marker.symbol"))
    pred_symbol = pred_marker.get("symbol", PLOTLY_DEFAULTS.get("marker.symbol"))

    gt_norm = normalize_symbol(gt_symbol)
    pred_norm = normalize_symbol(pred_symbol)

    return 1.0 if gt_norm == pred_norm else 0.0


def compute_dash_score(gt_trace: dict, pred_trace: dict) -> float:
    """
    Compute line dash pattern matching score.

    Uses alias normalization for flexible matching.

    Returns:
        1.0 if match, 0.0 otherwise
    """
    gt_line = gt_trace.get("line", {})
    pred_line = pred_trace.get("line", {})

    if not isinstance(gt_line, dict) or not isinstance(pred_line, dict):
        return 1.0  # No line

    gt_dash = gt_line.get("dash", PLOTLY_DEFAULTS.get("line.dash"))
    pred_dash = pred_line.get("dash", PLOTLY_DEFAULTS.get("line.dash"))

    gt_norm = normalize_dash(gt_dash)
    pred_norm = normalize_dash(pred_dash)

    return 1.0 if gt_norm == pred_norm else 0.0


def compute_size_score(gt_trace: dict, pred_trace: dict) -> float:
    """
    Compute marker size and line width similarity.

    Uses normalized error.

    Returns:
        Similarity in [0, 1]
    """
    scores = []

    # Marker size
    gt_marker = gt_trace.get("marker", {})
    pred_marker = pred_trace.get("marker", {})

    if isinstance(gt_marker, dict) and isinstance(pred_marker, dict):
        gt_size = gt_marker.get("size", PLOTLY_DEFAULTS.get("marker.size"))
        pred_size = pred_marker.get("size", PLOTLY_DEFAULTS.get("marker.size"))

        try:
            gt_size = float(gt_size) if gt_size is not None else 6.0
            pred_size = float(pred_size) if pred_size is not None else 6.0
            # Normalize by typical range (0-20)
            size_error = abs(gt_size - pred_size) / 20.0
            scores.append(max(0.0, 1.0 - size_error))
        except (ValueError, TypeError):
            scores.append(0.5)

    # Line width
    gt_line = gt_trace.get("line", {})
    pred_line = pred_trace.get("line", {})

    if isinstance(gt_line, dict) and isinstance(pred_line, dict):
        gt_width = gt_line.get("width", PLOTLY_DEFAULTS.get("line.width"))
        pred_width = pred_line.get("width", PLOTLY_DEFAULTS.get("line.width"))

        try:
            gt_width = float(gt_width) if gt_width is not None else 2.0
            pred_width = float(pred_width) if pred_width is not None else 2.0
            # Normalize by typical range (0-10)
            width_error = abs(gt_width - pred_width) / 10.0
            scores.append(max(0.0, 1.0 - width_error))
        except (ValueError, TypeError):
            scores.append(0.5)

    if not scores:
        return 1.0

    return sum(scores) / len(scores)


def compute_trace_style_score(
    gt_trace: dict,
    pred_trace: dict,
    property_weights: Dict[str, float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute style score for a single trace pair.

    Args:
        gt_trace: Ground truth trace
        pred_trace: Predicted trace
        property_weights: Weights for each property

    Returns:
        (score, details) where score is in [0, 1]
    """
    if property_weights is None:
        property_weights = {
            "color": 0.40,
            "mode": 0.20,
            "symbol": 0.15,
            "dash": 0.10,
            "size": 0.15,
        }

    # Compute individual scores
    color_score = compute_color_score(gt_trace, pred_trace)
    mode_score = compute_mode_score(gt_trace, pred_trace)
    symbol_score = compute_symbol_score(gt_trace, pred_trace)
    dash_score = compute_dash_score(gt_trace, pred_trace)
    size_score = compute_size_score(gt_trace, pred_trace)

    # Weighted combination
    total_score = (
        property_weights["color"] * color_score +
        property_weights["mode"] * mode_score +
        property_weights["symbol"] * symbol_score +
        property_weights["dash"] * dash_score +
        property_weights["size"] * size_score
    )

    details = {
        "color": color_score,
        "mode": mode_score,
        "symbol": symbol_score,
        "dash": dash_score,
        "size": size_score,
    }

    return total_score, details


def compute_style_score(
    gt_traces: List[dict],
    pred_traces: List[dict],
    matches: List[Tuple[Optional[int], Optional[int]]],
    property_weights: Dict[str, float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute S_Style across all matched trace pairs.

    Formula: S_Style = (1/max(|T|,|P|)) * sum(style_similarity)

    Args:
        gt_traces: Ground truth traces
        pred_traces: Predicted traces
        matches: List of (gt_idx, pred_idx) pairs
        property_weights: Weights for each style property

    Returns:
        (score, details) where score is in [0, 1]
    """
    n_gt = len(gt_traces)
    n_pred = len(pred_traces)
    n = max(n_gt, n_pred)

    if n == 0:
        return 1.0, {"total": 0, "traces": []}

    total_score = 0.0
    trace_details = []

    for gt_idx, pred_idx in matches:
        if gt_idx is None or pred_idx is None:
            # Unmatched trace contributes 0
            trace_details.append({
                "gt_idx": gt_idx,
                "pred_idx": pred_idx,
                "score": 0.0,
                "color": 0.0,
                "mode": 0.0,
                "symbol": 0.0,
                "dash": 0.0,
                "size": 0.0,
            })
        else:
            score, details = compute_trace_style_score(
                gt_traces[gt_idx],
                pred_traces[pred_idx],
                property_weights,
            )
            total_score += score
            trace_details.append({
                "gt_idx": gt_idx,
                "pred_idx": pred_idx,
                "score": score,
                **details,
            })

    final_score = total_score / n

    return final_score, {
        "total": n,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "traces": trace_details,
    }
