"""
S_Data: Data accuracy score.

Uses Chamfer Distance with hybrid distance function for
numerical and categorical dimensions.

For charts without axes (e.g., pie charts), values are normalized
to proportions since absolute values cannot be visually determined.
"""

import base64
import struct
from typing import List, Tuple, Optional, Dict, Any
import math

import numpy as np

from .trace_matching import infer_trace_type


def _decode_bdata(obj: Any) -> Any:
    """Decode Plotly's binary-encoded data format (bdata) to Python lists.

    Some model outputs (especially from tool-using configs) may contain
    binary-encoded arrays like: {"dtype": "f8", "bdata": "AAAA..."}
    This function converts them to plain Python lists.
    """
    if isinstance(obj, dict) and 'bdata' in obj and 'dtype' in obj:
        dtype_map = {'f8': 'd', 'f4': 'f', 'i4': 'i', 'i8': 'q', 'u1': 'B'}
        dtype = obj.get('dtype', 'f8')
        try:
            bdata = base64.b64decode(obj['bdata'])
            fmt = dtype_map.get(dtype, 'd')
            count = len(bdata) // struct.calcsize(fmt)
            return list(struct.unpack(f'{count}{fmt}', bdata))
        except Exception:
            return []  # Return empty list if decoding fails
    return obj


def has_axes(layout: dict) -> bool:
    """
    Check if layout has axes (xaxis or yaxis).

    Charts without axes (pie, donut, treemap) encode data as proportions,
    so absolute values cannot be determined visually.
    """
    if not layout:
        return False
    # Check for any axis definition
    return any(
        key.startswith(("xaxis", "yaxis"))
        for key in layout.keys()
    )


def normalize_to_proportions(values: List) -> List[float]:
    """
    Normalize numerical values to proportions (sum to 1).

    Used for charts without axes where only relative proportions
    can be visually determined.
    """
    try:
        float_values = [float(v) for v in values]
        total = sum(float_values)
        if total > 0:
            return [v / total for v in float_values]
    except (ValueError, TypeError):
        pass
    return values


def extract_data_points(
    trace: dict, chart_type: str, normalize_proportions: bool = False
) -> Tuple[List[Tuple], List[str]]:
    """
    Extract data points from trace based on chart type.

    Args:
        trace: Plotly trace dict
        chart_type: Type of chart (pie, hbar, vbar, line, etc.)
        normalize_proportions: If True, normalize numerical values to proportions.
                               Used for charts without axes (e.g., pie).

    Returns:
        (points, dim_types) where:
        - points: List of tuples (one per data point)
        - dim_types: List of 'numerical' or 'categorical' per dimension
    """
    if chart_type == "pie":
        labels = _decode_bdata(trace.get("labels", []))
        values = _decode_bdata(trace.get("values", []))
        if normalize_proportions:
            values = normalize_to_proportions(values)
        points = list(zip(labels, values))
        return points, ["categorical", "numerical"]

    elif chart_type == "hbar":
        # Horizontal bar: y is categorical, x is numerical
        x = _decode_bdata(trace.get("x", []))
        y = _decode_bdata(trace.get("y", []))
        if normalize_proportions:
            x = normalize_to_proportions(x)
        points = list(zip(y, x))  # (label, value)
        return points, ["categorical", "numerical"]

    elif chart_type == "vbar":
        # Vertical bar: x is categorical, y is numerical
        x = _decode_bdata(trace.get("x", []))
        y = _decode_bdata(trace.get("y", []))
        if normalize_proportions:
            y = normalize_to_proportions(y)
        points = list(zip(x, y))  # (label, value)
        return points, ["categorical", "numerical"]

    else:
        # Line, dot_line, scatter: both x and y are numerical
        x = _decode_bdata(trace.get("x", []))
        y = _decode_bdata(trace.get("y", []))
        # Note: For line charts, normalization doesn't apply
        # since both dimensions are positional
        points = list(zip(x, y))
        return points, ["numerical", "numerical"]


def compute_data_range(points: List[Tuple], dim_types: List[str]) -> Tuple[float, float]:
    """
    Compute min/max range of numerical values for normalization.
    """
    all_values = []

    for pt in points:
        for i, dtype in enumerate(dim_types):
            if dtype == "numerical" and i < len(pt):
                try:
                    all_values.append(float(pt[i]))
                except (ValueError, TypeError):
                    pass

    if not all_values:
        return (0.0, 1.0)

    return (min(all_values), max(all_values))


def hybrid_distance(
    pt1: Tuple,
    pt2: Tuple,
    dim_types: List[str],
    data_range: Tuple[float, float],
) -> float:
    """
    Compute hybrid distance between two points.

    - Numerical: |a_norm - b_norm| (range-normalized)
    - Categorical: 0 if equal, 1 if different

    Returns:
        Distance in [0, 1]
    """
    n_dims = len(dim_types)
    total_dist = 0.0

    range_size = data_range[1] - data_range[0]
    if range_size < 1e-10:
        range_size = 1.0

    for i in range(n_dims):
        if i >= len(pt1) or i >= len(pt2):
            total_dist += 1.0
            continue

        if dim_types[i] == "numerical":
            try:
                v1 = float(pt1[i])
                v2 = float(pt2[i])
                # Range-normalized distance
                v1_norm = (v1 - data_range[0]) / range_size
                v2_norm = (v2 - data_range[0]) / range_size
                total_dist += abs(v1_norm - v2_norm)
            except (ValueError, TypeError):
                total_dist += 1.0
        else:
            # Categorical: binary match
            s1 = str(pt1[i]).lower().strip()
            s2 = str(pt2[i]).lower().strip()
            total_dist += 0.0 if s1 == s2 else 1.0

    return total_dist / n_dims if n_dims > 0 else 0.0


def chamfer_distance(
    gt_points: List[Tuple],
    pred_points: List[Tuple],
    dim_types: List[str],
    data_range: Tuple[float, float],
) -> float:
    """
    Compute Chamfer distance between two point sets.

    Chamfer = (forward + backward) / 2
    where forward = avg min distance from GT to Pred
    and backward = avg min distance from Pred to GT

    Returns:
        Distance in [0, 1], where 0 = identical
    """
    if not gt_points and not pred_points:
        return 0.0
    if not gt_points or not pred_points:
        return 1.0

    # Forward: GT -> Pred
    forward_dist = 0.0
    for gt_pt in gt_points:
        min_dist = min(
            hybrid_distance(gt_pt, pred_pt, dim_types, data_range)
            for pred_pt in pred_points
        )
        forward_dist += min_dist
    forward_dist /= len(gt_points)

    # Backward: Pred -> GT
    backward_dist = 0.0
    for pred_pt in pred_points:
        min_dist = min(
            hybrid_distance(pred_pt, gt_pt, dim_types, data_range)
            for gt_pt in gt_points
        )
        backward_dist += min_dist
    backward_dist /= len(pred_points)

    return (forward_dist + backward_dist) / 2


def compute_trace_data_score(
    gt_trace: dict,
    pred_trace: dict,
    chamfer_scale: float = 5.0,
    normalize_proportions: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute data score for a single trace pair.

    Args:
        gt_trace: Ground truth trace
        pred_trace: Predicted trace
        chamfer_scale: Scale factor for exp(-scale * distance)
        normalize_proportions: If True, normalize values to proportions

    Returns:
        (score, details) where score is in [0, 1]
    """
    gt_type = infer_trace_type(gt_trace)

    # Extract data points
    gt_points, dim_types = extract_data_points(gt_trace, gt_type, normalize_proportions)
    pred_points, _ = extract_data_points(pred_trace, gt_type, normalize_proportions)

    # Compute data range for normalization
    all_points = gt_points + pred_points
    data_range = compute_data_range(all_points, dim_types)

    # Compute Chamfer distance
    d_chamfer = chamfer_distance(gt_points, pred_points, dim_types, data_range)

    # Convert to score: exp(-scale * distance)
    score = math.exp(-chamfer_scale * d_chamfer)

    details = {
        "chamfer_distance": d_chamfer,
        "n_gt_points": len(gt_points),
        "n_pred_points": len(pred_points),
        "data_range": data_range,
    }

    return score, details


def compute_data_score(
    gt_traces: List[dict],
    pred_traces: List[dict],
    matches: List[Tuple[Optional[int], Optional[int]]],
    chamfer_scale: float = 5.0,
    gt_layout: Optional[dict] = None,
    pred_layout: Optional[dict] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute S_Data score across all matched trace pairs.

    Formula: S_Data = (1/max(|T|,|P|)) * sum(exp(-scale * D_chamfer))

    For charts without axes (e.g., pie), values are normalized to proportions
    since absolute values cannot be visually determined.

    Args:
        gt_traces: Ground truth traces
        pred_traces: Predicted traces
        matches: List of (gt_idx, pred_idx) pairs
        chamfer_scale: Scale factor for exponential decay
        gt_layout: Ground truth layout (to check for axes)
        pred_layout: Predicted layout (to check for axes)

    Returns:
        (score, details) where score is in [0, 1]
    """
    n_gt = len(gt_traces)
    n_pred = len(pred_traces)
    n = max(n_gt, n_pred)

    if n == 0:
        return 1.0, {"total": 0, "traces": []}

    # Determine if we should normalize to proportions
    # If neither layout has axes, values are proportional (e.g., pie chart)
    gt_has_axes = has_axes(gt_layout or {})
    pred_has_axes = has_axes(pred_layout or {})
    normalize_proportions = not gt_has_axes and not pred_has_axes

    total_score = 0.0
    trace_details = []

    for gt_idx, pred_idx in matches:
        if gt_idx is None or pred_idx is None:
            # Unmatched trace contributes 0
            trace_details.append({
                "gt_idx": gt_idx,
                "pred_idx": pred_idx,
                "score": 0.0,
                "chamfer_distance": 1.0,
            })
        else:
            score, details = compute_trace_data_score(
                gt_traces[gt_idx],
                pred_traces[pred_idx],
                chamfer_scale,
                normalize_proportions,
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
        "chamfer_scale": chamfer_scale,
        "normalize_proportions": normalize_proportions,
        "traces": trace_details,
    }
