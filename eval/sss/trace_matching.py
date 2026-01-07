"""
Trace matching using Hungarian algorithm.

Aligns traces between ground truth and prediction figures
to enable pairwise comparison.
"""

from typing import List, Tuple, Optional
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def infer_trace_type(trace: dict) -> str:
    """
    Infer chart type from trace.

    Returns:
        One of: 'line', 'dot_line', 'vbar', 'hbar', 'pie', 'scatter', 'unknown'
    """
    trace_type = trace.get("type", "scatter")

    if trace_type == "pie":
        return "pie"
    elif trace_type == "bar":
        orientation = trace.get("orientation", "v")
        return "hbar" if orientation == "h" else "vbar"
    elif trace_type == "scatter":
        mode = trace.get("mode", "")
        if "markers" in mode and "lines" in mode:
            return "dot_line"
        elif "lines" in mode:
            return "line"
        elif "markers" in mode:
            return "scatter"
        return "dot_line"  # Default for scatter without mode (Plotly default is lines+markers)

    return trace_type if trace_type else "unknown"


def quick_data_distance(gt_trace: dict, pred_trace: dict) -> float:
    """
    Compute quick data distance for matching (not full Chamfer).

    Uses simplified comparison based on data statistics.

    Returns:
        Distance in [0, 1], where 0 = identical
    """
    gt_type = infer_trace_type(gt_trace)
    pred_type = infer_trace_type(pred_trace)

    # Type mismatch penalty
    if gt_type != pred_type:
        return 1.0

    # Extract data based on type
    if gt_type == "pie":
        gt_values = gt_trace.get("values", [])
        pred_values = pred_trace.get("values", [])
    else:
        gt_values = gt_trace.get("y", gt_trace.get("x", []))
        pred_values = pred_trace.get("y", pred_trace.get("x", []))

    # Convert to numeric arrays
    try:
        gt_arr = np.array([float(v) for v in gt_values if _is_numeric(v)])
        pred_arr = np.array([float(v) for v in pred_values if _is_numeric(v)])
    except (ValueError, TypeError):
        return 0.5  # Moderate penalty for conversion failure

    if len(gt_arr) == 0 and len(pred_arr) == 0:
        return 0.0
    if len(gt_arr) == 0 or len(pred_arr) == 0:
        return 1.0

    # Compare statistics: mean, std, range
    gt_mean, pred_mean = np.mean(gt_arr), np.mean(pred_arr)
    gt_std, pred_std = np.std(gt_arr), np.std(pred_arr)
    gt_range = np.ptp(gt_arr)  # max - min
    pred_range = np.ptp(pred_arr)

    # Normalize distances
    scale = max(gt_range, pred_range, 1e-10)

    mean_diff = abs(gt_mean - pred_mean) / scale
    std_diff = abs(gt_std - pred_std) / scale
    range_diff = abs(gt_range - pred_range) / scale

    # Weighted combination
    dist = 0.5 * mean_diff + 0.3 * std_diff + 0.2 * range_diff
    return min(dist, 1.0)


def _is_numeric(v) -> bool:
    """Check if value is numeric."""
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def compute_trace_distance(gt_trace: dict, pred_trace: dict) -> float:
    """
    Compute combined distance for Hungarian matching.

    Args:
        gt_trace: Ground truth trace
        pred_trace: Predicted trace

    Returns:
        Distance in [0, 1]
    """
    gt_type = infer_trace_type(gt_trace)
    pred_type = infer_trace_type(pred_trace)

    # Type penalty (0 if match, 0.3 if mismatch)
    type_penalty = 0.0 if gt_type == pred_type else 0.3

    # Data distance
    data_dist = quick_data_distance(gt_trace, pred_trace)

    # Combined: type + data
    return type_penalty + 0.7 * data_dist


def hungarian_match(
    gt_traces: List[dict], pred_traces: List[dict]
) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Match traces using Hungarian algorithm.

    Returns list of (gt_idx, pred_idx) pairs.
    Unmatched traces have the other index as None.

    Args:
        gt_traces: List of ground truth traces
        pred_traces: List of predicted traces

    Returns:
        List of (gt_idx, pred_idx) tuples
    """
    n_gt = len(gt_traces)
    n_pred = len(pred_traces)

    if n_gt == 0 and n_pred == 0:
        return []
    if n_gt == 0:
        return [(None, j) for j in range(n_pred)]
    if n_pred == 0:
        return [(i, None) for i in range(n_gt)]

    # Build cost matrix
    n = max(n_gt, n_pred)
    cost_matrix = np.ones((n, n))  # Default high cost

    for i in range(n_gt):
        for j in range(n_pred):
            cost_matrix[i, j] = compute_trace_distance(gt_traces[i], pred_traces[j])

    # Run Hungarian algorithm
    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    else:
        # Fallback: greedy matching
        row_ind, col_ind = _greedy_match(cost_matrix, n_gt, n_pred)

    # Build result list
    matches = []
    gt_matched = set()
    pred_matched = set()

    for i, j in zip(row_ind, col_ind):
        if i < n_gt and j < n_pred:
            matches.append((i, j))
            gt_matched.add(i)
            pred_matched.add(j)

    # Add unmatched GT traces
    for i in range(n_gt):
        if i not in gt_matched:
            matches.append((i, None))

    # Add unmatched pred traces
    for j in range(n_pred):
        if j not in pred_matched:
            matches.append((None, j))

    return matches


def _greedy_match(cost_matrix: np.ndarray, n_gt: int, n_pred: int) -> Tuple[List[int], List[int]]:
    """
    Greedy matching fallback when scipy is not available.

    Not optimal but works for small matrices.
    """
    row_ind = []
    col_ind = []
    used_cols = set()

    for i in range(n_gt):
        best_j = -1
        best_cost = float("inf")
        for j in range(n_pred):
            if j not in used_cols and cost_matrix[i, j] < best_cost:
                best_cost = cost_matrix[i, j]
                best_j = j
        if best_j >= 0:
            row_ind.append(i)
            col_ind.append(best_j)
            used_cols.add(best_j)

    return row_ind, col_ind
