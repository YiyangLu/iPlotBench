"""
S_Type: Chart type correctness score.

Verifies that each trace has the correct chart type.
"""

from typing import List, Tuple, Optional, Dict, Any

from .trace_matching import infer_trace_type


def compute_type_score(
    gt_traces: List[dict],
    pred_traces: List[dict],
    matches: List[Tuple[Optional[int], Optional[int]]],
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute chart type matching score.

    Formula: S_Type = (1/max(|T|,|P|)) * sum(1 if type_match else 0)

    Args:
        gt_traces: Ground truth traces
        pred_traces: Predicted traces
        matches: List of (gt_idx, pred_idx) pairs from Hungarian matching

    Returns:
        (score, details) where score is in [0, 1]
    """
    n_gt = len(gt_traces)
    n_pred = len(pred_traces)
    n = max(n_gt, n_pred)

    if n == 0:
        return 1.0, {"matched": 0, "total": 0, "traces": []}

    matched_count = 0
    trace_details = []

    for gt_idx, pred_idx in matches:
        if gt_idx is None:
            # Extra predicted trace (no GT match)
            pred_type = infer_trace_type(pred_traces[pred_idx])
            trace_details.append({
                "gt_idx": None,
                "pred_idx": pred_idx,
                "gt_type": None,
                "pred_type": pred_type,
                "match": False,
            })
        elif pred_idx is None:
            # Missing predicted trace (GT not matched)
            gt_type = infer_trace_type(gt_traces[gt_idx])
            trace_details.append({
                "gt_idx": gt_idx,
                "pred_idx": None,
                "gt_type": gt_type,
                "pred_type": None,
                "match": False,
            })
        else:
            # Both traces exist
            gt_type = infer_trace_type(gt_traces[gt_idx])
            pred_type = infer_trace_type(pred_traces[pred_idx])
            is_match = gt_type == pred_type

            if is_match:
                matched_count += 1

            trace_details.append({
                "gt_idx": gt_idx,
                "pred_idx": pred_idx,
                "gt_type": gt_type,
                "pred_type": pred_type,
                "match": is_match,
            })

    score = matched_count / n

    details = {
        "matched": matched_count,
        "total": n,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "traces": trace_details,
    }

    return score, details
