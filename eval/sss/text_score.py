"""
S_Text: Text correctness score.

Uses role-aware text extraction and fuzzy Jaccard similarity.
"""

from typing import Dict, Set, Tuple, Any
from difflib import SequenceMatcher

# Try to import rapidfuzz for faster fuzzy matching
try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


def fuzzy_ratio(s1: str, s2: str) -> float:
    """
    Compute fuzzy string similarity ratio.

    Returns:
        Similarity in [0, 1]
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    if HAS_RAPIDFUZZ:
        return rapidfuzz_fuzz.ratio(s1, s2) / 100.0
    else:
        return SequenceMatcher(None, s1, s2).ratio()


def fuzzy_jaccard(
    gt_texts: Set[str],
    pred_texts: Set[str],
    threshold: float = 0.8,
) -> float:
    """
    Compute fuzzy Jaccard similarity between two text sets.

    For each GT text, find best fuzzy match in pred.
    Match if similarity >= threshold.

    Args:
        gt_texts: Ground truth text set
        pred_texts: Predicted text set
        threshold: Minimum similarity for a match

    Returns:
        Similarity in [0, 1]
    """
    if not gt_texts and not pred_texts:
        return 1.0
    if not gt_texts or not pred_texts:
        return 0.0

    # Find fuzzy matches
    matches = 0
    matched_pred = set()

    for gt_text in gt_texts:
        gt_lower = gt_text.lower().strip()
        best_score = 0.0
        best_pred = None

        for pred_text in pred_texts:
            if pred_text in matched_pred:
                continue

            pred_lower = pred_text.lower().strip()
            score = fuzzy_ratio(gt_lower, pred_lower)

            if score > best_score:
                best_score = score
                best_pred = pred_text

        if best_score >= threshold and best_pred is not None:
            matches += 1
            matched_pred.add(best_pred)

    # Jaccard-style: intersection / union
    union = len(gt_texts) + len(pred_texts) - matches
    return matches / union if union > 0 else 1.0


def extract_text_buckets(fig: dict) -> Dict[str, Set[str]]:
    """
    Extract texts organized by semantic role.

    Buckets:
    - title: layout.title.text
    - axis: layout.xaxis.title.text, layout.yaxis.title.text
    - legend: data[i].name
    - data: data[i].text, layout.annotations[j].text

    Returns:
        Dict mapping bucket name to set of text strings
    """
    buckets: Dict[str, Set[str]] = {
        "title": set(),
        "axis": set(),
        "legend": set(),
        "data": set(),
    }

    layout = fig.get("layout", {})
    data = fig.get("data", [])

    # Title
    title = layout.get("title", {})
    if isinstance(title, dict):
        text = title.get("text", "")
    else:
        text = str(title) if title else ""
    if text and text.strip():
        buckets["title"].add(text.strip())

    # Axis labels
    for axis_key in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
        axis = layout.get(axis_key, {})
        if not isinstance(axis, dict):
            continue

        axis_title = axis.get("title", {})
        if isinstance(axis_title, dict):
            text = axis_title.get("text", "")
        else:
            text = str(axis_title) if axis_title else ""
        if text and text.strip():
            buckets["axis"].add(text.strip())

    # Legend (trace names)
    for trace in data:
        name = trace.get("name", "")
        if name and str(name).strip():
            buckets["legend"].add(str(name).strip())

    # Data labels and annotations
    for trace in data:
        # Trace text
        text_arr = trace.get("text", [])
        if isinstance(text_arr, list):
            for t in text_arr:
                if t and str(t).strip():
                    buckets["data"].add(str(t).strip())
        elif text_arr and str(text_arr).strip():
            buckets["data"].add(str(text_arr).strip())

        # Pie labels (also count as data labels)
        labels = trace.get("labels", [])
        if isinstance(labels, list):
            for label in labels:
                if label and str(label).strip():
                    buckets["data"].add(str(label).strip())

        # Bar x/y categorical labels (add to data bucket)
        trace_type = trace.get("type", "")
        if trace_type == "bar":
            orientation = trace.get("orientation", "v")
            cat_key = "y" if orientation == "h" else "x"
            categories = trace.get(cat_key, [])
            if isinstance(categories, list):
                for cat in categories:
                    if cat and str(cat).strip():
                        buckets["data"].add(str(cat).strip())

    # Layout annotations
    for ann in layout.get("annotations", []):
        if isinstance(ann, dict):
            text = ann.get("text", "")
            if text and str(text).strip():
                buckets["data"].add(str(text).strip())

    return buckets


def compute_text_score(
    gt: dict,
    pred: dict,
    bucket_weights: Dict[str, float] = None,
    fuzzy_threshold: float = 0.8,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute S_Text with role-aware bucketing.

    Args:
        gt: Ground truth figure
        pred: Predicted figure
        bucket_weights: Weights for each bucket (default: title=0.3, axis=0.25, legend=0.25, data=0.2)
        fuzzy_threshold: Minimum similarity for fuzzy matching

    Returns:
        (score, details) where score is in [0, 1]
    """
    if bucket_weights is None:
        bucket_weights = {
            "title": 0.30,
            "axis": 0.25,
            "legend": 0.25,
            "data": 0.20,
        }

    gt_buckets = extract_text_buckets(gt)
    pred_buckets = extract_text_buckets(pred)

    total_score = 0.0
    total_weight = 0.0
    bucket_details = {}

    for bucket, weight in bucket_weights.items():
        gt_texts = gt_buckets.get(bucket, set())
        pred_texts = pred_buckets.get(bucket, set())

        # Skip empty buckets (don't penalize)
        if not gt_texts and not pred_texts:
            bucket_details[bucket] = {
                "gt": [],
                "pred": [],
                "score": 1.0,
                "weight": weight,
                "skipped": True,
            }
            continue

        score = fuzzy_jaccard(gt_texts, pred_texts, fuzzy_threshold)
        total_score += weight * score
        total_weight += weight

        bucket_details[bucket] = {
            "gt": list(gt_texts),
            "pred": list(pred_texts),
            "score": score,
            "weight": weight,
            "skipped": False,
        }

    # Normalize by actual weight used
    final_score = total_score / total_weight if total_weight > 0 else 1.0

    return final_score, {
        "buckets": bucket_details,
        "total_weight": total_weight,
        "fuzzy_threshold": fuzzy_threshold,
    }
