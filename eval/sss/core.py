"""
Core SSS evaluator and metrics dataclass.

Semantic Structural Similarity (SSS) metric for iPlotBench Task 1.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

from .trace_matching import hungarian_match
from .type_score import compute_type_score
from .data_score import compute_data_score
from .text_score import compute_text_score
from .style_score import compute_style_score


@dataclass
class SSSMetrics:
    """Semantic Structural Similarity metrics for Task 1 (Recreation)."""

    figure_id: str

    # Component scores (0.0 to 1.0)
    s_type: float = 0.0   # Chart type correctness
    s_data: float = 0.0   # Data accuracy (Chamfer-based)
    s_text: float = 0.0   # Text correctness (Fuzzy Jaccard)
    s_style: float = 0.0  # Visual style similarity

    # Detailed breakdowns
    type_details: Optional[Dict[str, Any]] = None
    data_details: Optional[Dict[str, Any]] = None
    text_details: Optional[Dict[str, Any]] = None
    style_details: Optional[Dict[str, Any]] = None

    # Trace matching info
    trace_matching: Optional[List[Tuple[Optional[int], Optional[int]]]] = None
    n_gt_traces: int = 0
    n_pred_traces: int = 0

    # Error info
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "figure_id": self.figure_id,
            "s_type": self.s_type,
            "s_data": self.s_data,
            "s_text": self.s_text,
            "s_style": self.s_style,
            "n_gt_traces": self.n_gt_traces,
            "n_pred_traces": self.n_pred_traces,
            "error": self.error,
            "type_details": self.type_details,
            "data_details": self.data_details,
            "text_details": self.text_details,
            "style_details": self.style_details,
        }

    def summary_dict(self) -> Dict[str, Any]:
        """Return summary without detailed breakdowns."""
        return {
            "figure_id": self.figure_id,
            "s_type": self.s_type,
            "s_data": self.s_data,
            "s_text": self.s_text,
            "s_style": self.s_style,
            "n_gt_traces": self.n_gt_traces,
            "n_pred_traces": self.n_pred_traces,
            "error": self.error,
        }


class SSSEvaluator:
    """
    Semantic Structural Similarity evaluator.

    Computes four orthogonal metric dimensions:
    - S_Type: Chart type correctness
    - S_Data: Data accuracy using Chamfer distance
    - S_Text: Text correctness using Fuzzy Jaccard
    - S_Style: Visual style similarity
    """

    def __init__(
        self,
        chamfer_scale: float = 5.0,
        fuzzy_threshold: float = 0.8,
        include_details: bool = True,
    ):
        """
        Initialize evaluator.

        Args:
            chamfer_scale: Scale factor for exp(-scale * distance) in S_Data
            fuzzy_threshold: Minimum similarity for text matching
            include_details: Whether to include detailed breakdowns
        """
        self.chamfer_scale = chamfer_scale
        self.fuzzy_threshold = fuzzy_threshold
        self.include_details = include_details

    def evaluate(
        self,
        gt: dict,
        pred: dict,
        figure_id: str = "",
    ) -> SSSMetrics:
        """
        Compute all SSS metrics.

        Args:
            gt: Ground truth Plotly figure dict
            pred: Predicted Plotly figure dict
            figure_id: Figure identifier

        Returns:
            SSSMetrics with all scores and details
        """
        metrics = SSSMetrics(figure_id=figure_id)

        try:
            # Extract traces
            gt_traces = gt.get("data", [])
            pred_traces = pred.get("data", [])

            metrics.n_gt_traces = len(gt_traces)
            metrics.n_pred_traces = len(pred_traces)

            # Hungarian matching
            matches = hungarian_match(gt_traces, pred_traces)
            metrics.trace_matching = matches

            # S_Type: Chart type correctness
            s_type, type_details = compute_type_score(gt_traces, pred_traces, matches)
            metrics.s_type = s_type
            if self.include_details:
                metrics.type_details = type_details

            # S_Data: Data accuracy
            gt_layout = gt.get("layout", {})
            pred_layout = pred.get("layout", {})
            s_data, data_details = compute_data_score(
                gt_traces, pred_traces, matches, self.chamfer_scale,
                gt_layout=gt_layout, pred_layout=pred_layout
            )
            metrics.s_data = s_data
            if self.include_details:
                metrics.data_details = data_details

            # S_Text: Text correctness
            s_text, text_details = compute_text_score(
                gt, pred, fuzzy_threshold=self.fuzzy_threshold
            )
            metrics.s_text = s_text
            if self.include_details:
                metrics.text_details = text_details

            # S_Style: Visual style
            s_style, style_details = compute_style_score(gt_traces, pred_traces, matches)
            metrics.s_style = s_style
            if self.include_details:
                metrics.style_details = style_details

        except Exception as e:
            metrics.error = str(e)
            # Set all scores to 0 on error
            metrics.s_type = 0.0
            metrics.s_data = 0.0
            metrics.s_text = 0.0
            metrics.s_style = 0.0

        return metrics


def evaluate_figure(
    gt: dict,
    pred: dict,
    figure_id: str = "",
    **kwargs,
) -> SSSMetrics:
    """
    Convenience function to evaluate a single figure.

    Args:
        gt: Ground truth Plotly figure dict
        pred: Predicted Plotly figure dict
        figure_id: Figure identifier
        **kwargs: Additional arguments for SSSEvaluator

    Returns:
        SSSMetrics with all scores
    """
    evaluator = SSSEvaluator(**kwargs)
    return evaluator.evaluate(gt, pred, figure_id)
