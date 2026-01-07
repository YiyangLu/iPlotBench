"""
iPlotBench Evaluation.

Semantic Structural Similarity (SSS) metrics for Task 1 (Recreation).

Usage:
    from eval import SSSEvaluator, SSSMetrics

    evaluator = SSSEvaluator()
    metrics = evaluator.evaluate(gt, pred, "figure_id")

    # Access individual scores (all in [0, 1])
    print(metrics.s_type)   # Chart type correctness
    print(metrics.s_data)   # Data accuracy (Chamfer-based)
    print(metrics.s_text)   # Text correctness (Fuzzy Jaccard)
    print(metrics.s_style)  # Visual style similarity
"""

from .sss import SSSMetrics, SSSEvaluator

__all__ = ["SSSMetrics", "SSSEvaluator"]
