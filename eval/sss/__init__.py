"""
Semantic Structural Similarity (SSS) metrics for iPlotBench Task 1.

Evaluates figure similarity across four dimensions:
- S_Type: Chart type correctness
- S_Data: Data accuracy (Chamfer distance)
- S_Text: Text correctness (Fuzzy Jaccard)
- S_Style: Visual style similarity
"""

from .core import SSSMetrics, SSSEvaluator

__all__ = ["SSSMetrics", "SSSEvaluator"]
