#!/usr/bin/env python3
"""
Run evaluation on agent outputs for Task 1 (Recreation) and Task 2 (QA).

Usage:
    python -m eval.run_eval haiku                              # All figures
    python -m eval.run_eval haiku --task task1 --config vision # Task 1 only
    python -m eval.run_eval haiku --check                      # Quality check
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .sss import SSSEvaluator
from .validator import validate_task1, validate_task2, validate_task1_file, validate_task2_file


BENCHMARK_ROOT = Path(__file__).parent.parent
DATA_ROOT = BENCHMARK_ROOT / "v1" / "test"

FIGURE_TYPES = [
    "vbar_categorical",
    "hbar_categorical",
    "pie",
    "line",
    "dot_line",
]


# =============================================================================
# Common utilities
# =============================================================================

def get_test_cases() -> List[str]:
    """Get all test cases from the dataset directory."""
    figure_ids = []
    for fig_dir in sorted(DATA_ROOT.iterdir()):
        if fig_dir.is_dir():
            figure_ids.append(fig_dir.name)
    return figure_ids


def get_figure_type(figure_id: str) -> str:
    """Extract figure type from figure ID."""
    for fig_type in FIGURE_TYPES:
        if figure_id.startswith(fig_type):
            return fig_type
    return "unknown"


def find_configs(output_dir: Path, task: str = "task1") -> List[str]:
    """Find all unique configs in output directory."""
    configs = set()
    pattern = f"output_*_{task}.json" if task == "task1" else "output_*_task2_q0.json"

    for fig_dir in output_dir.iterdir():
        if not fig_dir.is_dir():
            continue
        for f in fig_dir.glob(pattern):
            name = f.stem
            if task == "task1":
                config = name.replace("output_", "").replace("_task1", "")
            else:
                config = name.replace("output_", "").replace("_task2_q0", "")
            configs.add(config)

    return sorted(configs)


def shorten_config(config: str) -> str:
    """Shorten config name for display."""
    return config.replace("vision_", "v_").replace("interactive", "int").replace("lint", "lnt")


# =============================================================================
# Task 1: Recreation (SSS metrics)
# =============================================================================

@dataclass
class Task1Result:
    """Task 1 evaluation result."""
    figure_id: str
    figure_type: str
    config: str
    s_type: float = 0.0
    s_data: float = 0.0
    s_text: float = 0.0
    s_style: float = 0.0
    error: Optional[str] = None


def load_ground_truth_fig(figure_id: str) -> Optional[dict]:
    """Load ground truth figure JSON."""
    path = DATA_ROOT / figure_id / "fig.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_task1_prediction(output_dir: Path, figure_id: str, config: str) -> Optional[dict]:
    """Load Task 1 prediction."""
    path = output_dir / figure_id / f"output_{config}_task1.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def is_valid_task1_prediction(pred: dict) -> bool:
    """Check if Task 1 prediction is valid Plotly figure with actual data."""
    return validate_task1(pred).valid


def evaluate_task1_figure(evaluator: SSSEvaluator, output_dir: Path, figure_id: str, config: str) -> Task1Result:
    """Evaluate a single figure for Task 1."""
    fig_type = get_figure_type(figure_id)

    gt = load_ground_truth_fig(figure_id)
    if gt is None:
        return Task1Result(figure_id=figure_id, figure_type=fig_type, config=config, error="Ground truth not found")

    pred = load_task1_prediction(output_dir, figure_id, config)
    if pred is None:
        return Task1Result(figure_id=figure_id, figure_type=fig_type, config=config, error="Prediction not found")

    # Invalid prediction (failed validation) = error with 0.0 for all metrics
    if not is_valid_task1_prediction(pred):
        return Task1Result(
            figure_id=figure_id,
            figure_type=fig_type,
            config=config,
            s_type=0.0,
            s_data=0.0,
            s_text=0.0,
            s_style=0.0,
            error="Invalid prediction",
        )

    metrics = evaluator.evaluate(gt, pred, figure_id)
    return Task1Result(
        figure_id=figure_id,
        figure_type=fig_type,
        config=config,
        s_type=metrics.s_type,
        s_data=metrics.s_data,
        s_text=metrics.s_text,
        s_style=metrics.s_style,
        error=metrics.error,
    )


def run_task1_eval(output_dir: Path, configs: List[str], test_cases: List[str], results_dir: Path,
                   include_invalid: bool = False):
    """Run Task 1 evaluation."""
    print("\n" + "=" * 80)
    print("TASK 1: RECREATION (SSS Metrics)")
    if include_invalid:
        print("Mode: include invalid as 0")
    print("=" * 80)

    evaluator = SSSEvaluator(include_details=False)
    all_results: Dict[str, List[Task1Result]] = {}

    for config in configs:
        print(f"  Evaluating {config}...", end=" ", flush=True)
        results = [evaluate_task1_figure(evaluator, output_dir, fig_id, config) for fig_id in test_cases]
        all_results[config] = results
        valid = sum(1 for r in results if not r.error)
        invalid = sum(1 for r in results if r.error == "Invalid prediction")
        missing = sum(1 for r in results if r.error and r.error != "Invalid prediction")
        print(f"{valid} valid, {invalid} invalid, {missing} missing")

        # Write per-config CSV
        write_task1_csv(results, results_dir / f"task1_{config}.csv")

    # Print comparison
    print_task1_comparison(all_results, configs, include_invalid)

    # Write summary
    write_task1_summary(all_results, configs, results_dir / "task1_summary.csv", include_invalid)


def write_task1_csv(results: List[Task1Result], path: Path):
    """Write Task 1 results to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["figure_id", "figure_type", "config", "s_type", "s_data", "s_text", "s_style", "error"])
        for r in results:
            writer.writerow([r.figure_id, r.figure_type, r.config,
                             f"{r.s_type:.4f}", f"{r.s_data:.4f}", f"{r.s_text:.4f}", f"{r.s_style:.4f}",
                             r.error or ""])


def print_task1_comparison(all_results: Dict[str, List[Task1Result]], configs: List[str],
                           include_invalid: bool = False):
    """Print Task 1 agent comparison."""
    header = f"{'Metric':<12}" + "".join(f" {shorten_config(c):>12}" for c in configs)
    print("\n" + header)
    print("-" * len(header))

    for metric in ["s_type", "s_data", "s_text", "s_style"]:
        row = f"{metric:<12}"
        for config in configs:
            if include_invalid:
                # Include all results (invalid = 0)
                results = all_results[config]
                avg = sum(getattr(r, metric) for r in results) / len(results) if results else 0
            else:
                # Only valid results (no error)
                valid = [r for r in all_results[config] if not r.error]
                avg = sum(getattr(r, metric) for r in valid) / len(valid) if valid else 0
            row += f" {avg:>12.4f}"
        print(row)

    if include_invalid:
        row = f"{'total':<12}"
        for config in configs:
            row += f" {len(all_results[config]):>12}"
    else:
        row = f"{'valid':<12}"
        for config in configs:
            row += f" {sum(1 for r in all_results[config] if not r.error):>12}"
    print(row)


def write_task1_summary(all_results: Dict[str, List[Task1Result]], configs: List[str], path: Path,
                        include_invalid: bool = False):
    """Write Task 1 summary CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["figure_type", "metric"] + configs)

        for fig_type in FIGURE_TYPES + ["OVERALL"]:
            for metric in ["s_type", "s_data", "s_text", "s_style"]:
                row = [fig_type, metric]
                for config in configs:
                    if include_invalid:
                        # Include all results (invalid = 0)
                        if fig_type == "OVERALL":
                            results = all_results[config]
                        else:
                            results = [r for r in all_results[config] if r.figure_type == fig_type]
                        row.append(f"{sum(getattr(r, metric) for r in results) / len(results):.4f}" if results else "")
                    else:
                        # Only valid results (no error)
                        if fig_type == "OVERALL":
                            valid = [r for r in all_results[config] if not r.error]
                        else:
                            valid = [r for r in all_results[config] if not r.error and r.figure_type == fig_type]
                        row.append(f"{sum(getattr(r, metric) for r in valid) / len(valid):.4f}" if valid else "")
                writer.writerow(row)

            # Write count per figure type
            count_label = "total" if include_invalid else "valid"
            row = [fig_type, count_label]
            for config in configs:
                if include_invalid:
                    if fig_type == "OVERALL":
                        results = all_results[config]
                    else:
                        results = [r for r in all_results[config] if r.figure_type == fig_type]
                    row.append(str(len(results)))
                else:
                    if fig_type == "OVERALL":
                        valid = [r for r in all_results[config] if not r.error]
                    else:
                        valid = [r for r in all_results[config] if not r.error and r.figure_type == fig_type]
                    row.append(str(len(valid)))
            writer.writerow(row)


# =============================================================================
# Task 2: QA (Accuracy)
# =============================================================================

@dataclass
class Task2Result:
    """Task 2 evaluation result for a single question."""
    figure_id: str
    figure_type: str
    config: str
    question_idx: int
    question_id: int
    question_string: str
    gt_answer: int
    pred_answer: Optional[int]
    correct: bool
    error: Optional[str] = None


def load_qa_pairs(figure_id: str) -> Optional[List[dict]]:
    """Load ground truth QA pairs."""
    path = DATA_ROOT / figure_id / "qa_pairs.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_task2_prediction(output_dir: Path, figure_id: str, config: str, question_idx: int) -> Optional[int]:
    """Load Task 2 prediction for a specific question. Returns 0, 1, or None."""
    path = output_dir / figure_id / f"output_{config}_task2_q{question_idx}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        result = validate_task2(data)
        return result.answer if result.valid else None
    except json.JSONDecodeError:
        return None


def evaluate_task2_figure(output_dir: Path, figure_id: str, config: str) -> List[Task2Result]:
    """Evaluate all QA pairs for a figure."""
    fig_type = get_figure_type(figure_id)
    results = []

    qa_pairs = load_qa_pairs(figure_id)
    if qa_pairs is None:
        return [Task2Result(
            figure_id=figure_id, figure_type=fig_type, config=config,
            question_idx=-1, question_id=-1, question_string="",
            gt_answer=-1, pred_answer=None, correct=False, error="QA pairs not found"
        )]

    for idx, qa in enumerate(qa_pairs):
        gt_answer = qa.get("answer", -1)
        pred_answer = load_task2_prediction(output_dir, figure_id, config, idx)

        results.append(Task2Result(
            figure_id=figure_id, figure_type=fig_type, config=config,
            question_idx=idx, question_id=qa.get("question_id", -1),
            question_string=qa.get("question_string", ""),
            gt_answer=gt_answer, pred_answer=pred_answer,
            correct=(pred_answer == gt_answer) if pred_answer is not None else False,
        ))

    return results


def run_task2_eval(output_dir: Path, configs: List[str], test_cases: List[str], results_dir: Path,
                   include_invalid: bool = False):
    """Run Task 2 evaluation."""
    print("\n" + "=" * 80)
    print("TASK 2: QA (Accuracy)")
    if include_invalid:
        print("Mode: include invalid as wrong")
    print("=" * 80)

    all_results: Dict[str, List[Task2Result]] = {}

    for config in configs:
        print(f"  Evaluating {config}...", end=" ", flush=True)
        results = []
        for fig_id in test_cases:
            results.extend(evaluate_task2_figure(output_dir, fig_id, config))
        all_results[config] = results

        valid = [r for r in results if not r.error and r.pred_answer is not None]
        empty = [r for r in results if not r.error and r.pred_answer is None]
        errors = [r for r in results if r.error]
        correct = sum(1 for r in valid if r.correct)
        acc = 100 * correct / len(valid) if valid else 0
        print(f"{len(valid)} valid ({acc:.1f}% acc), {len(empty)} empty, {len(errors)} errors")

        # Write per-config CSV
        write_task2_csv(results, results_dir / f"task2_{config}.csv")

    # Print comparison
    print_task2_comparison(all_results, configs, include_invalid)

    # Write summary
    write_task2_summary(all_results, configs, results_dir / "task2_summary.csv", include_invalid)


def write_task2_csv(results: List[Task2Result], path: Path):
    """Write Task 2 results to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["figure_id", "figure_type", "config", "question_idx", "question_id",
                         "gt_answer", "pred_answer", "correct", "error"])
        for r in results:
            writer.writerow([r.figure_id, r.figure_type, r.config, r.question_idx, r.question_id,
                             r.gt_answer, r.pred_answer if r.pred_answer is not None else "",
                             int(r.correct), r.error or ""])


def print_task2_comparison(all_results: Dict[str, List[Task2Result]], configs: List[str],
                           include_invalid: bool = False):
    """Print Task 2 agent comparison."""
    header = f"{'Figure Type':<20}" + "".join(f" {shorten_config(c):>12}" for c in configs)
    print("\n" + header)
    print("-" * len(header))

    for fig_type in FIGURE_TYPES + ["OVERALL"]:
        row = f"{fig_type:<20}"
        for config in configs:
            if include_invalid:
                # Include all results (invalid = wrong)
                if fig_type == "OVERALL":
                    results = [r for r in all_results[config] if not r.error]
                else:
                    results = [r for r in all_results[config] if not r.error and r.figure_type == fig_type]
                if results:
                    acc = sum(1 for r in results if r.correct) / len(results)
                    row += f" {acc:>12.4f}"
                else:
                    row += f" {'-':>12}"
            else:
                # Only valid results (pred_answer is not None)
                if fig_type == "OVERALL":
                    valid = [r for r in all_results[config] if not r.error and r.pred_answer is not None]
                else:
                    valid = [r for r in all_results[config] if not r.error and r.pred_answer is not None and r.figure_type == fig_type]
                if valid:
                    acc = sum(1 for r in valid if r.correct) / len(valid)
                    row += f" {acc:>12.4f}"
                else:
                    row += f" {'-':>12}"
        print(row)

    # Print counts
    if include_invalid:
        row = f"{'total':<20}"
        for config in configs:
            row += f" {sum(1 for r in all_results[config] if not r.error):>12}"
    else:
        row = f"{'valid':<20}"
        for config in configs:
            row += f" {sum(1 for r in all_results[config] if not r.error and r.pred_answer is not None):>12}"
    print(row)


def write_task2_summary(all_results: Dict[str, List[Task2Result]], configs: List[str], path: Path,
                        include_invalid: bool = False):
    """Write Task 2 summary CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["figure_type", "metric"] + configs)

        for fig_type in FIGURE_TYPES + ["OVERALL"]:
            row = [fig_type, "accuracy"]
            for config in configs:
                if include_invalid:
                    # Include all results (invalid = wrong)
                    if fig_type == "OVERALL":
                        results = [r for r in all_results[config] if not r.error]
                    else:
                        results = [r for r in all_results[config] if not r.error and r.figure_type == fig_type]
                    if results:
                        acc = sum(1 for r in results if r.correct) / len(results)
                        row.append(f"{acc:.4f}")
                    else:
                        row.append("")
                else:
                    # Only valid results (pred_answer is not None)
                    if fig_type == "OVERALL":
                        valid = [r for r in all_results[config] if not r.error and r.pred_answer is not None]
                    else:
                        valid = [r for r in all_results[config] if not r.error and r.pred_answer is not None and r.figure_type == fig_type]
                    if valid:
                        acc = sum(1 for r in valid if r.correct) / len(valid)
                        row.append(f"{acc:.4f}")
                    else:
                        row.append("")
            writer.writerow(row)

            # Write count
            count_label = "total" if include_invalid else "valid"
            row = [fig_type, count_label]
            for config in configs:
                if include_invalid:
                    if fig_type == "OVERALL":
                        results = [r for r in all_results[config] if not r.error]
                    else:
                        results = [r for r in all_results[config] if not r.error and r.figure_type == fig_type]
                    row.append(str(len(results)))
                else:
                    if fig_type == "OVERALL":
                        valid = [r for r in all_results[config] if not r.error and r.pred_answer is not None]
                    else:
                        valid = [r for r in all_results[config] if not r.error and r.pred_answer is not None and r.figure_type == fig_type]
                    row.append(str(len(valid)))
            writer.writerow(row)


# =============================================================================
# Quality Check
# =============================================================================

@dataclass
class ValidationStats:
    """Validation statistics for a config/type combination."""
    valid: int = 0
    invalid: int = 0
    missing: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.valid + self.invalid + self.missing

    @property
    def valid_pct(self) -> float:
        return self.valid / self.total * 100 if self.total > 0 else 0


def check_output_quality(output_dir: Path, configs: List[str], test_cases: List[str]):
    """Check quality of outputs using validator module."""
    print("\n" + "=" * 80)
    print("OUTPUT QUALITY CHECK")
    print("=" * 80)

    fig_types = FIGURE_TYPES

    # Collect all validation results
    task1_stats: Dict[str, Dict[str, ValidationStats]] = defaultdict(lambda: defaultdict(ValidationStats))
    task2_stats: Dict[str, Dict[str, ValidationStats]] = defaultdict(lambda: defaultdict(ValidationStats))

    for config in configs:
        for fig_id in test_cases:
            fig_type = get_figure_type(fig_id)

            # Task 1 validation
            out_file = output_dir / fig_id / f"output_{config}_task1.json"
            if not out_file.exists():
                task1_stats[config][fig_type].missing += 1
            else:
                result = validate_task1_file(out_file)
                if result.valid:
                    task1_stats[config][fig_type].valid += 1
                else:
                    task1_stats[config][fig_type].invalid += 1
                    if len(task1_stats[config][fig_type].errors) < 3:
                        task1_stats[config][fig_type].errors.append(f"{fig_id}: {result.error}")

            # Task 2 validation
            qa_pairs = load_qa_pairs(fig_id)
            if qa_pairs:
                for idx in range(len(qa_pairs)):
                    out_file = output_dir / fig_id / f"output_{config}_task2_q{idx}.json"
                    if not out_file.exists():
                        task2_stats[config][fig_type].missing += 1
                    else:
                        result = validate_task2_file(out_file)
                        if result.valid:
                            task2_stats[config][fig_type].valid += 1
                        else:
                            task2_stats[config][fig_type].invalid += 1
                            if len(task2_stats[config][fig_type].errors) < 3:
                                task2_stats[config][fig_type].errors.append(f"{fig_id}/q{idx}: {result.error}")

    # Print Task 1 results
    print("\n--- Task 1 (Recreation) ---")
    print(f"{'Config':<25} {'Type':<20} {'Valid':>7} {'Invalid':>7} {'Missing':>7} {'Valid%':>8}")
    print("-" * 80)

    for config in configs:
        for fig_type in fig_types:
            stats = task1_stats[config][fig_type]
            if stats.invalid > 0 or stats.missing > 0:
                print(f"{config:<25} {fig_type:<20} {stats.valid:>7} {stats.invalid:>7} {stats.missing:>7} {stats.valid_pct:>7.1f}%")

    # Print Task 2 results
    print("\n--- Task 2 (QA) ---")
    print(f"{'Config':<25} {'Type':<20} {'Valid':>7} {'Invalid':>7} {'Missing':>7} {'Valid%':>8}")
    print("-" * 80)

    for config in configs:
        for fig_type in fig_types:
            stats = task2_stats[config][fig_type]
            if stats.invalid > 0 or stats.missing > 0:
                print(f"{config:<25} {fig_type:<20} {stats.valid:>7} {stats.invalid:>7} {stats.missing:>7} {stats.valid_pct:>7.1f}%")

    # Summary per config
    print("\n--- Summary by Config ---")
    print(f"{'Config':<25} {'Task':<8} {'Valid':>7} {'Invalid':>7} {'Missing':>7} {'Valid%':>8}")
    print("-" * 70)

    for config in configs:
        # Task 1 totals
        t1_valid = sum(task1_stats[config][ft].valid for ft in fig_types)
        t1_invalid = sum(task1_stats[config][ft].invalid for ft in fig_types)
        t1_missing = sum(task1_stats[config][ft].missing for ft in fig_types)
        t1_total = t1_valid + t1_invalid + t1_missing
        t1_pct = t1_valid / t1_total * 100 if t1_total > 0 else 0
        print(f"{config:<25} {'Task1':<8} {t1_valid:>7} {t1_invalid:>7} {t1_missing:>7} {t1_pct:>7.1f}%")

        # Task 2 totals
        t2_valid = sum(task2_stats[config][ft].valid for ft in fig_types)
        t2_invalid = sum(task2_stats[config][ft].invalid for ft in fig_types)
        t2_missing = sum(task2_stats[config][ft].missing for ft in fig_types)
        t2_total = t2_valid + t2_invalid + t2_missing
        t2_pct = t2_valid / t2_total * 100 if t2_total > 0 else 0
        print(f"{'':<25} {'Task2':<8} {t2_valid:>7} {t2_invalid:>7} {t2_missing:>7} {t2_pct:>7.1f}%")

    # Sample errors
    print("\n--- Sample Validation Errors ---")
    for config in configs[:1]:  # Only first config
        all_t1_errors = []
        all_t2_errors = []
        for ft in fig_types:
            all_t1_errors.extend(task1_stats[config][ft].errors)
            all_t2_errors.extend(task2_stats[config][ft].errors)

        if all_t1_errors:
            print(f"\nTask 1 ({config}):")
            for err in all_t1_errors[:5]:
                print(f"  - {err}")
            if len(all_t1_errors) > 5:
                print(f"  ... and {len(all_t1_errors) - 5} more")

        if all_t2_errors:
            print(f"\nTask 2 ({config}):")
            for err in all_t2_errors[:5]:
                print(f"  - {err}")
            if len(all_t2_errors) > 5:
                print(f"  ... and {len(all_t2_errors) - 5} more")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run iPlotBench evaluation")
    parser.add_argument("dir", type=str, nargs="?", default=None,
                        help="Model directory name (e.g., 'haiku')")
    parser.add_argument("--task", type=str, default="all", choices=["task1", "task2", "all"],
                        help="Task to evaluate")
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_ROOT / "env" / "output",
                        help="Base directory containing agent outputs")
    parser.add_argument("--config", type=str, default="all",
                        help="Config to evaluate (e.g., 'vision') or 'all'")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Directory to save results")
    parser.add_argument("--check", action="store_true",
                        help="Run quality check on outputs (valid/empty/missing breakdown)")
    parser.add_argument("--include-invalid", action="store_true",
                        help="Include invalid results as 0 in metrics (default: only valid)")
    args = parser.parse_args()

    # Build output directory path
    if args.dir:
        output_dir = args.output_dir / args.dir
        default_results_dir = BENCHMARK_ROOT / "eval" / "eval_results" / args.dir
    else:
        output_dir = args.output_dir
        default_results_dir = BENCHMARK_ROOT / "eval" / "eval_results"

    results_dir = args.results_dir or default_results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get test cases
    test_cases = get_test_cases()
    print(f"Test cases: {len(test_cases)}")

    # Determine configs
    if args.config == "all":
        task_for_config = "task1" if args.task in ["task1", "all"] else "task2"
        configs = find_configs(output_dir, task_for_config)
        if not configs:
            print(f"No configs found in {output_dir}")
            return 1
        print(f"Configs: {configs}")
    else:
        configs = [args.config]

    # Run quality check if requested
    if args.check:
        check_output_quality(output_dir, configs, test_cases)
        return 0

    # Run evaluations
    if args.task in ["task1", "all"]:
        run_task1_eval(output_dir, configs, test_cases, results_dir, args.include_invalid)

    if args.task in ["task2", "all"]:
        run_task2_eval(output_dir, configs, test_cases, results_dir, args.include_invalid)

    print(f"\nResults saved to: {results_dir}/")


if __name__ == "__main__":
    exit(main() or 0)
