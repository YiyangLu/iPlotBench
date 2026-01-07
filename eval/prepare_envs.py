#!/usr/bin/env python3
"""
Prepare environments for iPlotBench evaluation.

Creates:
- env/test/{figure_id}/input.png - Input images
- env/query/{figure_id}/questions.json - Questions only (no answers)
- env/output/ - For agent outputs

Ground truth (answers) is NOT included.

Usage:
    python -m eval.prepare_envs --output ./env
    python -m eval.prepare_envs --output ./env --limit 50
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed


BENCHMARK_ROOT = Path(__file__).parent.parent
DATA_ROOT = BENCHMARK_ROOT / "v1" / "test"


def get_figure_ids(limit: Optional[int] = None) -> List[str]:
    """Get all figure IDs from the test split."""
    figure_dirs = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])
    if limit:
        figure_dirs = figure_dirs[:limit]
    return figure_dirs


def extract_questions(qa_pairs: list) -> list:
    """Extract questions only (no answers) from qa_pairs."""
    return [
        {
            "question_id": q["question_id"],
            "question_string": q["question_string"],
        }
        for q in qa_pairs
    ]


def prepare_one_env(figure_id: str, output_dir: Path) -> dict:
    """Prepare environment for one figure."""
    source_dir = DATA_ROOT / figure_id
    fig_png = source_dir / "fig.png"
    qa_json = source_dir / "qa_pairs.json"

    if not fig_png.exists():
        return {"figure_id": figure_id, "status": "error", "message": "fig.png not found"}

    # Create test/{figure_id}/input.png
    test_dir = output_dir / "test" / figure_id
    test_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(fig_png, test_dir / "input.png")

    # Create query/{figure_id}/questions.json (if qa_pairs exists)
    if qa_json.exists():
        query_dir = output_dir / "query" / figure_id
        query_dir.mkdir(parents=True, exist_ok=True)

        qa_pairs = json.loads(qa_json.read_text())
        questions = extract_questions(qa_pairs)
        (query_dir / "questions.json").write_text(json.dumps(questions, indent=2))

    return {"figure_id": figure_id, "status": "ok"}


def main():
    parser = argparse.ArgumentParser(description="Prepare iPlotBench environments")
    parser.add_argument("--output", type=Path, default=BENCHMARK_ROOT / "env",
                        help="Output directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of figures")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    args = parser.parse_args()

    figure_ids = get_figure_ids(args.limit)
    print(f"Preparing {len(figure_ids)} environments")

    # Create output directory
    (args.output / "output").mkdir(parents=True, exist_ok=True)

    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(prepare_one_env, fig_id, args.output): fig_id
            for fig_id in figure_ids
        }

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result["status"] != "ok":
                errors.append(result)
            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i + 1}/{len(figure_ids)}")

    print(f"\nCompleted: {len(figure_ids) - len(errors)}/{len(figure_ids)}")
    if errors:
        print(f"Errors: {len(errors)}")

    print(f"\nStructure:")
    print(f"  {args.output}/test/{{figure_id}}/input.png")
    print(f"  {args.output}/query/{{figure_id}}/questions.json")
    print(f"  {args.output}/output/")


if __name__ == "__main__":
    exit(main() or 0)
