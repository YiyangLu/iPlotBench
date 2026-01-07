"""Prepare local environment for local LLM evaluation.

Creates localenv/ directory structure mirroring env/ but separate from Docker results.

Usage:
    python -m eval.runners.prepare_localenv
    python -m eval.runners.prepare_localenv --output ./localenv
    python -m eval.runners.prepare_localenv --types vbar_categorical pie
"""

import argparse
import json
import shutil
from pathlib import Path


def prepare_localenv(
    source_dir: Path,
    output_dir: Path,
    figure_types: list[str] | None = None,
    force: bool = False,
) -> None:
    """Prepare local environment for evaluation.

    Args:
        source_dir: Path to v1/test/ directory with ground truth
        output_dir: Path to create localenv/ structure
        figure_types: Optional list of figure types to include
        force: Overwrite existing directories
    """
    test_dir = output_dir / "test"
    query_dir = output_dir / "query"
    output_results_dir = output_dir / "output"

    # Check if already exists
    if output_dir.exists() and not force:
        print(f"Output directory already exists: {output_dir}")
        print("Use --force to overwrite")
        return

    # Create directories
    test_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    output_results_dir.mkdir(parents=True, exist_ok=True)

    # Process each figure
    count = 0
    for figure_dir in sorted(source_dir.iterdir()):
        if not figure_dir.is_dir():
            continue

        figure_id = figure_dir.name

        # Filter by type if specified
        if figure_types:
            matched = any(figure_id.startswith(t) for t in figure_types)
            if not matched:
                continue

        # Copy input.png to test/
        src_png = figure_dir / "fig.png"
        dst_test_dir = test_dir / figure_id
        dst_test_dir.mkdir(exist_ok=True)

        if src_png.exists():
            shutil.copy(src_png, dst_test_dir / "input.png")

        # Create questions.json (without answers) in query/
        src_qa = figure_dir / "qa_pairs.json"
        dst_query_dir = query_dir / figure_id
        dst_query_dir.mkdir(exist_ok=True)

        if src_qa.exists():
            with open(src_qa) as f:
                qa_pairs = json.load(f)

            # Remove answers, keep only questions
            questions = []
            for i, qa in enumerate(qa_pairs):
                questions.append({
                    "question_id": qa.get("question_id", i),
                    "question_string": qa.get("question_string", qa.get("question", "")),
                })

            with open(dst_query_dir / "questions.json", "w") as f:
                json.dump(questions, f, indent=2)

        count += 1
        if count % 50 == 0:
            print(f"Processed {count} figures...")

    print(f"\nPrepared {count} figures in {output_dir}")
    print(f"  - test/: input images")
    print(f"  - query/: questions (no answers)")
    print(f"  - output/: (empty, for results)")


def main():
    parser = argparse.ArgumentParser(description="Prepare local environment for LLM evaluation")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("v1/test"),
        help="Source directory with ground truth (default: v1/test)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("localenv"),
        help="Output directory (default: localenv)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        help="Figure types to include (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory",
    )

    args = parser.parse_args()

    source_dir = args.source.resolve()
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return

    prepare_localenv(source_dir, args.output.resolve(), args.types, args.force)


if __name__ == "__main__":
    main()
