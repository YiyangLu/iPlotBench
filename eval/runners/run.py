"""CLI for running iPlotBench evaluation with local LLMs.

Usage:
    # Run with vision config (no tools, direct generation)
    python -m eval.runners.run qwen3vl --config vision

    # Run with vision_introspect_interactive config (all MCP tools)
    python -m eval.runners.run qwen3vl --config vision_introspect_interactive

    # Run specific figure types
    python -m eval.runners.run qwen3vl --config vision --types vbar_categorical pie

    # Run specific figures
    python -m eval.runners.run qwen3vl --config vision --figures vbar_categorical_0000

    # Run with custom endpoint
    python -m eval.runners.run qwen3vl --config vision --api-base http://localhost:8000/v1

    # Resume interrupted run
    python -m eval.runners.run qwen3vl --config vision --resume
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .local_llm import RunnerConfig, run_figure, AGENT_CONFIGS

# Figure types in the benchmark
FIGURE_TYPES = ["vbar_categorical", "hbar_categorical", "pie", "line", "dot_line"]


def get_figure_ids(
    data_dir: Path,
    figure_types: list[str] | None = None,
    figure_ids: list[str] | None = None,
) -> list[str]:
    """Get list of figure IDs to process."""
    if figure_ids:
        return figure_ids

    types_to_process = figure_types or FIGURE_TYPES
    ids = []

    test_dir = data_dir / "test"
    if test_dir.exists():
        for path in sorted(test_dir.iterdir()):
            if path.is_dir():
                for t in types_to_process:
                    if path.name.startswith(t):
                        ids.append(path.name)
                        break

    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Run iPlotBench evaluation with local LLM"
    )
    parser.add_argument(
        "model_name",
        help="Model name for output directory (e.g., 'qwen3vl')",
    )
    parser.add_argument(
        "--config",
        choices=list(AGENT_CONFIGS.keys()),
        required=True,
        help="Agent config: 'vision' (no tools) or 'vision_introspect_interactive' (all tools)",
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8666/v1",
        help="API base URL (default: http://127.0.0.1:8666/v1)",
    )
    parser.add_argument(
        "--api-key",
        default="none",
        help="API key (default: 'none' for local vLLM)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Model ID to use (default: Qwen/Qwen3-VL-4B-Instruct)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=FIGURE_TYPES,
        help="Figure types to process (default: all)",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        help="Specific figure IDs to process",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("env"),
        help="Data directory (default: env)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base output directory; {model_name} subfolder created automatically (default: localenv/output)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens for generation (default: 4096)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for generation (default: 0.1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="API timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip figures that already have output files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = args.data_dir.resolve()
    # Always create {model_name} subfolder in output directory
    if args.output_dir:
        output_dir = args.output_dir / args.model_name
    else:
        # Use localenv/output by default to avoid Docker permission issues
        output_dir = Path("localenv") / "output" / args.model_name
    output_dir = output_dir.resolve()

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Get figure IDs
    figure_ids = get_figure_ids(data_dir, args.types, args.figures)
    if not figure_ids:
        print("No figures found to process")
        sys.exit(1)

    # Filter already processed if resuming
    config_name = args.config
    if args.resume:
        remaining = []
        for fig_id in figure_ids:
            task1_path = output_dir / fig_id / f"output_{config_name}_task1.json"
            if not task1_path.exists():
                remaining.append(fig_id)
        skipped = len(figure_ids) - len(remaining)
        if skipped > 0:
            print(f"Resuming: skipping {skipped} already processed figures")
        figure_ids = remaining

    if not figure_ids:
        print("All figures already processed")
        return

    # Setup extra headers for OpenRouter
    extra_headers = {}
    if "openrouter.ai" in args.api_base:
        extra_headers = {
            "HTTP-Referer": "https://github.com/iPlotBench",
            "X-Title": "iPlotBench",
        }

    # Create runner config from preset
    config = RunnerConfig.from_preset(
        args.config,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        logs_root=output_dir / "logs",
        extra_headers=extra_headers,
    )

    print(f"=" * 60)
    print(f"iPlotBench Local LLM Evaluation")
    print(f"=" * 60)
    print(f"Config: {config_name}")
    print(f"Model: {args.model}")
    print(f"API base: {args.api_base}")
    print(f"Tools: {'enabled' if config.use_tools else 'disabled'}")
    print(f"Figures: {len(figure_ids)}")
    print(f"Workers: {args.workers}")
    print(f"Output: {output_dir}")
    print(f"=" * 60)
    print()

    # Process figures
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    completed = 0
    task1_success = 0
    task2_total = 0
    task2_success = 0

    import threading
    lock = threading.Lock()

    def process_figure(fig_id):
        nonlocal completed, task1_success, task2_total, task2_success
        try:
            result = run_figure(fig_id, config, data_dir, output_dir, config_name)
            with lock:
                completed += 1
                if result["task1"] and result["task1"]["success"]:
                    task1_success += 1
                for t2 in result["task2"]:
                    task2_total += 1
                    if t2["success"]:
                        task2_success += 1
                # Progress update
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(figure_ids) - completed) / rate if rate > 0 else 0
                print(
                    f"\r[{completed}/{len(figure_ids)}] {fig_id} | "
                    f"T1: {task1_success}/{completed} | "
                    f"T2: {task2_success}/{task2_total} | "
                    f"ETA: {eta:.0f}s",
                    end="",
                    flush=True,
                )
            return True
        except Exception as e:
            with lock:
                completed += 1
                print(f"\nError processing {fig_id}: {e}")
            return False

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(process_figure, figure_ids))
    else:
        for fig_id in figure_ids:
            process_figure(fig_id)

    print()  # Newline after progress
    elapsed = time.time() - start_time

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Config: {config_name}")
    print(f"Total figures: {completed}")
    print(f"Task 1 success: {task1_success}/{completed} ({100*task1_success/completed:.1f}%)")
    if task2_total > 0:
        print(f"Task 2 success: {task2_success}/{task2_total} ({100*task2_success/task2_total:.1f}%)")
    print(f"Total time: {elapsed:.1f}s ({elapsed/completed:.2f}s per figure)")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
