#!/usr/bin/env python
"""Convert a Plotly figure JSON file to an HTML page.

Examples:
  python scripts/showweb.py v1/test/pie_0042/fig.json -o pie_0042.html
  python scripts/showweb.py figure.json --offline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import plotly.io as pio


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default="figure.json",
        help="Path to Plotly JSON (default: figure.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="plot.html",
        help="Output HTML path (default: plot.html)",
    )
    parser.add_argument(
        "--include-plotlyjs",
        default="cdn",
        choices=["cdn", "inline", "directory", "none"],
        help="How to include plotly.js (default: cdn; use 'none' if you load it yourself).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Alias for --include-plotlyjs inline (fully self-contained HTML).",
    )
    parser.add_argument(
        "--no-full-html",
        action="store_true",
        help="Write an HTML fragment instead of a full document.",
    )
    return parser.parse_args()


def _load_plotly_figure_json(input_path: Path) -> tuple[str, dict]:
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {input_path}, got {type(raw)}")

    figure_payload = raw.get("figure", raw)
    if not isinstance(figure_payload, dict):
        raise ValueError(
            f"Expected a Plotly figure object in {input_path} (keys: data/layout), got {type(figure_payload)}"
        )

    config = raw.get("config") or raw.get("figure", {}).get("config") or {}
    return json.dumps(figure_payload), (config if isinstance(config, dict) else {})


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    include_plotlyjs = "inline" if args.offline else args.include_plotlyjs
    if include_plotlyjs == "none":
        include_plotlyjs = False
    figure_json, config = _load_plotly_figure_json(input_path)
    fig = pio.from_json(figure_json)

    fig.write_html(
        str(output_path),
        include_plotlyjs=include_plotlyjs,
        full_html=not args.no_full_html,
        config=(config or None),
    )


if __name__ == "__main__":
    main()
