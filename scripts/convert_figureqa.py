#!/usr/bin/env python
"""Convert source data to Plotly figures.

Reads: data/annotations.json, data/qa_pairs.json
Outputs: v1/{figure_id}/
    - fig.json (Plotly figure)
    - fig.png (rendered image)
    - metadata.json (figure info)
    - qa_pairs.json (QA pairs for this figure)

Usage:
    python scripts/convert_figureqa.py [--limit N] [--no-png]
"""
import argparse
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "v1"


def convert_line_to_plotly(annotation: dict, mode: str = "lines") -> go.Figure:
    """Convert line/dot_line figure to Plotly."""
    fig = go.Figure()

    for model in annotation.get("models", []):
        fig.add_trace(go.Scatter(
            x=model.get("x", []),
            y=model.get("y", []),
            mode=mode,
            name=model.get("name", model.get("label", "")),
            line=dict(color=model.get("color")),
            marker=dict(color=model.get("color")) if "markers" in mode else None,
        ))

    # Add layout
    layout_info = annotation.get("general_figure_info", {})
    fig.update_layout(
        title=layout_info.get("title", {}).get("text", ""),
        xaxis_title=layout_info.get("x_axis", {}).get("label", {}).get("text", ""),
        yaxis_title=layout_info.get("y_axis", {}).get("label", {}).get("text", ""),
        showlegend=True,
    )

    return fig


def convert_bar_to_plotly(annotation: dict, orientation: str = "v") -> go.Figure:
    """Convert bar chart to Plotly."""
    models = annotation.get("models", [])
    if not models:
        return go.Figure()

    model = models[0]  # Bar charts have single model with all bars

    if orientation == "v":
        fig = go.Figure(go.Bar(
            x=model.get("labels", model.get("x", [])),
            y=model.get("y", []),
            marker_color=model.get("colors", []),
        ))
    else:  # horizontal
        fig = go.Figure(go.Bar(
            y=model.get("labels", model.get("y", [])),
            x=model.get("x", []),
            orientation="h",
            marker_color=model.get("colors", []),
        ))

    layout_info = annotation.get("general_figure_info", {})
    fig.update_layout(
        title=layout_info.get("title", {}).get("text", ""),
        showlegend=False,
    )

    return fig


def convert_pie_to_plotly(annotation: dict) -> go.Figure:
    """Convert pie chart to Plotly."""
    models = annotation.get("models", [])

    labels = [m.get("label", m.get("name", "")) for m in models]
    values = [m.get("span", 0) for m in models]  # span in radians
    colors = [m.get("color", "") for m in models]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
    ))

    layout_info = annotation.get("general_figure_info", {})
    fig.update_layout(
        title=layout_info.get("title", {}).get("text", ""),
    )

    return fig


def convert_figure(annotation: dict) -> go.Figure:
    """Convert annotation to Plotly figure based on type."""
    figure_type = annotation.get("type", "")

    if figure_type == "line":
        return convert_line_to_plotly(annotation, mode="lines")
    elif figure_type == "dot_line":
        return convert_line_to_plotly(annotation, mode="lines+markers")
    elif figure_type == "vbar_categorical":
        return convert_bar_to_plotly(annotation, orientation="v")
    elif figure_type == "hbar_categorical":
        return convert_bar_to_plotly(annotation, orientation="h")
    elif figure_type == "pie":
        return convert_pie_to_plotly(annotation)
    else:
        raise ValueError(f"Unknown figure type: {figure_type}")


def main(limit: int = None, save_png: bool = True, output_dir: Path = None):
    """Convert all figures to Plotly."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    else:
        output_dir = Path(output_dir)

    # Load annotations
    annotations_file = DATA_DIR / "annotations.json"
    if not annotations_file.exists():
        print(f"Annotations not found: {annotations_file}")
        print("Run generate_source_data.py first.")
        return

    with open(annotations_file) as f:
        annotations = json.load(f)

    # Load QA pairs
    qa_file = DATA_DIR / "qa_pairs.json"
    with open(qa_file) as f:
        qa_data = json.load(f)

    # Build QA lookup by image_index
    qa_by_image = {}
    for qa in qa_data.get("qa_pairs", []):
        img_idx = qa.get("image_index")
        if img_idx not in qa_by_image:
            qa_by_image[img_idx] = []
        qa_by_image[img_idx].append(qa)

    # Apply limit
    if limit:
        annotations = annotations[:limit]

    # Convert each figure
    output_dir.mkdir(parents=True, exist_ok=True)
    counts_by_type = {}  # Track per-type counts for 0-indexed naming
    qa_counts_by_type = {}

    for ann in annotations:
        image_index = ann.get("image_index", 0)
        figure_type = ann.get("type", "unknown")

        # Use per-type 0-indexed naming (e.g., pie_0000, pie_0001, ...)
        type_index = counts_by_type.get(figure_type, 0)
        figure_id = f"{figure_type}_{type_index:04d}"

        # Create figure directory
        figure_dir = output_dir / figure_id
        figure_dir.mkdir(parents=True, exist_ok=True)

        # Convert to Plotly
        try:
            fig = convert_figure(ann)
        except Exception as e:
            print(f"Error converting {figure_id}: {e}")
            continue

        # Save Plotly JSON
        with open(figure_dir / "fig.json", "w") as f:
            f.write(fig.to_json())

        # Save PNG
        if save_png:
            try:
                pio.write_image(fig, figure_dir / "fig.png", width=800, height=600)
            except Exception as e:
                print(f"Warning: Could not save PNG for {figure_id}: {e}")

        # Save metadata
        metadata = {
            "figure_id": figure_id,
            "image_index": image_index,
            "figure_type": figure_type,
            "num_qa_pairs": len(qa_by_image.get(image_index, [])),
        }
        with open(figure_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save QA pairs for this figure
        figure_qa = qa_by_image.get(image_index, [])
        if figure_qa:
            with open(figure_dir / "qa_pairs.json", "w") as f:
                json.dump(figure_qa, f, indent=2)

        # Track counts
        counts_by_type[figure_type] = counts_by_type.get(figure_type, 0) + 1
        qa_counts_by_type[figure_type] = qa_counts_by_type.get(figure_type, 0) + len(figure_qa)

    # Print summary
    total_figures = sum(counts_by_type.values())
    total_qa = sum(qa_counts_by_type.values())

    print("")
    print("=" * 60)
    print("DATASET GENERATED")
    print("=" * 60)
    print("")
    print(f"{'Figure Type':<20} {'Figures':>10} {'QA Pairs':>12} {'Range'}")
    print("-" * 60)

    type_order = ['vbar_categorical', 'hbar_categorical', 'pie', 'line', 'dot_line']
    for fig_type in type_order:
        if fig_type in counts_by_type:
            count = counts_by_type[fig_type]
            qa_count = qa_counts_by_type.get(fig_type, 0)
            range_str = f"{fig_type}_0000 - {fig_type}_{count-1:04d}"
            print(f"{fig_type:<20} {count:>10} {qa_count:>12}   {range_str}")

    print("-" * 60)
    avg_qa = total_qa / total_figures if total_figures > 0 else 0
    print(f"{'TOTAL':<20} {total_figures:>10} {total_qa:>12}   (avg {avg_qa:.1f} QA/fig)")
    print("")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Max figures to convert (default: all)")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip PNG generation (faster)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: v1)")
    args = parser.parse_args()

    main(limit=args.limit, save_png=not args.no_png, output_dir=args.output)
