# iPlotBench

**Interactive Plot Benchmark** for evaluating visualization agents with spec-grounded introspection and view-grounded interaction.

## Overview

iPlotBench provides **500 interactive Plotly figures** with **6,706 binary questions** for evaluating vision-language models on chart understanding:

| Task | Description | Evaluation |
|------|-------------|------------|
| **Task 1: Chart Recreation** | Recreate chart from reference image | SSS metrics (S_Type, S_Data, S_Text, S_Style) |
| **Task 2: Visual QA** | Answer yes/no questions about the chart | Accuracy |

**Evaluation Protocol:** Task 2 questions are asked **after Task 1** in the **same agent session**, so the agent reasons over its own recreated chart.

## Installation

```bash
git clone https://github.com/YiyangLu/iPlotBench.git
cd iPlotBench
pip install -r requirements.txt

# Generate dataset (or download from artifact)
./scripts/generate_v1.sh
```

## Running Evaluation

Choose **one** of the following methods:

### Option A: Script-based (Recommended for API models)

For models with OpenAI-compatible API (vLLM, OpenRouter, etc.):

```bash
# 1. Prepare environments
python -m eval.prepare_envs --output ./env

# 2. Run evaluation
python -m eval.runners.run my_model --config vision

# 3. Compute metrics
python -m eval.run_eval my_model
```

See **[eval/runners/README.md](eval/runners/README.md)** for detailed options.

### Option B: Docker-based (Recommended for coding agents)

For agents that need sandboxed execution (e.g., Claude Code). See **[docker/README.md](docker/README.md)** for detailed instructions.

## Key Features

- **Configurable size**: Generate 500 (default), 1000, or any number of figures
- **5 figure types**: 100 figures per type (at default 500)
- **~13 QA pairs per figure**: 15 question types with balanced yes/no answers (50/50)
- **Ground truth available**: Source data enables deterministic Plotly conversion
- **Interactive Plotly**: All figures support zoom, filter, select
- **Structured access**: Agents can inspect chart JSON structure
- **FigureQA compatible**: Question generation ported from original FigureQA code

## Directory Structure

```
iPlotBench/
├── v1/test/                      # Ground-truth dataset
│   └── {figure_id}/
│       ├── fig.json              # Plotly figure JSON (ground truth)
│       ├── fig.png               # Rendered image
│       ├── metadata.json         # Figure type, image_index
│       └── qa_pairs.json         # QA pairs for this figure
├── env/                          # Evaluation sandbox (generated)
│   ├── test/{figure_id}/input.png
│   └── query/{figure_id}/questions.json
├── scripts/                      # Dataset generation
│   ├── generate_v1.sh
│   ├── generate_source_data.py
│   └── convert_figureqa.py
├── eval/                         # Evaluation code
│   ├── run_eval.py               # Compute SSS metrics
│   ├── prepare_envs.py           # Create eval sandbox
│   ├── sss/                      # SSS metric implementation
│   ├── runners/                  # Run your own models
│   │   └── README.md             # Guide for API-based models
│   └── validator.py              # Output validators
├── docker/
│   ├── Dockerfile                # Base image
│   └── README.md                 # Guide for Docker-based agents
└── README.md
```

## Figure Types

| Type | Plotly Equivalent | Count (default 500) |
|------|-------------------|---------------------|
| `vbar_categorical` | `go.Bar(orientation='v')` | 100 |
| `hbar_categorical` | `go.Bar(orientation='h')` | 100 |
| `pie` | `go.Pie()` | 100 |
| `line` | `go.Scatter(mode='lines')` | 100 |
| `dot_line` | `go.Scatter(mode='lines+markers')` | 100 |
| **Total** | | **500** |

## Question Statistics

| Figure Type | Questions | Avg/Figure |
|-------------|-----------|------------|
| Vertical Bar | ~1,142 | 11.4 |
| Horizontal Bar | ~1,187 | 11.9 |
| Pie | ~1,183 | 11.8 |
| Line | ~1,504 | 15.0 |
| Dot-Line | ~1,690 | 16.9 |
| **Total** | **6,706** | **13.4** |

## Question Types

**Bar/Pie Charts (6 types):**

| ID | Type | Example |
|----|------|---------|
| 0 | is_minimum | "Is Blue Violet the minimum?" |
| 1 | is_maximum | "Is Blue Violet the maximum?" |
| 2 | less_than | "Is Blue Violet less than Teal?" |
| 3 | greater_than | "Is Blue Violet greater than Teal?" |
| 4 | low_median | "Is Teal the low median?" |
| 5 | high_median | "Is Teal the high median?" |

**Line/Dot-Line Charts (9 types):**

| ID | Type | Example |
|----|------|---------|
| 6 | auc_minimum | "Does Blue have the minimum area under the curve?" |
| 7 | auc_maximum | "Does Red have the maximum area under the curve?" |
| 8 | smoothest | "Is Green the smoothest?" |
| 9 | roughest | "Is Yellow the roughest?" |
| 10 | lowest_value | "Does Blue have the lowest value?" |
| 11 | highest_value | "Does Red have the highest value?" |
| 12 | less_than_strict | "Is Blue less than Red?" (at all points) |
| 13 | greater_than_strict | "Is Red greater than Blue?" (at all points) |
| 14 | intersect | "Does Blue intersect Red?" |

**Answer Balance:** 50% Yes, 50% No


## Data Generation

```bash
# Generate 500 figures (default)
./scripts/generate_v1.sh

# Generate custom size (e.g., 1000 figures)
./scripts/generate_v1.sh 1000

# Skip PNG generation (faster)
./scripts/generate_v1.sh --no-png
```

**Note:** Total is rounded to nearest multiple of 5 (equal figures per type).
For example, `./scripts/generate_v1.sh 999` generates 995 figures (199 × 5 types).

## File Formats

### fig.json (Plotly format)
```json
{
  "data": [{
    "type": "bar",
    "x": ["Blue Violet", "Teal"],
    "y": [24.2, 13.8],
    "marker": {"color": ["#8A2BE2", "#008080"]}
  }],
  "layout": {...}
}
```

### qa_pairs.json
```json
[
  {"question": "Is Blue Violet the maximum?", "answer": 1},
  {"question": "Is Teal the minimum?", "answer": 0}
]
```

### metadata.json
```json
{
  "figure_id": "pie_0042",
  "image_index": 242,
  "figure_type": "pie",
  "num_qa_pairs": 12
}
```
Note: `image_index` is the global index from source data generation (used internally).


## Output Validation

You can validate agent outputs using the provided validator:

```bash
# Validate single files
python -m eval.validator task1 output.json
python -m eval.validator task2 answer.json

# Check all outputs in a directory
python -m eval.validator check-dir env/output/my_agent
```

Or in Python:
```python
from eval.validator import validate_task1, validate_task2

result = validate_task1(output_dict)
if not result.valid:
    print(f"Invalid: {result.error}")
```