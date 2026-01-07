# Running Your Own Models

Evaluate your vision-language model on iPlotBench using any OpenAI-compatible API.

## Quick Start

```bash
# 1. Install iPlotBench
pip install -r requirements.txt

# 2. Prepare evaluation environments
python -m eval.prepare_envs --output ./env

# 3. Start your model server (example: vLLM)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8666

# 4. Run evaluation
python -m eval.runners.run my_model --config vision

# 5. Compute metrics
python -m eval.run_eval my_model
```

## Configurations

| Config | Introspection | Interaction | Description |
|--------|---------------|-------------|-------------|
| `vision` | - | - | Baseline (no tools) |
| `vision_interactive` | - | ✓ | View manipulation only |
| `vision_introspect` | ✓ | - | Spec access only |
| `vision_introspect_interactive` | ✓ | ✓ | Full IVG (all tools) |

**Tools:**
- **Introspection**: `get_plot_json` - read chart specification
- **Interaction**: `relayout`, `legendclick`, `selected`, `query_interactions` - view manipulation

## Usage

```bash
# Baseline (no tools)
python -m eval.runners.run my_model --config vision

# With introspection only
python -m eval.runners.run my_model --config vision_introspect

# With interaction only
python -m eval.runners.run my_model --config vision_interactive

# Full IVG (all tools)
python -m eval.runners.run my_model --config vision_introspect_interactive

# With custom endpoint
python -m eval.runners.run MODEL_NAME --config vision \
    --model MODEL_ID \
    --api-base http://HOST:PORT/v1

# Run specific figure types
python -m eval.runners.run MODEL_NAME --config vision \
    --types vbar_categorical pie

# Resume interrupted run
python -m eval.runners.run MODEL_NAME --config vision --resume
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | required | One of: `vision`, `vision_introspect`, `vision_interactive`, `vision_introspect_interactive` |
| `--api-base` | `http://127.0.0.1:8666/v1` | API endpoint |
| `--model` | `Qwen/Qwen3-VL-4B-Instruct` | Model ID |
| `--types` | all | Figure types to process |
| `--figures` | all | Specific figure IDs |
| `--data-dir` | `env` | Input data directory |
| `--output-dir` | `localenv/output/{model}` | Output directory |
| `--resume` | false | Skip completed figures |

## Output Structure

```
localenv/output/{model_name}/
├── {figure_id}/
│   ├── output_{config}_task1.json    # Recreated figure
│   ├── output_{config}_task2_q0.json # QA answer 0
│   ├── output_{config}_task2_q1.json # QA answer 1
│   └── ...
└── logs/                              # Session logs
```

## Supported Models

Any VLM with OpenAI-compatible API:

```bash
# Qwen series
--model Qwen/Qwen2.5-VL-7B-Instruct
--model Qwen/Qwen3-VL-4B-Instruct

# Mistral Pixtral
--model mistralai/Pixtral-12B-2409

# LLaVA
--model llava-hf/llava-v1.6-mistral-7b-hf

# InternVL
--model OpenGVLab/InternVL2-8B
```

**Note**: `vision_introspect_interactive` requires models with function calling support.
