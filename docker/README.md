# Docker Evaluation

For coding agents that need sandboxed execution. Docker ensures agents cannot access ground truth during testing.

## Quick Start

```bash
# 1. Build base image
docker build -t iplotbench:latest -f docker/Dockerfile .

# 2. Prepare environments
python -m eval.prepare_envs --output ./env

# 3. Build and run your agent
docker run -v $(pwd)/env:/data my-agent:latest

# 4. Compute metrics
python -m eval.run_eval my_agent
```

## Step-by-Step Guide

### 1. Build iPlotBench Base Image

```bash
cd iPlotBench
docker build -t iplotbench:latest -f docker/Dockerfile .
```

### 2. Prepare Environments

```bash
python -m eval.prepare_envs --output ./env
```

This creates:
```
env/
├── test/
│   └── {figure_id}/
│       └── input.png           # Input for both task1 and task2
├── query/
│   └── {figure_id}/
│       └── questions.json      # Questions only (no answers)
└── output/                     # Agent writes results here
```

### 3. Create Your Agent Image

Create a `Dockerfile` in your agent project:

```dockerfile
FROM iplotbench:latest

# Install your dependencies

# Copy your agent code
COPY . /agent
WORKDIR /agent

# Set entry point
ENTRYPOINT ["python", "-m", "your_runner"]
```

Build your image:
```bash
docker build -t my-agent-eval:latest .
```

### 4. Run Evaluation

```bash
docker run \
  -v /path/to/iPlotBench/env:/data \
  my-agent-eval:latest
```

### 5. Evaluate Results

```bash
cd iPlotBench
python -m eval.run_eval my_agent
```

## Input/Output Format

**Input:**
```
/data/test/{figure_id}/
└── input.png          # The visualization to analyze
```

For task2, questions are provided:
```
/data/query/{figure_id}/
└── questions.json     # Questions only (no answers)
```

**questions.json format:**
```json
[
  {"question_id": 0, "question_string": "Is X the minimum?"},
  {"question_id": 1, "question_string": "Is Y greater than Z?"}
]
```

**Output:**

Agent should produce:
```
/data/output/{figure_id}/
├── output_{config}_task1.json      # Recreation task
└── output_{config}_task2_q{n}.json # QA task
```

**Task 1 (Recreation) Output Format:**
```json
{
  "data": [...],
  "layout": {...}
}
```

**Task 2 (QA) Output Format:**
```json
{
  "answer": 0
}
```
Where `answer` is `0` (No) or `1` (Yes).
