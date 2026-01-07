# iPlotBench Prompts (Canonical)

These prompts define the benchmark interface for iPlotBench.

## Task 1: Recreation

```
Read ./input.png and recreate this plot.

Output the Plotly figure as JSON with "data" and "layout" keys:
{"data": [...], "layout": {...}}
```

## Task 2: QA (single question)

Template (replace `{question}` with `question_string`):

```
{question}

Reply with ONLY a single digit: 0 or 1
```

