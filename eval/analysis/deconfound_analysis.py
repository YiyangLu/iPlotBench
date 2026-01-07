#!/usr/bin/env python3
"""
Deconfounding analysis for iPlotBench paper.
Computes conditional QA accuracy and question family breakdowns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

# Paths
RESULTS_DIR = Path("/home/ylu21/proj/iPlotBench/eval/eval_results/haiku_20260104_140607")
CONFIGS = ["vision", "vision_interactive", "vision_lint", "vision_lint_interactive"]
CONFIG_LABELS = {
    "vision": "Vision",
    "vision_interactive": "+Inter",
    "vision_lint": "+Intro",
    "vision_lint_interactive": "Full"
}

# Question family mapping by question_id
# Aggregation: min/max/median
AGGREGATION = {0, 1, 10, 11, 12, 13}
# Comparison: less/greater
COMPARISON = {2, 3, 4, 5}
# Topology: AUC min/max, smoothest/roughest, intersect
TOPOLOGY = {6, 7, 8, 9, 14}

def get_family(qid):
    if qid in AGGREGATION:
        return "Aggregation"
    if qid in COMPARISON:
        return "Comparison"
    if qid in TOPOLOGY:
        return "Topology"
    return "Other"

def load_data():
    """Load and merge Task 1 and Task 2 data."""
    # Load Task 1 (SSS scores)
    t1_dfs = []
    for config in CONFIGS:
        df = pd.read_csv(RESULTS_DIR / f"task1_{config}.csv")
        t1_dfs.append(df)
    t1 = pd.concat(t1_dfs, ignore_index=True)

    # Load Task 2 (QA results)
    t2_dfs = []
    for config in CONFIGS:
        df = pd.read_csv(RESULTS_DIR / f"task2_{config}.csv")
        t2_dfs.append(df)
    t2 = pd.concat(t2_dfs, ignore_index=True)

    # Filter valid predictions
    t2 = t2[t2['pred_answer'].notna()]
    if 'error' in t2.columns:
        t2 = t2[t2['error'].isna() | (t2['error'] == "")]

    # Add question family
    t2['family'] = t2['question_id'].apply(get_family)

    # Merge T1 scores into T2
    df = t2.merge(
        t1[['figure_id', 'config', 's_type', 's_data', 's_text', 's_style']],
        on=['figure_id', 'config'],
        how='left'
    )

    return t1, t2, df

def compute_conditional_accuracy(df, threshold=0.95):
    """Compute QA accuracy conditioned on S_Data >= threshold."""
    # Per-figure accuracy (to avoid question-count weighting)
    fig_acc = df.groupby(['figure_id', 'config']).agg({
        'correct': 'mean',
        's_data': 'first'
    }).reset_index()

    # Overall accuracy
    overall = fig_acc.groupby('config')['correct'].mean()

    # Conditional accuracy
    cond_fig = fig_acc[fig_acc['s_data'] >= threshold]
    cond = cond_fig.groupby('config')['correct'].mean()

    n_total = fig_acc.groupby('config')['figure_id'].count()
    n_cond = cond_fig.groupby('config')['figure_id'].count()

    return overall, cond, n_total, n_cond

def compute_family_breakdown(df, conditional=False, threshold=0.95):
    """Compute accuracy by question family."""
    if conditional:
        df = df[df['s_data'] >= threshold]

    return df.groupby(['config', 'family'])['correct'].mean().unstack()

def compute_correlation(df):
    """Compute Spearman correlation between S_Data and QA accuracy per figure."""
    fig_acc = df.groupby(['config', 'figure_id']).agg({
        'correct': 'mean',
        's_data': 'first'
    }).reset_index()

    correlations = {}
    for config in CONFIGS:
        c_df = fig_acc[fig_acc['config'] == config]
        if len(c_df) > 2:
            corr, pval = spearmanr(c_df['s_data'], c_df['correct'])
            correlations[config] = (corr, pval)
    return correlations

def main():
    print("Loading data...")
    t1, t2, df = load_data()

    print(f"\nTotal Task 1 records: {len(t1)}")
    print(f"Total Task 2 records: {len(t2)}")
    print(f"Merged records: {len(df)}")

    # 1. Conditional Accuracy Table (for paper Table 3)
    print("\n" + "="*60)
    print("TABLE 3: Conditional QA Accuracy (S_Data >= 0.9)")
    print("="*60)

    overall, cond, n_total, n_cond = compute_conditional_accuracy(df, threshold=0.9)

    print("\nConfig           | Overall Acc | Cond. Acc | Figures (cond/total)")
    print("-" * 65)
    for config in CONFIGS:
        label = CONFIG_LABELS[config]
        o = overall.get(config, 0)
        c = cond.get(config, 0)
        nt = n_total.get(config, 0)
        nc = n_cond.get(config, 0)
        print(f"{label:16} | {o:.4f}      | {c:.4f}    | {nc}/{nt}")

    # LaTeX format
    print("\n% LaTeX table row format:")
    for config in CONFIGS:
        label = CONFIG_LABELS[config]
        o = overall.get(config, 0)
        c = cond.get(config, 0)
        print(f"{label} & {o:.4f} & {c:.4f} \\\\")

    # 2. Question Family Breakdown
    print("\n" + "="*60)
    print("QUESTION FAMILY BREAKDOWN (Overall)")
    print("="*60)

    family_overall = compute_family_breakdown(df, conditional=False)
    print(family_overall.round(4))

    print("\n" + "="*60)
    print("QUESTION FAMILY BREAKDOWN (Conditional S_Data >= 0.9)")
    print("="*60)

    family_cond = compute_family_breakdown(df, conditional=True, threshold=0.9)
    print(family_cond.round(4))

    # 3. Correlation
    print("\n" + "="*60)
    print("CORRELATION: S_Data vs QA Accuracy (Spearman)")
    print("="*60)

    correlations = compute_correlation(df)
    for config in CONFIGS:
        label = CONFIG_LABELS[config]
        if config in correlations:
            corr, pval = correlations[config]
            print(f"{label:16}: r={corr:.4f}, p={pval:.4e}")

    # 4. S_Data distribution
    print("\n" + "="*60)
    print("S_Data DISTRIBUTION")
    print("="*60)

    for config in CONFIGS:
        label = CONFIG_LABELS[config]
        s_data = t1[t1['config'] == config]['s_data']
        n_high = (s_data >= 0.9).sum()
        print(f"{label:16}: mean={s_data.mean():.4f}, >=0.9: {n_high}/{len(s_data)} ({100*n_high/len(s_data):.1f}%)")

if __name__ == "__main__":
    main()
