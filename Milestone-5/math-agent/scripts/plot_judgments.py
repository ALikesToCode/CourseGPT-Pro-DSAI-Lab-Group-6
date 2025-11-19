#!/usr/bin/env python3
"""Read one or more judgment JSONL files and produce diagnostic plots.

Saves PNGs to an output directory (default: `plots/` next to the script's folder).

Plots produced per input file:
- score histogram (if `score` present and numeric)
- correct_answer counts
- boxplots for numeric rating fields (did_it_solve_in_easy_and_fast_approach, easy_to_understand_explanation, etc.)
- scatter of two chosen metrics if present

Usage:
  python plot_judgments.py --files path/to/a.jsonl path/to/b.jsonl --out-dir path/to/plots
"""
import argparse
import json
import os
from pathlib import Path
import math

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    raise SystemExit("This script requires pandas, matplotlib and seaborn. Install them in your environment.")


def read_judgments(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # skip invalid lines
                continue
            # some files wrap the judgment under `judgment`
            if 'judgment' in obj and isinstance(obj['judgment'], dict):
                row = obj['judgment'].copy()
            else:
                # fallback: use top-level fields
                row = obj.copy()
            # keep original index for traceability
            row['_line'] = i
            rows.append(row)
    return rows


def safe_float(x):
    try:
        if x is None:
            return float('nan')
        if isinstance(x, (int, float)):
            return float(x)
        return float(x)
    except Exception:
        return float('nan')


def ensure_outdir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_for_file(path: Path, out_dir: Path):
    rows = read_judgments(path)
    if not rows:
        print(f"No valid rows found in {path}")
        return []
    df = pd.DataFrame(rows)

    # Normalize some common fields
    numeric_fields = []
    for col in ['score', 'did_it_solve_in_easy_and_fast_approach', 'easy_to_understand_explanation', 'did_it_stop_midway']:
        if col in df.columns:
            # convert booleans to numeric if needed
            if df[col].dtype == 'bool':
                df[col] = df[col].astype(int)
            df[col] = df[col].apply(safe_float)
            numeric_fields.append(col)

    model_name = path.stem
    out_files = []

    sns.set(style='whitegrid')

    # 1) score histogram
    if 'score' in df.columns and not df['score'].dropna().empty:
        plt.figure(figsize=(6,4))
        sns.histplot(df['score'].dropna(), bins=20, kde=False)
        plt.title(f'{model_name} — score distribution')
        plt.xlabel('score')
        plt.tight_layout()
        out = out_dir / f'{model_name}.score_hist.png'
        plt.savefig(out)
        plt.close()
        out_files.append(str(out))

    # 2) correct_answer counts (if present)
    if 'correct_answer' in df.columns:
        plt.figure(figsize=(5,4))
        counts = df['correct_answer'].value_counts(dropna=False)
        sns.barplot(x=counts.index.astype(str), y=counts.values)
        plt.title(f'{model_name} — correct_answer counts')
        plt.xlabel('correct_answer')
        plt.ylabel('count')
        plt.tight_layout()
        out = out_dir / f'{model_name}.correct_counts.png'
        plt.savefig(out)
        plt.close()
        out_files.append(str(out))

    # 3) boxplots for numeric ratings
    numeric_cols = [c for c in numeric_fields if c in df.columns]
    if numeric_cols:
        plt.figure(figsize=(8,4 + 0.4*len(numeric_cols)))
        to_plot = df[numeric_cols].melt(var_name='metric', value_name='value')
        sns.boxplot(x='metric', y='value', data=to_plot)
        plt.title(f'{model_name} — rating boxplots')
        plt.xticks(rotation=20)
        plt.tight_layout()
        out = out_dir / f'{model_name}.rating_boxplots.png'
        plt.savefig(out)
        plt.close()
        out_files.append(str(out))

    # 4) scatter of two metrics if both present
    scatter_candidates = ['did_it_solve_in_easy_and_fast_approach', 'easy_to_understand_explanation']
    if all(c in df.columns for c in scatter_candidates):
        a, b = scatter_candidates
        plt.figure(figsize=(6,6))
        sns.scatterplot(x=df[a].apply(safe_float), y=df[b].apply(safe_float))
        plt.xlabel(a)
        plt.ylabel(b)
        plt.title(f'{model_name} — {a} vs {b}')
        plt.tight_layout()
        out = out_dir / f'{model_name}.{a}_vs_{b}.png'
        plt.savefig(out)
        plt.close()
        out_files.append(str(out))

    # 5) additional diagnostics: missing answer rate (if `score` in df.columns and `correct_answer` in df.columns)
    if 'score' in df.columns and 'correct_answer' in df.columns:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df['correct_answer'].astype(str), y=df['score'].apply(safe_float))
        plt.title(f'{model_name} — score by correctness')
        plt.xlabel('correct_answer')
        plt.ylabel('score')
        plt.tight_layout()
        out = out_dir / f'{model_name}.score_by_correct.png'
        plt.savefig(out)
        plt.close()
        out_files.append(str(out))

    print(f"Generated {len(out_files)} plot(s) for {path.name}")
    return out_files


def plot_comparison(paths, out_dir: Path):
    """Create comparison plots for multiple judgment files.

    Returns list of output file paths.
    """
    dfs = []
    for p in paths:
        rows = read_judgments(p)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df['__model'] = p.stem
        dfs.append(df)
    if not dfs:
        raise ValueError('No valid rows found in input files')

    big = pd.concat(dfs, ignore_index=True, sort=False)

    # normalize numeric columns
    numeric_candidates = ['score', 'did_it_solve_in_easy_and_fast_approach', 'easy_to_understand_explanation', 'did_it_stop_midway']
    for col in numeric_candidates:
        if col in big.columns:
            if big[col].dtype == 'bool':
                big[col] = big[col].astype(int)
            big[col] = big[col].apply(safe_float)

    out_files = []
    model_col = '__model'

    sns.set(style='whitegrid')

    # 1) Overlaid score histogram (hue=model)
    if 'score' in big.columns and not big['score'].dropna().empty:
        plt.figure(figsize=(8,5))
        sns.histplot(data=big, x='score', hue=model_col, bins=20, element='step', stat='count', common_norm=False)
        plt.title('Score distribution — comparison')
        plt.tight_layout()
        out = out_dir / 'compare.score_overlay.png'
        plt.savefig(out)
        plt.close()
        out_files.append(str(out))

    # 2) Boxplots per model for numeric ratings
    numeric_cols = [c for c in numeric_candidates if c in big.columns]
    if numeric_cols:
        # melt to get model + metric + value
        melt = big[[model_col] + numeric_cols].melt(id_vars=[model_col], var_name='metric', value_name='value')
        plt.figure(figsize=(10, 4 + 0.6*len(numeric_cols)))
        sns.boxplot(x='metric', y='value', hue=model_col, data=melt)
        plt.title('Ratings by metric — comparison')
        plt.legend(title='model', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        out = out_dir / 'compare.rating_boxplots.png'
        plt.savefig(out)
        plt.close()
        out_files.append(str(out))

    # 3) Correct_answer proportion per model
    if 'correct_answer' in big.columns:
        pivot = big.groupby([model_col, 'correct_answer']).size().unstack(fill_value=0)
        pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)
        plt.figure(figsize=(8,4))
        pivot_norm.plot(kind='bar', stacked=False)
        plt.title('Correct answer proportions by model')
        plt.xlabel('model')
        plt.ylabel('proportion')
        plt.tight_layout()
        out = out_dir / 'compare.correct_by_model.png'
        plt.savefig(out)
        plt.close()
        out_files.append(str(out))

    # 4) Scatter of two metrics colored by model (if both present)
    scatter_candidates = ['did_it_solve_in_easy_and_fast_approach', 'easy_to_understand_explanation']
    if all(c in big.columns for c in scatter_candidates):
        a, b = scatter_candidates
        plt.figure(figsize=(7,7))
        sns.scatterplot(data=big, x=a, y=b, hue=model_col, alpha=0.7)
        plt.title(f'Comparison: {a} vs {b}')
        plt.tight_layout()
        out = out_dir / f'compare.{a}_vs_{b}.png'
        plt.savefig(out)
        plt.close()
        out_files.append(str(out))

    print(f'Generated {len(out_files)} comparison plot(s)')
    return out_files


def main():
    p = argparse.ArgumentParser(description='Produce plots from judgment JSONL files')
    p.add_argument('--files', nargs='+', required=True, help='One or more JSONL judgment files')
    p.add_argument('--out-dir', default=None, help='Directory to save PNGs (default: scripts/plots/)')
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_out = script_dir / 'plots'
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    ensure_outdir(out_dir)

    all_outputs = []
    input_paths = []
    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"Warning: file not found: {path}")
            continue
        input_paths.append(path)
        outs = plot_for_file(path, out_dir)
        all_outputs.extend(outs)

    # If multiple files provided, produce comparison plots
    if len(input_paths) > 1:
        try:
            comp_outs = plot_comparison(input_paths, out_dir)
            all_outputs.extend(comp_outs)
        except Exception as e:
            print('Comparison plots failed:', e)

    if all_outputs:
        print('\nSaved plots:')
        for o in all_outputs:
            print(' -', o)
    else:
        print('No plots were generated.')


if __name__ == '__main__':
    main()
