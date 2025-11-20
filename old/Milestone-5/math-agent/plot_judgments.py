#!/usr/bin/env python3
"""
Produce comparison plots from multiple judgment JSONL files.

This script creates:
- Overlaid score density comparison
- Per-metric boxplots by model
- Correct-answer percentage stacked bars
- Scatter of two metrics (if present)
- Grouped mean bars (optionally excluding `score`)

Usage:
  python plot_judgments.py --files a.jsonl b.jsonl --out-dir plots
"""

import argparse
import json
from pathlib import Path

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    raise SystemExit("This script requires pandas, matplotlib, seaborn.")


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def read_judgments(path: Path):
    """Read JSONL file and extract judgment rows."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if 'judgment' in obj and isinstance(obj['judgment'], dict):
                row = obj['judgment'].copy()
            else:
                row = obj.copy()

            row['_line'] = idx
            rows.append(row)
    return rows


def safe_float(x):
    try:
        if x is None:
            return float("nan")
        if isinstance(x, (int, float)):
            return float(x)
        return float(x)
    except Exception:
        return float("nan")


def ensure_outdir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# Comparison Plots
# ---------------------------------------------------------------

def plot_comparison(paths, out_dir: Path, exclude_score_from_mean=True):
    """Generate comparison plots across multiple models."""

    dfs = []
    for p in paths:
        rows = read_judgments(p)
        if not rows:
            continue
        df = pd.DataFrame(rows)

        if "did_it_stop_midway" in df.columns:
            df = df.drop(columns=["did_it_stop_midway"])

        df["__model"] = p.stem
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid rows found in input files.")

    big = pd.concat(dfs, ignore_index=True, sort=False)

    # Numeric fields for comparison
    numeric_candidates = []
    for col in ["score", "did_it_solve_in_easy_and_fast_approach", "easy_to_understand_explanation"]:
        if col in big.columns:
            if big[col].dtype == bool:
                big[col] = big[col].astype(int)
            big[col] = big[col].apply(safe_float)
            numeric_candidates.append(col)

    sns.set_theme(style='whitegrid', palette='Set2', 
                  rc={'figure.dpi': 140, 'font.size': 12})

    out_files = []
    model_col = "__model"

    # -----------------------------------------------------------
    # 1️⃣ Overlaid score density
    # -----------------------------------------------------------
    if "score" in big.columns and not big["score"].dropna().empty:
        plt.figure(figsize=(10, 6))
        bins = min(30, max(10, int(len(big["score"].dropna()) / 5)))

        for model, sub in big.groupby(model_col):
            sns.histplot(sub["score"].dropna(), bins=bins,
                         stat='density', element='step',
                         fill=True, alpha=0.35, label=str(model))

        plt.legend(title='model', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.xlabel("score")
        plt.ylabel("density")
        plt.title("Score distribution — comparison")
        plt.tight_layout()

        out = out_dir / "compare.score_overlay.png"
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        out_files.append(str(out))

    # -----------------------------------------------------------
    # 2️⃣ Per-metric boxplots (except score)
    # -----------------------------------------------------------
    metrics = [c for c in numeric_candidates if c != "score"]
    if metrics:
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))

        if len(metrics) == 1:
            axes = [axes]

        for ax, col in zip(axes, metrics):
            sns.boxplot(x=model_col, y=col, data=big, ax=ax, palette='Set2', showfliers=False)
            sns.stripplot(x=model_col, y=col, data=big, ax=ax,
                          color='k', size=3, alpha=0.4, jitter=0.15)

            means = big.groupby(model_col)[col].mean()
            for j, m in enumerate(means.values):
                ax.scatter(j, m, color='white', edgecolor='black', s=80, zorder=10)
                ax.text(j, m, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

            ax.set_title(col.replace("_", " "))
            ax.set_xlabel("")
            ax.set_ylabel("")

        fig.suptitle("Ratings by model (boxplots)")
        plt.tight_layout()

        out = out_dir / "compare.rating_boxplots.png"
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        out_files.append(str(out))

    # -----------------------------------------------------------
    # 3️⃣ Correct-answer stacked bar %
    # -----------------------------------------------------------
    if "correct_answer" in big.columns:
        pivot = big.groupby([model_col, "correct_answer"]).size().unstack(fill_value=0)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

        ax = pivot_pct.plot(kind="bar", stacked=True, figsize=(10, 6), colormap='Set2')
        ax.set_ylabel("percentage (%)")
        ax.set_xlabel("model")
        # ensure x-axis labels are horizontal for readability
        ax.tick_params(axis='x', rotation=0)
        plt.title("Correct answer distribution by model (percent)")

        for p in ax.patches:
            h = p.get_height()
            if h > 2:
                ax.text(p.get_x() + p.get_width() / 2.,
                        p.get_y() + h / 2., f"{h:.0f}%",
                        ha='center', va='center', fontsize=9)

        plt.tight_layout()
        out = out_dir / "compare.correct_by_model.png"
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        out_files.append(str(out))

    # -----------------------------------------------------------
    # 4️⃣ Scatter of two metrics
    # -----------------------------------------------------------
    scatter_cols = ["did_it_solve_in_easy_and_fast_approach", "easy_to_understand_explanation"]
    if all(col in big.columns for col in scatter_cols):
        a, b = scatter_cols
        plt.figure(figsize=(9, 7))

        sns.scatterplot(data=big, x=a, y=b, hue=model_col, alpha=0.75, s=80)

        plt.xlabel(a.replace("_", " "))
        plt.ylabel(b.replace("_", " "))
        plt.title(f"Comparison: {a.replace('_', ' ')} vs {b.replace('_', ' ')}")

        plt.legend(title="model", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()

        out = out_dir / f"compare.{a}_vs_{b}.png"
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        out_files.append(str(out))

    # -----------------------------------------------------------
    # 5️⃣ Mean summary bar chart
    # -----------------------------------------------------------
    mean_cols = [c for c in numeric_candidates
                 if not (exclude_score_from_mean and c == "score")]

    if mean_cols:
        means = big.groupby(model_col)[mean_cols].mean().transpose()

        plt.figure(figsize=(max(8, len(mean_cols) * 2), 6))
        ax = means.plot(kind="bar", rot=0, colormap="Set2")

        plt.ylabel("mean")
        plt.title("Mean values per metric by model")

        for p in ax.patches:
            h = p.get_height()
            if pd.notna(h):
                ax.text(p.get_x() + p.get_width() / 2., h,
                        f"{h:.2f}", ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        out = out_dir / "compare.mean_ratings_by_model.png"
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        out_files.append(str(out))

    return out_files


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Produce comparison plots from judgment JSONL files")
    parser.add_argument("--files", nargs="+", required=True, help="Input JSONL files")
    parser.add_argument("--out-dir", default=None, help="Directory to save images (default: scripts/plots/)")
    parser.add_argument("--include-score-in-mean", dest="include_score", action="store_true",
                        help="Include score in mean summary plot")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_dir = script_dir / "plots"
    out_dir = Path(args.out_dir) if args.out_dir else default_dir
    ensure_outdir(out_dir)

    files = [Path(f) for f in args.files if Path(f).exists()]
    if not files:
        raise SystemExit("No valid input files.")

    outs = plot_comparison(files, out_dir, exclude_score_from_mean=not args.include_score)

    print("\nSaved plots:")
    for o in outs:
        print(" -", o)


if __name__ == "__main__":
    main()
