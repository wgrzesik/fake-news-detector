import ast
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent / "experiments"
TRAINING_CSV = BASE_DIR / "metrics" / "training_results.csv"
WEB_RESULTS_DIR = BASE_DIR / "web_test_results"
RESULTS_DIR = BASE_DIR / "results"

# The three web text-type result files produced by evaluate_web.py
TEXT_TYPES = ["title", "text", "short_text"]

# Metric columns used in analysis
METRIC_COLS = ["accuracy", "precision", "recall", "f1_score"]

# Key that uniquely identifies a model-experiment (within a single text_type)
MERGE_KEY = ["model", "embedding", "dataset"]


def parse_confusion_matrix(cm_string) -> Optional[np.ndarray]:
    """Parse a confusion-matrix string into a numpy array. Returns None if parsing fails."""
    if isinstance(cm_string, np.ndarray):
        return cm_string
    if not isinstance(cm_string, str):
        return None
    try:
        return np.array(ast.literal_eval(cm_string))
    except (ValueError, SyntaxError):
        return None


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load a CSV with experiment results, clean the data and return a DataFrame.
    """
    df = pd.read_csv(csv_path)

    # Convert metrics to float
    for col in METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse confusion matrices
    if "confusion_matrix" in df.columns:
        df["confusion_matrix"] = df["confusion_matrix"].apply(parse_confusion_matrix)

    # Parse timestamp and remove duplicates (keep newest)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.sort_values("timestamp", ascending=True)
    df = df.drop_duplicates(subset=MERGE_KEY, keep="last")
    df = df.reset_index(drop=True)

    return df


def compare_within(
    df: pd.DataFrame,
    label: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Rank models within a single result set. Returns the sorted ranking DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["model_label"] = df["dataset"] + " + " + df["model"] + " + " + df["embedding"]

    # Sort – primary metric: f1_score, then accuracy
    ranking = (
        df[["model_label", "model", "embedding", "dataset"] + METRIC_COLS]
        .sort_values(["f1_score", "accuracy"], ascending=[False, False])
        .reset_index(drop=True)
    )
    ranking.index += 1
    ranking.index.name = "rank"

    # Save CSV
    csv_path = output_dir / f"ranking_{label.lower().replace(' ', '_')}.csv"
    ranking.to_csv(csv_path)

    # Save TXT
    txt_path = output_dir / f"ranking_{label.lower().replace(' ', '_')}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f" Model Ranking – {label.upper()}\n")
        f.write(f"{'='*60}\n\n")
        f.write(ranking[["model_label"] + METRIC_COLS].to_string())
        f.write("\n")

    # Grouped bar plot
    fig, ax = plt.subplots(figsize=(max(8, len(ranking) * 2.5), 6))

    x = np.arange(len(ranking))
    width = 0.18
    offsets = np.arange(len(METRIC_COLS)) - (len(METRIC_COLS) - 1) / 2

    colors = sns.color_palette("Set2", len(METRIC_COLS))
    for i, metric in enumerate(METRIC_COLS):
        bars = ax.bar(
            x + offsets[i] * width,
            ranking[metric],
            width,
            label=metric,
            color=colors[i],
            edgecolor="white",
        )
        # Value labels on bars
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(ranking["model_label"], rotation=20, ha="right")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Model Comparison – {label}")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    plot_path = output_dir / f"ranking_{label.lower().replace(' ', '_')}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return ranking


def compare_between(
    df_train: pd.DataFrame,
    df_web: pd.DataFrame,
) -> pd.DataFrame:
    """Merge training and web results on MERGE_KEY.

    Computes metric deltas (drops): delta = train - web.
    Sorts by delta_f1_score ascending (best generalization first).

    Returns the merged DataFrame.
    """
    train_cols = MERGE_KEY + METRIC_COLS + ["confusion_matrix"]
    web_cols = MERGE_KEY + METRIC_COLS + ["confusion_matrix"]

    df_t = df_train[[c for c in train_cols if c in df_train.columns]].copy()
    df_w = df_web[[c for c in web_cols if c in df_web.columns]].copy()

    merged = pd.merge(df_t, df_w, on=MERGE_KEY, suffixes=("_train", "_web"))

    # Compute deltas (train - web)
    for metric in METRIC_COLS:
        col_t = f"{metric}_train"
        col_w = f"{metric}_web"
        if col_t in merged.columns and col_w in merged.columns:
            merged[f"delta_{metric}"] = merged[col_t] - merged[col_w]

    # Sort – best generalization (smallest delta_f1_score) first
    if "delta_f1_score" in merged.columns:
        merged = merged.sort_values("delta_f1_score", ascending=True).reset_index(drop=True)

    # Helper column for plots
    merged["model_label"] = merged["dataset"] + " + " + merged["model"] + " + " + merged["embedding"]

    return merged


def plot_train_vs_web_bars(merged: pd.DataFrame, output_dir: Path) -> None:
    """Side-by-side bar plot: accuracy and F1 (training vs web) per model."""
    if merged.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, metric in zip(axes, ["accuracy", "f1_score"]):
        x = np.arange(len(merged))
        width = 0.35

        bars_train = ax.bar(
            x - width / 2,
            merged[f"{metric}_train"],
            width,
            label="Training",
            color=sns.color_palette("Set2")[0],
            edgecolor="white",
        )
        bars_web = ax.bar(
            x + width / 2,
            merged[f"{metric}_web"],
            width,
            label="Web",
            color=sns.color_palette("Set2")[1],
            edgecolor="white",
        )

        # Value labels
        for bar in list(bars_train) + list(bars_web):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(merged["model_label"], rotation=20, ha="right")
        ax.set_ylabel("Metric Value")
        ax.set_title(f"{metric}: Training vs Web")
        ax.set_ylim(0, 1.15)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Performance Comparison: Training vs Web-scraped", fontsize=14, y=1.02)
    fig.tight_layout()

    path = output_dir / "comparison_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_generalization_scatter(merged: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot: accuracy_train vs accuracy_web with y=x reference line.

    Uses per-dataset subplots and color-coded model types for readability.
    """
    if merged.empty:
        return

    datasets = sorted(merged["dataset"].unique())
    models = sorted(merged["model"].unique())

    # Colour palette keyed by model name
    palette = dict(zip(models, sns.color_palette("tab10", len(models))))
    # Marker styles cycled per model
    marker_pool = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "p"]
    markers = {m: marker_pool[i % len(marker_pool)] for i, m in enumerate(models)}

    n_datasets = len(datasets)
    fig, axes = plt.subplots(
        1, n_datasets,
        figsize=(7 * n_datasets, 7),
        squeeze=False,
    )

    for col_idx, ds in enumerate(datasets):
        ax = axes[0, col_idx]
        subset = merged[merged["dataset"] == ds]

        for _, row in subset.iterrows():
            m = row["model"]
            ax.scatter(
                row["accuracy_train"],
                row["accuracy_web"],
                s=140,
                c=[palette[m]],
                marker=markers[m],
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )

        # Stagger annotations so they don't overlap
        texts_placed: list = []
        for _, row in subset.iterrows():
            short_label = f"{row['model']} + {row['embedding']}"
            x_pt, y_pt = row["accuracy_train"], row["accuracy_web"]

            # Choose offset direction to reduce overlap
            x_off, y_off = 10, -10
            for (px, py) in texts_placed:
                if abs(px - x_pt) < 0.02 and abs(py - y_pt) < 0.02:
                    y_off += 14  # shift down if close to another point
            texts_placed.append((x_pt, y_pt))

            ax.annotate(
                short_label,
                (x_pt, y_pt),
                textcoords="offset points",
                xytext=(x_off, y_off),
                fontsize=8,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )

        # Ideal y=x line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "--", color="gray", alpha=0.6, label="Perfect generalization (y=x)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel("Accuracy – Training")
        ax.set_ylabel("Accuracy – Web-scraped")
        ax.set_title(f"Generalization – {ds.upper()}")
        ax.grid(alpha=0.3)

    # Shared legend for model types
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker=markers[m], color="w", markerfacecolor=palette[m],
               markeredgecolor="black", markersize=9, label=m)
        for m in models
    ]
    legend_handles.append(
        Line2D([0], [0], linestyle="--", color="gray", alpha=0.6, label="Perfect generalization (y=x)")
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(len(legend_handles), 6),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Model Generalization: Training vs Web", fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    path = output_dir / "generalization_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_heatmaps(merged: pd.DataFrame, output_dir: Path) -> None:
    """Confusion-matrix heatmaps – training vs web – for each model."""
    cols_needed = ["confusion_matrix_train", "confusion_matrix_web"]
    if not all(c in merged.columns for c in cols_needed):
        return

    models_with_cm = merged.dropna(subset=cols_needed)
    if models_with_cm.empty:
        return

    for _, row in models_with_cm.iterrows():
        label = row["model_label"]
        cm_train = row["confusion_matrix_train"]
        cm_web = row["confusion_matrix_web"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, cm, title in zip(
            axes,
            [cm_train, cm_web],
            [f"{label} – Training", f"{label} – Web"],
        ):
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"],
                cbar=False,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(title)

        fig.suptitle(f"Confusion Matrix: {label}", fontsize=13)
        fig.tight_layout()

        safe_name = label.replace(" ", "").replace("+", "_").strip("_")
        path = output_dir / f"confusion_{safe_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)


def save_f1_comparison(merged: pd.DataFrame, output_dir: Path) -> None:
    """Save a concise TXT with model_label, training_f1, web_f1, delta_f1."""
    needed = ["model_label", "f1_score_train", "f1_score_web", "delta_f1_score"]
    if not all(c in merged.columns for c in needed):
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    comp = (
        merged[needed]
        .rename(columns={
            "f1_score_train": "training_f1",
            "f1_score_web": "web_f1",
            "delta_f1_score": "delta_f1",
        })
        .sort_values("delta_f1", ascending=True)
        .reset_index(drop=True)
    )
    comp.index += 1
    comp.index.name = "rank"

    txt_path = output_dir / "f1_comparison.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 70}\n")
        f.write(" F1 COMPARISON – TRAINING vs WEB\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(comp.to_string())
        f.write("\n")


def save_summary(merged: pd.DataFrame, output_dir: Path) -> None:
    """Save a summary table to CSV and TXT."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Report columns (confusion_matrix excluded – not printable)
    summary_cols = [
        "model_label", "dataset",
        "accuracy_train", "accuracy_web", "delta_accuracy",
        "f1_score_train", "f1_score_web", "delta_f1_score",
        "precision_train", "precision_web", "delta_precision",
        "recall_train", "recall_web", "delta_recall",
    ]
    existing = [c for c in summary_cols if c in merged.columns]
    summary_df = merged[existing].copy()

    csv_path = output_dir / "summary.csv"
    summary_df.to_csv(csv_path, index=False)

    txt_path = output_dir / "summary.txt"
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append(" SUMMARY – TRAINING VS WEB COMPARISON")
    lines.append("=" * 80)
    lines.append("")
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    # Best model by web F1
    if "f1_score_web" in merged.columns and not merged.empty:
        best_web = merged.loc[merged["f1_score_web"].idxmax()]
        lines.append("-" * 80)
        lines.append("* Best model by web F1 score:")
        lines.append(
            f"  {best_web['model_label']}  |  "
            f"F1_web = {best_web['f1_score_web']:.4f}  |  "
            f"Accuracy_web = {best_web['accuracy_web']:.4f}"
        )

    # Best generalization (smallest delta_f1_score)
    if "delta_f1_score" in merged.columns and not merged.empty:
        best_gen = merged.loc[merged["delta_f1_score"].abs().idxmin()]
        lines.append("")
        lines.append("* Best generalization (smallest F1 drop):")
        lines.append(
            f"  {best_gen['model_label']}  |  "
            f"delta_F1 = {best_gen['delta_f1_score']:.4f}  |  "
            f"delta_Accuracy = {best_gen['delta_accuracy']:.4f}"
        )

    lines.append("")
    lines.append("=" * 80)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also generate the focused F1 comparison file
    save_f1_comparison(merged, output_dir)


def run_analysis_pipeline(
    df_train: pd.DataFrame,
    df_web: pd.DataFrame,
    output_dir: Path,
    subset_label: str,
) -> Optional[pd.DataFrame]:
    """Run the full analysis pipeline for a given data subset.
    Generates rankings (within), comparison (between), plots and summary.
    Returns the merged DataFrame (or None when no common models exist).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Within-training ranking
    if not df_train.empty:
        compare_within(df_train, label="training", output_dir=output_dir)

    # Within-web ranking
    if not df_web.empty:
        compare_within(df_web, label="web", output_dir=output_dir)

    # Cross-dataset comparison
    if df_train.empty or df_web.empty:
        return None

    merged = compare_between(df_train, df_web)
    if merged.empty:
        return None

    # Visualizations
    plot_train_vs_web_bars(merged, output_dir)
    plot_generalization_scatter(merged, output_dir)
    plot_confusion_heatmaps(merged, output_dir)

    # Summary
    save_summary(merged, output_dir)

    return merged


def plot_cross_text_type_comparison(
    web_dfs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Bar chart comparing average F1 across text types, grouped by model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for text_type, df in web_dfs.items():
        for _, r in df.iterrows():
            rows.append({
                "model": r["model"],
                "embedding": r["embedding"],
                "dataset": r["dataset"],
                "text_type": text_type,
                "f1_score": r["f1_score"],
                "accuracy": r["accuracy"],
            })

    if not rows:
        return

    cdf = pd.DataFrame(rows)

    # Grouped bar: text_type on x, bars = models, y = mean f1
    fig, ax = plt.subplots(figsize=(max(8, len(cdf["text_type"].unique()) * 4), 6))

    pivot = cdf.pivot_table(
        index="text_type", columns="model", values="f1_score", aggfunc="mean"
    )

    pivot.plot(kind="bar", ax=ax, edgecolor="white")
    ax.set_ylabel("Mean F1 Score")
    ax.set_title("Cross-Text-Type Comparison (mean F1 per model)")
    ax.set_ylim(0, 1.15)
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = output_dir / "cross_text_type_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save CSV summary
    summary = cdf.groupby("text_type").agg({
        "f1_score": ["count", "mean", "max", "min", "std"],
        "accuracy": "mean",
    }).round(4)
    summary.to_csv(output_dir / "cross_text_type_summary.csv")

    # Save TXT
    txt_path = output_dir / "cross_text_type_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(" CROSS-TEXT-TYPE COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        f.write(str(summary))
        f.write("\n\n")
        f.write("Per-model mean F1 by text type:\n")
        f.write(pivot.round(4).to_string())
        f.write("\n")


def main() -> None:
    """Entry point – orchestrates the full analysis pipeline."""

    # 1. Load training results
    print("[1/5] Loading training data...")
    df_train = load_data(TRAINING_CSV)
    print(f"  Training results: {len(df_train)} models")

    # 2. Load per-text-type web results
    print(f"\n[2/5] Loading web results for text types: {TEXT_TYPES}...")
    web_dfs: dict[str, pd.DataFrame] = {}
    for text_type in TEXT_TYPES:
        csv_path = WEB_RESULTS_DIR / f"results_{text_type}.csv"
        if csv_path.exists():
            web_dfs[text_type] = load_data(csv_path)
            print(f"  {text_type:12s}: {len(web_dfs[text_type])} models")
        else:
            print(f"  {text_type:12s}: NOT FOUND ({csv_path})")

    if not web_dfs:
        print("\n  [Warning] No web result files found. Only training analysis will run.")

    # 3. Per text-type analysis
    step = 3
    for text_type, df_web in web_dfs.items():
        print(f"\n[{step}/5] Analysis for text_type='{text_type}'...")

        datasets = sorted(
            set(df_train["dataset"].dropna().unique())
            | set(df_web["dataset"].dropna().unique())
        )

        # Per-dataset
        for ds in datasets:
            ds_label = ds.lower()
            ds_dir = RESULTS_DIR / text_type / ds_label
            print(f"\n  -- {ds.upper()} / {text_type} --")

            dt = df_train[df_train["dataset"] == ds].copy()
            dw = df_web[df_web["dataset"] == ds].copy()

            result = run_analysis_pipeline(dt, dw, ds_dir, subset_label=f"{ds_label}_{text_type}")
            if result is None:
                print(f"! No common models to compare for dataset '{ds}'.")
            else:
                print(f"OK  Results saved to: {ds_dir}")

        # Global (all datasets combined)
        all_dir = RESULTS_DIR / text_type / "all"
        result_all = run_analysis_pipeline(
            df_train, df_web, all_dir, subset_label=f"all_{text_type}"
        )
        if result_all is None:
            print(f"  ! No common models for global comparison ({text_type}).")
        else:
            print(f"  OK  Global results saved to: {all_dir}")

    # 4. Training-only analysis (no web data needed)
    print(f"\n[4/5] Training-only ranking...")
    train_dir = RESULTS_DIR / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    if not df_train.empty:
        compare_within(df_train, label="training", output_dir=train_dir)
        print(f"  OK  Saved to: {train_dir}")

    # 5. Cross-text-type comparison
    if len(web_dfs) > 1:
        print(f"\n[5/5] Cross-text-type comparison...")
        cross_dir = RESULTS_DIR / "cross_text_type"
        plot_cross_text_type_comparison(web_dfs, cross_dir)
        print(f"  OK  Saved to: {cross_dir}")
    else:
        print(f"\n[5/5] Skipping cross-text-type comparison (need ≥ 2 text types).")

    print(f"\n{'=' * 60}")
    print(f"Analysis complete. All results in: {RESULTS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
