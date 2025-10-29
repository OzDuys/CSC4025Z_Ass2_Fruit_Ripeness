#!/usr/bin/env python3
"""Plot macro-averaged ROC curves for the hyperparameter sweep.

This script reads the CSV exported from the W&B sweep (see artefacts folder),
computes a macro-averaged ROC curve for each run by interpolating the
per-class curves onto a shared grid, and then plots all curves together. The
selected run (``still-sweep-1``) is highlighted in red while the remaining runs
are rendered in grey.

Usage:
    python training notebooks/cnn/hyperparam_tuning/plot_roc_curves.py

The figure is saved to ``results/figures/roc_sweep.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


CSV_PATH = Path(
    "training notebooks/cnn/hyperparam_tuning/artefacts/"
    "Fruit Ripeness CNN Export Oct 29 2025.csv"
)
BEST_RUN = "still-sweep-1"
OUTPUT_PATH = Path("training notebooks/cnn/hyperparam_tuning/artefacts/roc_sweep.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot macro-averaged ROC curves across the sweep.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="Path to the exported ROC CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--best-run",
        default=BEST_RUN,
        help="Name of the run to highlight (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Where to save the figure (default: %(default)s)",
    )
    parser.add_argument(
        "--no-log", action="store_true", help="Disable log scale on the FPR axis."
    )
    parser.add_argument(
        "--points",
        type=int,
        default=2000,
        help="Number of interpolation points along the FPR axis (default: %(default)s)",
    )
    return parser


def macro_roc_curve(run_df: pd.DataFrame, fpr_grid: np.ndarray) -> np.ndarray:
    """Compute a macro-averaged ROC curve for a single run."""

    interpolated: list[np.ndarray] = []

    for _, class_df in run_df.groupby("class"):
        curve = class_df[["fpr", "tpr"]].drop_duplicates()
        curve = curve.groupby("fpr", as_index=False).max().sort_values("fpr")

        fpr = curve["fpr"].to_numpy(float)
        tpr = curve["tpr"].to_numpy(float)

        if fpr.size == 0:
            continue

        # Ensure the curve starts at (0, 0) and ends at (1, 1).
        if fpr[0] > 0.0:
            fpr = np.concatenate(([0.0], fpr))
            tpr = np.concatenate(([0.0], tpr))
        else:
            fpr[0] = 0.0
            tpr[0] = 0.0
        if fpr[-1] < 1.0:
            fpr = np.concatenate((fpr, [1.0]))
            tpr = np.concatenate((tpr, [1.0]))
        else:
            fpr[-1] = 1.0
            tpr[-1] = 1.0

        interpolated.append(np.interp(fpr_grid, fpr, tpr))

    if not interpolated:
        raise ValueError("No ROC data found for run.")

    return np.mean(interpolated, axis=0)


def plot_curves(
    df: pd.DataFrame,
    runs: Iterable[str],
    fpr_grid: np.ndarray,
    highlight: str,
    output_path: Path,
    use_log_scale: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    for run in runs:
        run_df = df[df["name"] == run]
        macro_tpr = macro_roc_curve(run_df, fpr_grid)

        is_best = run == highlight
        colour = "#d62728" if is_best else "#b0b0b0"
        linewidth = 2.5 if is_best else 1.0
        alpha = 1.0 if is_best else 0.4
        label = f"{run} (selected)" if is_best else None

        plot_fpr = np.clip(fpr_grid, 1e-6, 1.0) if use_log_scale else fpr_grid
        ax.plot(plot_fpr, macro_tpr, color=colour, lw=linewidth, alpha=alpha, label=label)

    diag = np.clip(fpr_grid, 1e-6, 1.0) if use_log_scale else fpr_grid
    ax.plot(diag, diag, color="black", ls="--", lw=0.8, alpha=0.6)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Macro-averaged ROC curves across hyperparameter sweep")
    ax.set_ylim(0.0, 1.02)

    if use_log_scale:
        ax.set_xscale("log")
        ax.set_xlim(1e-6, 1.0)
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax.set_xlim(0.0, 1.0)

    if any(run == highlight for run in runs):
        ax.legend(loc="lower right", frameon=False)

    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved ROC plot to {output_path.resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    runs = sorted(df["name"].unique())
    if not runs:
        raise ValueError(f"No runs found in {args.csv}")

    fpr_grid = np.linspace(0.0, 1.0, args.points)
    plot_curves(
        df,
        runs,
        fpr_grid,
        highlight=args.best_run,
        output_path=args.output,
        use_log_scale=not args.no_log,
    )


if __name__ == "__main__":
    main()
