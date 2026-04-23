#!/usr/bin/env python3
"""
Generate paper-inspired visualizations from earthquake model training logs.

This script is designed for the user's uploaded logs, which contain newline-separated
JSON objects intermixed with plain-text lines such as "Saved train graphs" and
"Early stopping ...".

Supported input:
- a .zip file containing .log files
- a directory of .log files
- a single .log file

Outputs:
- parsed CSV summaries
- paper-inspired PNG charts
- a markdown summary describing what was generated

The plots are inspired by figure styles mentioned in the supplied paper list:
- Convertito et al. (2024) / PreD-Net: comparative metric bars
- Zlydenko et al. (2023): ROC-style operating-point comparison at fixed FPR
- Wang et al. (2020): grouped accuracy-bar comparisons
- SeismoQuakeGNN (2025): epoch curves and pairplot-style multivariate views

Important limitation:
This script only recreates figure *styles* that are supportable from the logs.
It does NOT fabricate event-level predictions, spatial maps, confusion matrices,
or full ROC curves when those raw data are not present in the logs.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix


# -----------------------------
# Benchmarks distilled from metricGaps.md
# -----------------------------
BENCHMARKS = {
    "pred_net_total": {
        "precision": 0.851,
        "recall": 0.838,
        "f1": 0.839,
        "auc_roc": 0.758,
    },
    "pred_net_geysers": {
        "precision": 0.926,
        "recall": 0.924,
        "f1": 0.923,
        "auc_roc": 0.817,
    },
    "pred_net_cooper": {
        "precision": 0.852,
        "recall": 0.819,
        "f1": 0.831,
        "auc_roc": 0.773,
    },
    "pred_net_hengill": {
        "precision": 0.829,
        "recall": 0.803,
        "f1": 0.817,
        "auc_roc": 0.762,
    },
    "pred_net_basel": {
        "precision": 0.782,
        "recall": 0.783,
        "f1": 0.765,
        "auc_roc": 0.684,
    },
    "zlydenko": {
        "etas_tpr_at_20_fpr": 0.80,
        "fern_plus_tpr_at_20_fpr": 0.90,
        "igpe_gain_floor": 0.04,
    },
    "wang_lstm": {
        "accuracy_2d": 0.7481,
        "tp_accuracy_2d": 0.6856,
        "tn_accuracy_2d": 0.8131,
        "accuracy_decomp": 0.8512,
        "tp_accuracy_decomp": 0.7707,
        "tn_accuracy_decomp": 0.9349,
    },
    "seismoquakegnn": {
        "accuracy": 0.98,
        "lstm_accuracy": 0.9745,
        "r2": 0.88,
        "lstm_r2": 0.7719,
        "mse": 0.07,
        "lstm_mse": 0.1245,
    },
    "speed_target": {
        "ms_per_event_goal": 100.0,
    },
}


# -----------------------------
# Parsing helpers
# -----------------------------
def find_log_files(input_path: Path) -> Tuple[List[Path], Path | None]:
    """Return log files and an optional temporary directory to clean up."""
    temp_dir: Path | None = None

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        temp_dir = Path(tempfile.mkdtemp(prefix="logs_extract_"))
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(temp_dir)
        log_files = sorted(temp_dir.rglob("*.log"))
        return log_files, temp_dir

    if input_path.is_file() and input_path.suffix.lower() == ".log":
        return [input_path], None

    if input_path.is_dir():
        log_files = sorted(input_path.rglob("*.log"))
        return log_files, None

    raise FileNotFoundError(f"Could not locate logs at: {input_path}")


def parse_json_objects(text: str) -> List[dict]:
    """Parse a stream containing JSON objects plus non-JSON lines."""
    decoder = json.JSONDecoder()
    idx = 0
    objs: List[dict] = []

    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break

        if text[idx] != "{":
            next_newline = text.find("\n", idx)
            idx = len(text) if next_newline == -1 else next_newline + 1
            continue

        try:
            obj, end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                objs.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx += 1

    return objs


def flatten_trial_config(config: dict) -> dict:
    row = {"trial_index": config.get("trial_index")}
    for group in ("lstm", "gat"):
        for key, value in config.get(group, {}).items():
            row[f"{group}_{key}"] = value
    return row


def load_data(log_files: Iterable[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs_by_trial: Dict[int, dict] = {}
    epoch_rows: List[dict] = []
    test_rows: List[dict] = []

    for log_file in log_files:
        text = log_file.read_text(encoding="utf-8", errors="ignore")
        for obj in parse_json_objects(text):
            if "config" in obj:
                configs_by_trial[obj["trial_index"]] = obj["config"]
            elif "epoch" in obj:
                epoch_rows.append(dict(obj))
            elif "test_metrics" in obj:
                row = {
                    "trial_index": obj["trial_index"],
                    "model": obj["model"],
                    **obj["test_metrics"],
                }
                test_rows.append(row)

    config_df = pd.DataFrame(
        [flatten_trial_config(cfg) for _, cfg in sorted(configs_by_trial.items())]
    ).sort_values("trial_index")

    epoch_df = pd.DataFrame(epoch_rows).sort_values(["model", "trial_index", "epoch"])
    test_df = pd.DataFrame(test_rows).sort_values(["model", "trial_index"])

    # Normalize dtypes
    for df in (config_df, epoch_df, test_df):
        for col in df.columns:
            if df[col].dtype == object:
                if set(df[col].dropna().astype(str).unique()) <= {"True", "False"}:
                    df[col] = df[col].map({"True": True, "False": False})

    return config_df, epoch_df, test_df


# -----------------------------
# Selection logic
# -----------------------------
def get_best_trials(test_df: pd.DataFrame) -> pd.DataFrame:
    # Choose best by F1, breaking ties by AUC then latency.
    ranked = test_df.sort_values(
        ["model", "f1", "auc_roc", "ms_per_event"],
        ascending=[True, False, False, True],
    )
    return ranked.groupby("model", as_index=False).head(1)


def get_epoch_series_for_best(epoch_df: pd.DataFrame, best_df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, row in best_df.iterrows():
        subset = epoch_df[
            (epoch_df["model"] == row["model"]) & (epoch_df["trial_index"] == row["trial_index"])
        ].copy()
        pieces.append(subset)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def compute_pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Pareto frontier for minimizing x and maximizing y."""
    frontier_rows = []
    for i, row_i in df.iterrows():
        dominated = False
        for j, row_j in df.iterrows():
            if i == j:
                continue
            no_worse = row_j[x_col] <= row_i[x_col] and row_j[y_col] >= row_i[y_col]
            strictly_better = row_j[x_col] < row_i[x_col] or row_j[y_col] > row_i[y_col]
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier_rows.append(row_i)
    if not frontier_rows:
        return pd.DataFrame(columns=df.columns)
    frontier = pd.DataFrame(frontier_rows).sort_values(x_col)
    return frontier


# -----------------------------
# Plot helpers
# -----------------------------
def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_epoch_curves(best_epoch_df: pd.DataFrame, output_dir: Path) -> None:
    # Inspired by SeismoQuakeGNN Fig. 13: metrics across epochs.
    metric_groups = [
        (["train_loss"], "Train loss across epochs"),
        (["val_precision", "val_recall", "val_f1", "val_auc_roc"], "Validation classification metrics across epochs"),
        (["val_accuracy", "val_balanced_accuracy", "val_tpr_at_20_fpr"], "Validation accuracy-style metrics across epochs"),
    ]

    for model in sorted(best_epoch_df["model"].unique()):
        model_df = best_epoch_df[best_epoch_df["model"] == model].sort_values("epoch")
        if model_df.empty:
            continue

        for metrics, title in metric_groups:
            plt.figure(figsize=(9, 5))
            for metric in metrics:
                if metric in model_df.columns and not model_df[metric].isna().all():
                    plt.plot(model_df["epoch"], model_df[metric], marker="o", label=metric)
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.title(f"{model}: {title} (trial {int(model_df['trial_index'].iloc[0])})")
            plt.grid(True, alpha=0.3)
            plt.legend()
            filename = f"01_{model.lower()}_{'_'.join(m.replace('val_', '') for m in metrics)}.png"
            savefig(output_dir / filename)


def plot_trial_distributions(test_df: pd.DataFrame, output_dir: Path) -> None:
    # Paper-inspired comparative bars / distributions.
    metrics = ["precision", "recall", "f1", "auc_roc", "accuracy", "balanced_accuracy", "tpr_at_20_fpr"]
    summary = test_df.groupby("model")[metrics].agg(["mean", "median", "max"])  # type: ignore[arg-type]

    # mean/median/max grouped bars for each model and metric
    for stat in ["mean", "median", "max"]:
        stat_df = summary.xs(stat, axis=1, level=1)
        x = np.arange(len(stat_df.columns))
        width = 0.35

        plt.figure(figsize=(11, 5.5))
        models = list(stat_df.index)
        for i, model in enumerate(models):
            offset = (i - (len(models) - 1) / 2) * width
            plt.bar(x + offset, stat_df.loc[model].values, width=width, label=model)
        plt.xticks(x, stat_df.columns, rotation=20)
        plt.ylim(0, 1.0)
        plt.ylabel("Metric value")
        plt.title(f"Across-trial {stat} test metrics by model")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        savefig(output_dir / f"02_trial_metric_{stat}.png")


def plot_preD_net_style_benchmark(best_df: pd.DataFrame, output_dir: Path) -> None:
    # Inspired by Convertito Fig. 9 / metric-comparison tables.
    metrics = ["precision", "recall", "f1", "auc_roc"]
    benchmark_rows = {
        "Best LSTM": best_df.loc[best_df["model"] == "LSTM", metrics].iloc[0].to_dict()
        if (best_df["model"] == "LSTM").any() else None,
        "Best GAT": best_df.loc[best_df["model"] == "GAT", metrics].iloc[0].to_dict()
        if (best_df["model"] == "GAT").any() else None,
        "PreD-Net total": BENCHMARKS["pred_net_total"],
        "PreD-Net Basel": BENCHMARKS["pred_net_basel"],
        "PreD-Net Geysers": BENCHMARKS["pred_net_geysers"],
    }
    benchmark_rows = {k: v for k, v in benchmark_rows.items() if v is not None}

    df = pd.DataFrame(benchmark_rows).T[metrics]
    x = np.arange(len(metrics))
    width = 0.14 if len(df) >= 5 else 0.18

    plt.figure(figsize=(12, 6))
    for i, (label, row) in enumerate(df.iterrows()):
        offset = (i - (len(df) - 1) / 2) * width
        plt.bar(x + offset, row.values, width=width, label=label)
    plt.xticks(x, [m.upper() if m != "auc_roc" else "AUC" for m in metrics])
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title("PreD-Net-style comparison: your best logged models vs paper foreshock benchmarks")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(ncol=2)
    savefig(output_dir / "03_prednet_style_benchmark_prf_auc.png")


def plot_wang_style_accuracy_bars(best_df: pd.DataFrame, output_dir: Path) -> None:
    # Inspired by Wang et al. grouped bars in Fig. 15-17.
    rows = {
        "Best LSTM": {
            "overall_accuracy": float(best_df.loc[best_df["model"] == "LSTM", "accuracy"].iloc[0]) if (best_df["model"] == "LSTM").any() else np.nan,
            "positive_like": float(best_df.loc[best_df["model"] == "LSTM", "recall"].iloc[0]) if (best_df["model"] == "LSTM").any() else np.nan,
            "balanced_accuracy": float(best_df.loc[best_df["model"] == "LSTM", "balanced_accuracy"].iloc[0]) if (best_df["model"] == "LSTM").any() else np.nan,
        },
        "Best GAT": {
            "overall_accuracy": float(best_df.loc[best_df["model"] == "GAT", "accuracy"].iloc[0]) if (best_df["model"] == "GAT").any() else np.nan,
            "positive_like": float(best_df.loc[best_df["model"] == "GAT", "recall"].iloc[0]) if (best_df["model"] == "GAT").any() else np.nan,
            "balanced_accuracy": float(best_df.loc[best_df["model"] == "GAT", "balanced_accuracy"].iloc[0]) if (best_df["model"] == "GAT").any() else np.nan,
        },
        "Wang 2D LSTM": {
            "overall_accuracy": BENCHMARKS["wang_lstm"]["accuracy_2d"],
            "positive_like": BENCHMARKS["wang_lstm"]["tp_accuracy_2d"],
            "balanced_accuracy": np.nan,
        },
        "Wang LSTM + decomp": {
            "overall_accuracy": BENCHMARKS["wang_lstm"]["accuracy_decomp"],
            "positive_like": BENCHMARKS["wang_lstm"]["tp_accuracy_decomp"],
            "balanced_accuracy": np.nan,
        },
    }
    df = pd.DataFrame(rows).T
    metrics = ["overall_accuracy", "positive_like", "balanced_accuracy"]
    x = np.arange(len(df.index))
    width = 0.22

    plt.figure(figsize=(11, 6))
    for i, metric in enumerate(metrics):
        plt.bar(x + (i - 1) * width, df[metric].values, width=width, label=metric)
    plt.xticks(x, df.index, rotation=20)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Wang-style grouped accuracy bars")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    savefig(output_dir / "04_wang_style_accuracy_bars.png")


def plot_zlydenko_operating_point(test_df: pd.DataFrame, output_dir: Path) -> None:
    # Inspired by Zlydenko Fig. 3, but only at fixed 20% FPR because full ROC data are absent.
    plt.figure(figsize=(10, 5.5))
    for model in sorted(test_df["model"].unique()):
        subset = test_df[test_df["model"] == model].sort_values("trial_index")
        plt.scatter(
            subset["actual_fpr_at_selected_roc_point"],
            subset["tpr_at_20_fpr"],
            label=model,
            s=55,
            alpha=0.85,
        )
        for _, row in subset.iterrows():
            plt.annotate(
                str(int(row["trial_index"])),
                (row["actual_fpr_at_selected_roc_point"], row["tpr_at_20_fpr"]),
                fontsize=8,
                alpha=0.8,
            )

    plt.axhline(BENCHMARKS["zlydenko"]["etas_tpr_at_20_fpr"], linestyle="--", label="ETAS @20% FPR")
    plt.axhline(BENCHMARKS["zlydenko"]["fern_plus_tpr_at_20_fpr"], linestyle=":", label="FERN+ @20% FPR")
    plt.axvline(0.20, linestyle="-.", label="Target FPR = 0.20")
    plt.xlabel("Actual FPR at selected ROC operating point")
    plt.ylabel("TPR at ~20% FPR")
    plt.xlim(0.17, 0.23)
    plt.ylim(0.0, 1.0)
    plt.title("Zlydenko-style fixed-FPR operating-point comparison")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    savefig(output_dir / "05_zlydenko_style_operating_point.png")


def plot_latency_frontier(test_df: pd.DataFrame, output_dir: Path) -> None:
    # Efficiency vs effectiveness tradeoff.
    plt.figure(figsize=(10, 5.5))
    for model in sorted(test_df["model"].unique()):
        subset = test_df[test_df["model"] == model]
        plt.scatter(subset["ms_per_event"], subset["f1"], label=model, s=55, alpha=0.85)

    frontier = compute_pareto_frontier(test_df, x_col="ms_per_event", y_col="f1")
    if not frontier.empty:
        plt.plot(frontier["ms_per_event"], frontier["f1"], linestyle="--", marker="o", label="Pareto frontier")

    plt.axvline(BENCHMARKS["speed_target"]["ms_per_event_goal"], linestyle=":", label="100 ms/event target")
    plt.xlabel("Milliseconds per event (lower is better)")
    plt.ylabel("F1 score (higher is better)")
    plt.title("Latency vs F1 Pareto frontier")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(output_dir / "06_latency_vs_f1_pareto.png")


def plot_ranked_trials(test_df: pd.DataFrame, output_dir: Path) -> None:
    ranked = test_df.sort_values(["f1", "auc_roc"], ascending=[False, False]).copy()
    ranked["label"] = ranked["model"] + "-T" + ranked["trial_index"].astype(int).astype(str)

    x = np.arange(len(ranked))
    plt.figure(figsize=(12, 5.5))
    plt.bar(x, ranked["f1"].values, label="F1")
    plt.plot(x, ranked["auc_roc"].values, marker="o", label="AUC")
    plt.xticks(x, ranked["label"], rotation=70)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Ranked test trials: F1 bars with AUC overlay")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    savefig(output_dir / "07_ranked_trials_f1_auc.png")


def plot_heatmaps(test_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["precision", "recall", "f1", "accuracy", "balanced_accuracy", "auc_roc", "tpr_at_20_fpr"]

    for model in sorted(test_df["model"].unique()):
        subset = test_df[test_df["model"] == model].sort_values("trial_index").copy()
        if subset.empty:
            continue

        mat = subset[metrics].to_numpy(dtype=float)
        col_means = np.nanmean(mat, axis=0)
        col_stds = np.nanstd(mat, axis=0)
        col_stds[col_stds == 0] = 1.0
        zmat = (mat - col_means) / col_stds

        plt.figure(figsize=(10, 5.5))
        plt.imshow(zmat, aspect="auto")
        plt.colorbar(label="z-score within model")
        plt.yticks(np.arange(len(subset)), subset["trial_index"].astype(int).tolist())
        plt.xticks(np.arange(len(metrics)), metrics, rotation=30, ha="right")
        plt.xlabel("Metric")
        plt.ylabel("Trial index")
        plt.title(f"Heatmap of standardized test metrics for {model}")
        savefig(output_dir / f"08_{model.lower()}_metric_heatmap.png")


def plot_pairplot_style(config_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    # Inspired by SeismoQuakeGNN Fig. 5 pairplot; use scatter_matrix without seaborn.
    merged = test_df.merge(config_df, on="trial_index", how="left")

    for model, prefix in [("GAT", "gat_"), ("LSTM", "lstm_")]:
        subset = merged[merged["model"] == model].copy()
        if subset.empty:
            continue

        candidate_cols = [
            c
            for c in subset.columns
            if c.startswith(prefix)
            and pd.api.types.is_numeric_dtype(subset[c])
            and subset[c].nunique() > 1
        ]
        if not candidate_cols:
            continue

        corr_scores: List[Tuple[str, float]] = []
        for col in candidate_cols:
            corr = subset[col].corr(subset["f1"])
            if pd.notna(corr):
                corr_scores.append((col, abs(float(corr))))

        selected = [c for c, _ in sorted(corr_scores, key=lambda x: x[1], reverse=True)[:4]]
        selected += ["f1", "auc_roc", "ms_per_event"]
        selected = list(dict.fromkeys(selected))  # stable dedupe

        plot_df = subset[selected].copy()
        axes = scatter_matrix(plot_df, figsize=(12, 12), diagonal="hist", alpha=0.8)
        # Add readable labels
        for ax in np.ravel(axes):
            ax.grid(True, alpha=0.2)
        plt.suptitle(f"Pairplot-style hyperparameter view for {model}", y=0.92)
        plt.savefig(output_dir / f"09_{model.lower()}_pairplot_style.png", dpi=220, bbox_inches="tight")
        plt.close()


def plot_auc_vs_tpr20(test_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    for model in sorted(test_df["model"].unique()):
        subset = test_df[test_df["model"] == model]
        plt.scatter(subset["auc_roc"], subset["tpr_at_20_fpr"], s=55, alpha=0.85, label=model)

    plt.axvline(BENCHMARKS["pred_net_total"]["auc_roc"], linestyle="--", label="PreD-Net total AUC")
    plt.axhline(BENCHMARKS["zlydenko"]["fern_plus_tpr_at_20_fpr"], linestyle=":", label="FERN+ TPR @20% FPR")
    plt.xlabel("Test AUC-ROC")
    plt.ylabel("TPR at ~20% FPR")
    plt.xlim(0.45, 0.95)
    plt.ylim(0.0, 1.0)
    plt.title("AUC vs fixed-FPR sensitivity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(output_dir / "10_auc_vs_tpr20.png")


def write_summary(best_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    summary_lines = []
    summary_lines.append("# Visualization summary\n")
    summary_lines.append("This folder was generated from the uploaded training/test logs.\n")

    if not best_df.empty:
        summary_lines.append("## Best trial per model (ranked by F1, then AUC, then latency)\n")
        keep = [
            "model",
            "trial_index",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "balanced_accuracy",
            "auc_roc",
            "tpr_at_20_fpr",
            "ms_per_event",
        ]
        summary_lines.append(best_df[keep].to_markdown(index=False))
        summary_lines.append("\n")

    summary_lines.append("## Generated charts\n")
    summary_lines.extend(
        [
            "- 01_* : SeismoQuakeGNN-style epoch curves\n",
            "- 02_* : across-trial metric summaries\n",
            "- 03_* : PreD-Net-style benchmark bars for Precision/Recall/F1/AUC\n",
            "- 04_* : Wang-style grouped accuracy bars\n",
            "- 05_* : Zlydenko-style operating-point comparison at fixed FPR\n",
            "- 06_* : latency vs F1 Pareto frontier\n",
            "- 07_* : ranked trials\n",
            "- 08_* : per-model test-metric heatmaps\n",
            "- 09_* : pairplot-style hyperparameter scatter matrices\n",
            "- 10_* : AUC vs TPR@20%FPR\n",
        ]
    )
    summary_lines.append("\n")
    summary_lines.append("## Important limitations\n")
    summary_lines.extend(
        [
            "- Full ROC curves cannot be reconstructed because the logs contain only AUC and a fixed operating point, not raw prediction scores.\n",
            "- Actual-vs-predicted magnitude traces cannot be recreated because the logs do not include per-sample predictions.\n",
            "- Spatial seismicity maps and event timelines cannot be recreated because the logs do not include latitude/longitude/time per event.\n",
        ]
    )

    (output_dir / "README_generated.md").write_text("".join(summary_lines), encoding="utf-8")

def plot_lstm_vs_gat_train_loss_difference(best_epoch_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Compare training loss of the best LSTM trial vs the best GAT trial.
    Also saves a CSV with per-epoch loss difference:
        loss_diff = GAT_train_loss - LSTM_train_loss
    """
    subset = best_epoch_df[
        best_epoch_df["model"].isin(["LSTM", "GAT"])
    ][["model", "trial_index", "epoch", "train_loss"]].copy()

    if subset.empty or subset["model"].nunique() < 2:
        print("Could not compare LSTM vs GAT training loss: both models were not found.")
        return

    # Pivot to align both models by epoch
    pivot = (
        subset.pivot_table(index="epoch", columns="model", values="train_loss", aggfunc="first")
        .sort_index()
    )

    if "LSTM" not in pivot.columns or "GAT" not in pivot.columns:
        print("Could not compare LSTM vs GAT training loss: missing one model after pivot.")
        return

    # Keep only epochs present in both
    pivot = pivot.dropna(subset=["LSTM", "GAT"]).copy()
    if pivot.empty:
        print("No overlapping epochs found between best LSTM and best GAT trials.")
        return

    pivot["loss_diff_gat_minus_lstm"] = pivot["GAT"] - pivot["LSTM"]
    pivot["better_model"] = np.where(
        pivot["loss_diff_gat_minus_lstm"] < 0, "GAT",
        np.where(pivot["loss_diff_gat_minus_lstm"] > 0, "LSTM", "Tie")
    )

    # Save numeric comparison
    pivot.reset_index().to_csv(output_dir / "train_loss_difference_lstm_vs_gat.csv", index=False)

    # Plot both training-loss curves + difference
    plt.figure(figsize=(10, 6))
    plt.plot(pivot.index, pivot["LSTM"], marker="o", label="LSTM train_loss")
    plt.plot(pivot.index, pivot["GAT"], marker="o", label="GAT train_loss")
    plt.plot(
        pivot.index,
        pivot["loss_diff_gat_minus_lstm"],
        marker="o",
        linestyle="--",
        label="GAT - LSTM loss difference",
    )

    plt.axhline(0.0, linestyle=":")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM vs GAT training loss difference")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(output_dir / "01_lstm_vs_gat_train_loss_difference.png")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-inspired visualizations from model logs.")
    parser.add_argument(
        "--input",
        default="./logs/",
        help="Path to logs.zip, a logs directory, or a single .log file (default: ./logs/)",
    )
    parser.add_argument("--output-dir", default="visualization_outputs", help="Directory to save charts and summaries")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_dir: Path | None = None
    try:
        log_files, temp_dir = find_log_files(input_path)
        if not log_files:
            raise RuntimeError(f"No .log files found under {input_path}")

        config_df, epoch_df, test_df = load_data(log_files)
        if test_df.empty:
            raise RuntimeError("No test metrics found in logs.")

        # Save parsed tables
        config_df.to_csv(output_dir / "parsed_configs.csv", index=False)
        epoch_df.to_csv(output_dir / "parsed_epochs.csv", index=False)
        test_df.to_csv(output_dir / "parsed_test_metrics.csv", index=False)

        best_df = get_best_trials(test_df)
        best_epoch_df = get_epoch_series_for_best(epoch_df, best_df)
        best_df.to_csv(output_dir / "best_trials.csv", index=False)

        # Generate plots
        if not best_epoch_df.empty:
            plot_epoch_curves(best_epoch_df, output_dir)
        plot_trial_distributions(test_df, output_dir)
        plot_preD_net_style_benchmark(best_df, output_dir)
        plot_wang_style_accuracy_bars(best_df, output_dir)
        plot_zlydenko_operating_point(test_df, output_dir)
        plot_latency_frontier(test_df, output_dir)
        plot_ranked_trials(test_df, output_dir)
        plot_heatmaps(test_df, output_dir)
        plot_pairplot_style(config_df, test_df, output_dir)
        plot_auc_vs_tpr20(test_df, output_dir)
        write_summary(best_df, test_df, output_dir)
        plot_lstm_vs_gat_train_loss_difference(best_epoch_df, output_dir)

        print(f"Generated outputs in: {output_dir}")
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
