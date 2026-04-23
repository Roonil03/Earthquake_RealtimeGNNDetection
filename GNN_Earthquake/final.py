from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import torch

import main2
from main2 import (
    BASE_FEATURE_COLS,
    IDEAL_TARGETS,
    LAT_COL,
    LON_COL,
    MAG_COL,
    TARGET_COL,
    TIME_COL,
    build_full_dataset_gat_config,
    build_graph_prediction_frame,
    build_paths,
    build_sequence_prediction_frame,
    build_target_report,
    build_full_sequence_loader,
    classification_metrics,
    ensure_cuda_device,
    graph_cache_path,
    instantiate_gat_model,
    instantiate_lstm_model,
    load_or_build_graphs,
    load_or_prepare_clean_dataframe,
    load_or_prepare_labeled_dataframe,
    load_or_prepare_splits,
    load_state_dict_safely,
    log_json,
    log_line,
    measure_geo_latency_per_event,
    measure_latency_per_event,
    predict_geo_loader,
    predict_loader,
    sanitize_for_log,
    score_against_targets,
    select_graph_for_visualization,
    set_seed,
)

try:
    from torch_geometric.loader import DataLoader as GeoDataLoader
except Exception:
    GeoDataLoader = None


FINAL_LOG_FILENAME = "final_logs.log"
FINAL_SUMMARY_FILENAME = "final_results_from_best_models.json"
FINAL_GRAPH_VIS_FILENAME = "final_full_graph_sample_3d.png"
FINAL_POINTS_VIS_FILENAME = "final_processed_points_3d.png"
FINAL_GRAPH_EDGES_FILENAME = "final_full_graph_edges.csv"
BEST_GAT_RESULTS_FILENAME = "best_gat_results.json"
BEST_LSTM_RESULTS_FILENAME = "best_lstm_results.json"


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final full-dataset earthquake pipeline from best model result JSON files")
    parser.add_argument("--project-path", default=None)
    parser.add_argument("--raw-csv-path", default=None)
    parser.add_argument("--best-gat-results-path", default=None)
    parser.add_argument("--best-lstm-results-path", default=None)
    parser.add_argument("--run-lstm", type=str2bool, default=True)
    parser.add_argument("--run-gat", type=str2bool, default=True)
    parser.add_argument("--use-sample-for-debug", type=str2bool, default=False)
    parser.add_argument("--debug-n-rows", type=int, default=400_000)
    parser.add_argument("--mainshock-mag-threshold", type=float, default=5.5)
    parser.add_argument("--lead-days", type=int, default=30)
    parser.add_argument("--spatial-radius-km", type=float, default=100.0)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.075)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual-sample-size", type=int, default=10000)
    parser.add_argument("--all-point-chunk-size", type=int, default=250_000)
    return parser.parse_args()


def build_final_paths(base_dir: Path, project_path: str | None) -> dict[str, Path]:
    paths = build_paths(base_dir, project_path)
    paths["final_log_path"] = paths["logs_dir"] / FINAL_LOG_FILENAME
    paths["final_summary_path"] = paths["artifacts_dir"] / FINAL_SUMMARY_FILENAME
    paths["final_graph_visual_path"] = paths["visualizations_dir"] / FINAL_GRAPH_VIS_FILENAME
    paths["final_points_visual_path"] = paths["visualizations_dir"] / FINAL_POINTS_VIS_FILENAME
    paths["final_graph_edges_path"] = paths["visualizations_dir"] / FINAL_GRAPH_EDGES_FILENAME
    return paths


def resolve_json_path(base_dir: Path, explicit_path: str | None, filename: str, artifacts_dir: Path) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    candidates.extend([
        base_dir / filename,
        artifacts_dir / filename,
        Path.cwd() / filename,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find {filename}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path.name}")
    return payload


def local_checkpoint_path(model_name: str, trial_index: int, models_dir: Path) -> Path:
    return models_dir / f"{model_name.lower()}_trial_{trial_index:03d}.pt"


def normalize_candidate(payload: dict[str, Any], model_name: str, models_dir: Path) -> dict[str, Any]:
    trial_index = int(payload["trial_index"])
    result = dict(payload.get("result", {}))
    metrics = dict(result.get("metrics", {}))
    config = dict(result.get("config", {}))
    original_model_path = result.get("model_path")
    local_model_path = local_checkpoint_path(model_name, trial_index, models_dir)
    candidate = {
        "score": float(payload.get("score", score_against_targets(metrics, payload.get("proxy_ltss_improvement_pct") if model_name == "GAT" else None))),
        "model": model_name,
        "trial_index": trial_index,
        "result": {
            "config": config,
            "history": list(result.get("history", [])),
            "metrics": metrics,
            "model_path": str(local_model_path),
            "original_model_path": original_model_path,
        },
        "target_report": payload.get("target_report", build_target_report(metrics, payload.get("proxy_ltss_improvement_pct") if model_name == "GAT" else None, model_name == "GAT")),
        "proxy_ltss_improvement_pct": payload.get("proxy_ltss_improvement_pct"),
    }
    return candidate


def resolve_existing_checkpoint(candidate: dict[str, Any], models_dir: Path) -> Path:
    local_path = Path(candidate["result"]["model_path"])
    if local_path.exists():
        return local_path
    original_model_path = candidate["result"].get("original_model_path")
    if original_model_path:
        original_path = Path(str(original_model_path))
        if original_path.exists():
            candidate["result"]["model_path"] = str(original_path)
            return original_path
    fallback = local_checkpoint_path(candidate["model"], int(candidate["trial_index"]), models_dir)
    candidate["result"]["model_path"] = str(fallback)
    return fallback


def maybe_rebuild_checkpoint(candidate: dict[str, Any] | None, model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], device: torch.device, paths: dict[str, Path], final_log_path: Path) -> dict[str, Any] | None:
    if candidate is None:
        return None
    checkpoint_path = resolve_existing_checkpoint(candidate, paths["models_dir"])
    if checkpoint_path.exists():
        candidate["result"]["model_path"] = str(checkpoint_path)
        return candidate
    log_line(final_log_path, f"Checkpoint missing for {model_name} trial {candidate['trial_index']}. Rebuilding it now.")
    trial_config = {
        "trial_index": int(candidate["trial_index"]),
        "lstm": {},
        "gat": {},
    }
    trial_config[model_name.lower()].update(candidate["result"]["config"])
    if model_name == "LSTM":
        rebuilt = main2.run_lstm_trial(
            int(candidate["trial_index"]),
            trial_config,
            train_df,
            val_df,
            test_df,
            feature_cols,
            device,
            paths["models_dir"],
            final_log_path,
        )
        if rebuilt is not None:
            rebuilt["model_path"] = str(local_checkpoint_path("LSTM", int(candidate["trial_index"]), paths["models_dir"]))
            rebuilt["original_model_path"] = rebuilt["model_path"]
            candidate["result"] = rebuilt
    else:
        rebuilt, _ = main2.run_gat_trial(
            int(candidate["trial_index"]),
            trial_config,
            train_df,
            val_df,
            test_df,
            feature_cols,
            device,
            paths["graphs_dir"],
            paths["models_dir"],
            paths["visualizations_dir"],
            final_log_path,
            True,
        )
        if rebuilt is not None:
            rebuilt["model_path"] = str(local_checkpoint_path("GAT", int(candidate["trial_index"]), paths["models_dir"]))
            rebuilt["original_model_path"] = rebuilt["model_path"]
            candidate["result"] = rebuilt
    return candidate


def build_scaled_full_dataframe(df_labeled: pd.DataFrame, scaler: Any, feature_cols: list[str]) -> pd.DataFrame:
    df_full = df_labeled.copy()
    df_full[feature_cols] = scaler.transform(df_full[feature_cols])
    return df_full


def visualize_full_graph_sample_time_space(full_graphs: list[Any], output_path: Path, edge_csv_path: Path, log_path: Path | None) -> None:
    graph = select_graph_for_visualization(full_graphs)
    if graph is None:
        log_line(log_path, "No graph available for final graph visualization.")
        return
    longitude = graph.node_lon.detach().cpu().numpy()
    latitude = graph.node_lat.detach().cpu().numpy()
    magnitude = graph.node_mag.detach().cpu().numpy()
    edge_index = graph.edge_index.detach().cpu().numpy()
    edge_distance_km = graph.edge_distance_km.detach().cpu().numpy()
    edge_time_days = graph.edge_time_days.detach().cpu().numpy()
    center_time = pd.to_datetime(int(graph.graph_time))
    delta_days = graph.node_delta_days.detach().cpu().numpy()
    node_time = center_time - pd.to_timedelta(delta_days, unit="D")
    node_time_num = mdates.date2num(pd.to_datetime(node_time).to_pydatetime())
    unique_mask = edge_index[0] < edge_index[1]
    source_idx = edge_index[0][unique_mask]
    target_idx = edge_index[1][unique_mask]
    if source_idx.size == 0:
        log_line(log_path, "Final graph sample has no unique edges to visualize.")
        return
    edge_table = pd.DataFrame({
        "source_node": source_idx.astype(int),
        "target_node": target_idx.astype(int),
        "source_longitude": longitude[source_idx],
        "source_latitude": latitude[source_idx],
        "source_time": pd.to_datetime(node_time[source_idx]).astype(str),
        "source_magnitude": magnitude[source_idx],
        "target_longitude": longitude[target_idx],
        "target_latitude": latitude[target_idx],
        "target_time": pd.to_datetime(node_time[target_idx]).astype(str),
        "target_magnitude": magnitude[target_idx],
        "distance_km": edge_distance_km[unique_mask],
        "time_gap_days": edge_time_days[unique_mask],
    }).sort_values(["distance_km", "time_gap_days"], ascending=[False, False]).reset_index(drop=True)
    edge_table.to_csv(edge_csv_path, index=False)
    figure = plt.figure(figsize=(14, 10))
    axis = figure.add_subplot(111, projection="3d")
    norm = mcolors.Normalize(vmin=float(np.min(magnitude)), vmax=float(np.max(magnitude)))
    mapper = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    colors = mapper.to_rgba(magnitude)
    axis.scatter(longitude, latitude, node_time_num, c=colors, s=28 + 10 * magnitude, alpha=0.95)
    draw_count = min(len(edge_table), 160)
    for row in edge_table.head(draw_count).itertuples(index=False):
        axis.plot(
            [row.source_longitude, row.target_longitude],
            [row.source_latitude, row.target_latitude],
            [mdates.date2num(pd.to_datetime(row.source_time)), mdates.date2num(pd.to_datetime(row.target_time))],
            color="gray",
            alpha=0.22,
            linewidth=0.8,
        )
    annotate_count = min(len(edge_table), 18)
    for row in edge_table.head(annotate_count).itertuples(index=False):
        x_mid = (row.source_longitude + row.target_longitude) / 2.0
        y_mid = (row.source_latitude + row.target_latitude) / 2.0
        z_mid = (mdates.date2num(pd.to_datetime(row.source_time)) + mdates.date2num(pd.to_datetime(row.target_time))) / 2.0
        axis.text(x_mid, y_mid, z_mid, f"d={row.distance_km:.1f}km\nΔt={row.time_gap_days:.2f}d", fontsize=7)
    axis.set_title("Final Full-Dataset GAT Graph Sample")
    axis.set_xlabel(LON_COL)
    axis.set_ylabel(LAT_COL)
    axis.set_zlabel(TIME_COL)
    axis.zaxis.set_major_formatter(FuncFormatter(lambda value, _: mdates.num2date(value).strftime("%Y-%m")))
    colorbar = figure.colorbar(mapper, ax=axis, pad=0.08)
    colorbar.set_label(MAG_COL)
    figure.savefig(output_path, bbox_inches="tight", dpi=220)
    plt.close(figure)
    log_line(log_path, f"Saved final graph visualization to {output_path.name}")


def visualize_all_processed_points_3d(df_in: pd.DataFrame, output_path: Path, chunk_size: int, log_path: Path | None) -> None:
    if df_in.empty:
        return
    figure = plt.figure(figsize=(16, 10))
    axis = figure.add_subplot(111, projection="3d")
    df_sorted = df_in.sort_values(TIME_COL).reset_index(drop=True)
    time_num = mdates.date2num(pd.to_datetime(df_sorted[TIME_COL]).dt.to_pydatetime())
    magnitude = df_sorted[MAG_COL].to_numpy(dtype=np.float32)
    longitude = df_sorted[LON_COL].to_numpy(dtype=np.float32)
    latitude = df_sorted[LAT_COL].to_numpy(dtype=np.float32)
    norm = mcolors.Normalize(vmin=float(np.min(magnitude)), vmax=float(np.max(magnitude)))
    mapper = cm.ScalarMappable(norm=norm, cmap="coolwarm")
    for start in range(0, len(df_sorted), max(1, chunk_size)):
        end = min(start + max(1, chunk_size), len(df_sorted))
        axis.scatter(
            longitude[start:end],
            latitude[start:end],
            time_num[start:end],
            c=mapper.to_rgba(magnitude[start:end]),
            s=0.8,
            alpha=0.28,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )
    axis.set_title("Processed Earthquake Dataset in 3D")
    axis.set_xlabel(LON_COL)
    axis.set_ylabel(LAT_COL)
    axis.set_zlabel(TIME_COL)
    axis.zaxis.set_major_formatter(FuncFormatter(lambda value, _: mdates.num2date(value).strftime("%Y-%m")))
    colorbar = figure.colorbar(mapper, ax=axis, pad=0.08)
    colorbar.set_label(MAG_COL)
    figure.savefig(output_path, bbox_inches="tight", dpi=220)
    plt.close(figure)
    log_line(log_path, f"Saved final processed-points visualization to {output_path.name}")


def run_final_lstm(candidate: dict[str, Any] | None, df_full_scaled: pd.DataFrame, feature_cols: list[str], device: torch.device, paths: dict[str, Path], log_path: Path) -> dict[str, Any] | None:
    if candidate is None:
        return None
    checkpoint_path = resolve_existing_checkpoint(candidate, paths["models_dir"])
    config = dict(candidate["result"]["config"])
    threshold = float(candidate["result"]["metrics"].get("threshold", 0.5))
    model = instantiate_lstm_model(config, len(feature_cols), device)
    load_state_dict_safely(model, checkpoint_path, device)
    dataset, loader = build_full_sequence_loader(df_full_scaled, feature_cols, config, device)
    if len(dataset) == 0:
        log_line(log_path, "Final LSTM full-dataset loader produced zero windows.")
        return None
    y_true, y_prob = predict_loader(model, loader, device)
    metrics = classification_metrics(y_true, y_prob, threshold=threshold)
    example_x, _ = next(iter(loader))
    avg_batch_latency_ms, ms_per_event = measure_latency_per_event(model, example_x[: min(256, len(example_x))], device)
    metrics["avg_batch_latency_ms"] = avg_batch_latency_ms
    metrics["ms_per_event"] = ms_per_event
    target_report = build_target_report(metrics, None, False)
    predictions = build_sequence_prediction_frame(df_full_scaled, dataset, y_prob, threshold)
    predictions.to_parquet(paths["full_lstm_predictions_path"], index=False)
    result = {
        "trial_index": int(candidate["trial_index"]),
        "score": float(candidate["score"]),
        "config": config,
        "checkpoint_file": checkpoint_path.name,
        "metrics": metrics,
        "target_report": target_report,
        "predictions_rows": int(len(predictions)),
        "prediction_file": paths["full_lstm_predictions_path"].name,
    }
    log_json(log_path, {"final_model": "LSTM", "result": result})
    return result


def run_final_gat(candidate: dict[str, Any] | None, df_full_scaled: pd.DataFrame, feature_cols: list[str], device: torch.device, paths: dict[str, Path], log_path: Path) -> dict[str, Any] | None:
    if candidate is None:
        return None
    if GeoDataLoader is None:
        raise RuntimeError("PyTorch Geometric DataLoader is unavailable. Final GAT execution cannot proceed.")
    checkpoint_path = resolve_existing_checkpoint(candidate, paths["models_dir"])
    config = build_full_dataset_gat_config(candidate["result"]["config"])
    threshold = float(candidate["result"]["metrics"].get("threshold", 0.5))
    cache_path = graph_cache_path(paths["graphs_dir"], "full", config)
    full_graphs = load_or_build_graphs(df_full_scaled, "full", paths["graphs_dir"], feature_cols, config, log_path)
    if not full_graphs:
        log_line(log_path, "Final GAT full-dataset graph build produced zero graphs.")
        return None
    loader = GeoDataLoader(full_graphs, batch_size=int(config["batch_size"]), shuffle=False)
    model = instantiate_gat_model(config, int(full_graphs[0].x.shape[1]), device)
    load_state_dict_safely(model, checkpoint_path, device)
    y_true, y_prob = predict_geo_loader(model, loader, device)
    metrics = classification_metrics(y_true, y_prob, threshold=threshold)
    avg_batch_latency_ms, ms_per_event = measure_geo_latency_per_event(model, next(iter(loader)), device)
    metrics["avg_batch_latency_ms"] = avg_batch_latency_ms
    metrics["ms_per_event"] = ms_per_event
    baseline_metrics = candidate.get("paired_lstm_metrics")
    proxy_ltss = None
    if baseline_metrics is not None:
        baseline_f1 = max(float(baseline_metrics.get("f1", 0.0)), 1e-8)
        proxy_ltss = 100.0 * (float(metrics.get("f1", 0.0)) - float(baseline_metrics.get("f1", 0.0))) / baseline_f1
    target_report = build_target_report(metrics, proxy_ltss, baseline_metrics is not None)
    predictions = build_graph_prediction_frame(df_full_scaled, full_graphs, y_prob, threshold)
    predictions.to_parquet(paths["full_gat_predictions_path"], index=False)
    visualize_full_graph_sample_time_space(full_graphs, paths["final_graph_visual_path"], paths["final_graph_edges_path"], log_path)
    result = {
        "trial_index": int(candidate["trial_index"]),
        "score": float(candidate["score"]),
        "config": config,
        "checkpoint_file": checkpoint_path.name,
        "graph_cache_file": cache_path.name,
        "metrics": metrics,
        "target_report": target_report,
        "proxy_ltss_improvement_pct": proxy_ltss,
        "graphs_built": int(len(full_graphs)),
        "predictions_rows": int(len(predictions)),
        "prediction_file": paths["full_gat_predictions_path"].name,
        "graph_visual_file": paths["final_graph_visual_path"].name,
        "graph_edges_file": paths["final_graph_edges_path"].name,
    }
    log_json(log_path, {"final_model": "GAT", "result": result})
    return result


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    paths = build_final_paths(base_dir, args.project_path)
    final_log_path = paths["final_log_path"]
    final_log_path.write_text("", encoding="utf-8")
    device = ensure_cuda_device()
    set_seed(args.seed)
    best_gat_results_path = resolve_json_path(base_dir, args.best_gat_results_path, BEST_GAT_RESULTS_FILENAME, paths["artifacts_dir"])
    best_lstm_results_path = resolve_json_path(base_dir, args.best_lstm_results_path, BEST_LSTM_RESULTS_FILENAME, paths["artifacts_dir"])
    best_gat = normalize_candidate(load_json(best_gat_results_path), "GAT", paths["models_dir"])
    best_lstm = normalize_candidate(load_json(best_lstm_results_path), "LSTM", paths["models_dir"])
    best_gat["paired_lstm_metrics"] = best_lstm["result"]["metrics"]
    log_line(final_log_path, f"DEVICE: {device}")
    log_line(final_log_path, f"PYG_AVAILABLE: {main2.PYG_AVAILABLE}")
    log_line(final_log_path, f"Loaded best GAT results from {best_gat_results_path.name}")
    log_line(final_log_path, f"Loaded best LSTM results from {best_lstm_results_path.name}")
    log_json(final_log_path, {"hardcoded_targets": IDEAL_TARGETS})
    log_json(final_log_path, {"selected_best_lstm": sanitize_for_log(best_lstm)})
    log_json(final_log_path, {"selected_best_gat": sanitize_for_log(best_gat)})
    df_clean = load_or_prepare_clean_dataframe(args, paths, final_log_path)
    df_labeled = load_or_prepare_labeled_dataframe(df_clean, args, paths, final_log_path)
    feature_cols = [column for column in BASE_FEATURE_COLS if column in df_labeled.columns]
    train_df, val_df, test_df, scaler = load_or_prepare_splits(df_labeled, args, paths, feature_cols, final_log_path)
    best_lstm = maybe_rebuild_checkpoint(best_lstm if args.run_lstm else None, "LSTM", train_df, val_df, test_df, feature_cols, device, paths, final_log_path)
    best_gat = maybe_rebuild_checkpoint(best_gat if args.run_gat else None, "GAT", train_df, val_df, test_df, feature_cols, device, paths, final_log_path)
    if best_gat is not None and best_lstm is not None:
        best_gat["paired_lstm_metrics"] = best_lstm["result"]["metrics"]
    df_full_scaled = build_scaled_full_dataframe(df_labeled, scaler, feature_cols)
    visualize_all_processed_points_3d(df_labeled, paths["final_points_visual_path"], args.all_point_chunk_size, final_log_path)
    final_results: dict[str, Any] = {
        "hardcoded_targets": IDEAL_TARGETS,
        "selected_inputs": {
            "best_lstm_results_file": best_lstm_results_path.name,
            "best_gat_results_file": best_gat_results_path.name,
        },
        "full_dataset": {
            "rows": int(len(df_labeled)),
            "positive_rate": float(df_labeled[TARGET_COL].mean()),
            "feature_cols": feature_cols,
            "points_visual_file": paths["final_points_visual_path"].name,
            "log_file": final_log_path.name,
        },
    }
    if args.run_lstm:
        final_results["lstm"] = run_final_lstm(best_lstm, df_full_scaled, feature_cols, device, paths, final_log_path)
    if args.run_gat:
        if best_gat is not None and final_results.get("lstm") is not None:
            best_gat["paired_lstm_metrics"] = final_results["lstm"]["metrics"]
        final_results["gat"] = run_final_gat(best_gat, df_full_scaled, feature_cols, device, paths, final_log_path)
    paths["final_summary_path"].write_text(json.dumps(final_results, indent=2, default=str), encoding="utf-8")
    log_json(final_log_path, {"final_results": final_results})
    print(json.dumps(sanitize_for_log(final_results), indent=2, default=str))


if __name__ == "__main__":
    main()
