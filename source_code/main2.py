from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    import kagglehub
except Exception:
    kagglehub = None

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as GeoDataLoader
    from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
    PYG_AVAILABLE = True
    PYG_IMPORT_ERROR = None
except Exception as exc:
    Data = None
    GeoDataLoader = None
    GATv2Conv = None
    global_max_pool = None
    global_mean_pool = None
    PYG_AVAILABLE = False
    PYG_IMPORT_ERROR = exc

from hp import DEFAULT_HYPERPARAMETER_SET, HYPERPARAMETER_SETS

warnings.filterwarnings("ignore")

KAGGLE_DATASET_ID = "alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023"
TIME_COL = "time"
LAT_COL = "latitude"
LON_COL = "longitude"
DEPTH_COL = "depth"
MAG_COL = "magnitudo"
SIG_COL = "significance"
TARGET_COL = "target"
TIME_DIFF_COL = "time_diff"
DIST_PREV_COL = "dist_prev"
BASE_FEATURE_COLS = [LAT_COL, LON_COL, DEPTH_COL, MAG_COL, DIST_PREV_COL, TIME_DIFF_COL, SIG_COL]
COLUMN_ALIASES = {
    TIME_COL: [TIME_COL, "timestamp", "event_time", "datetime"],
    LAT_COL: [LAT_COL, "lat"],
    LON_COL: [LON_COL, "lon", "lng"],
    DEPTH_COL: [DEPTH_COL, "depth_km"],
    MAG_COL: [MAG_COL, "magnitude", "mag"],
    SIG_COL: [SIG_COL, "sig"],
    "date": ["date"],
}
IDEAL_TARGETS = {
    "precision": 0.851,
    "recall": 0.838,
    "f1": 0.839,
    "auc_roc": 0.758,
    "tpr_at_20_fpr": 0.90,
    "ms_per_event_max": 100.0,
    "ltss_proxy_min_pct": 4.0,
}


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
    parser = argparse.ArgumentParser(description="Earthquake foreshock prediction pipeline")
    parser.add_argument("--project-path", default=None)
    parser.add_argument("--raw-csv-path", default=os.environ.get("RAW_CSV_PATH"))
    parser.add_argument("--run-lstm", type=str2bool, default=True)
    parser.add_argument("--run-gat", type=str2bool, default=True)
    parser.add_argument("--use-sample-for-debug", type=str2bool, default=False)
    parser.add_argument("--debug-n-rows", type=int, default=400_000)
    parser.add_argument("--max-hp-runs", type=int, default=len(HYPERPARAMETER_SETS))
    parser.add_argument("--mainshock-mag-threshold", type=float, default=5.5)
    parser.add_argument("--lead-days", type=int, default=30)
    parser.add_argument("--spatial-radius-km", type=float, default=100.0)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.075)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual-sample-size", type=int, default=10000)
    return parser.parse_args()


def ensure_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this pipeline. CPU execution is disabled.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_for_log(payload: Any) -> Any:
    if isinstance(payload, dict):
        sanitized = {}
        for key, value in payload.items():
            if str(key).endswith("_path"):
                continue
            sanitized[key] = sanitize_for_log(value)
        return sanitized
    if isinstance(payload, (list, tuple)):
        return [sanitize_for_log(item) for item in payload]
    if isinstance(payload, Path):
        return payload.name
    return payload


def log_line(log_path: Path | None, message: str) -> None:
    print(message)
    if log_path is None:
        return
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def log_json(log_path: Path | None, payload: Any) -> None:
    log_line(log_path, json.dumps(sanitize_for_log(payload), indent=2, default=str))


def prepare_trial_log(logs_dir: Path, trial_index: int) -> Path:
    log_path = logs_dir / f"log{trial_index}.log"
    log_path.write_text("", encoding="utf-8")
    return log_path


def build_paths(base_dir: Path, project_path: str | None) -> dict[str, Path]:
    artifacts_dir = Path(project_path).expanduser().resolve() if project_path else (base_dir / "artifacts").resolve()
    logs_dir = base_dir / "logs"
    visualizations_dir = base_dir / "visualizations"
    models_dir = artifacts_dir / "models"
    graphs_dir = artifacts_dir / "graphs"
    for path in [artifacts_dir, logs_dir, visualizations_dir, models_dir, graphs_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "base_dir": base_dir,
        "artifacts_dir": artifacts_dir,
        "logs_dir": logs_dir,
        "visualizations_dir": visualizations_dir,
        "models_dir": models_dir,
        "graphs_dir": graphs_dir,
        "clean_parquet": artifacts_dir / "earthquake_cleaned.parquet",
        "labeled_parquet": artifacts_dir / "earthquake_labeled.parquet",
        "train_parquet": artifacts_dir / "train_df.parquet",
        "val_parquet": artifacts_dir / "val_df.parquet",
        "test_parquet": artifacts_dir / "test_df.parquet",
        "scaler_path": artifacts_dir / "feature_scaler.joblib",
        "best_summary_path": artifacts_dir / "best_results.json",
        "best_lstm_summary_path": artifacts_dir / "best_lstm_results.json",
        "best_gat_summary_path": artifacts_dir / "best_gat_results.json",
        "all_results_path": artifacts_dir / "all_results.json",
        "final_full_run_path": artifacts_dir / "final_full_run.json",
        "full_lstm_predictions_path": artifacts_dir / "full_lstm_predictions.parquet",
        "full_gat_predictions_path": artifacts_dir / "full_gat_predictions.parquet",
    }


def resolve_column(df_in: pd.DataFrame, aliases: list[str]) -> str | None:
    lower_map = {column.lower(): column for column in df_in.columns}
    for name in aliases:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def standardize_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for canonical_name, aliases in COLUMN_ALIASES.items():
        resolved = resolve_column(df_in, aliases)
        if resolved is not None and resolved != canonical_name:
            rename_map[resolved] = canonical_name
    return df_in.rename(columns=rename_map).copy()


def ensure_datetime_time(df_in: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    df_out = df_in.copy()
    if time_col not in df_out.columns:
        raise KeyError(f"Missing required time column: {time_col}")
    series = df_out[time_col]
    if is_datetime64_any_dtype(series):
        if getattr(series.dt, "tz", None) is not None:
            df_out[time_col] = series.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            df_out[time_col] = series
    elif is_numeric_dtype(series):
        sample = series.dropna()
        if sample.empty:
            raise ValueError(f"Column '{time_col}' contains no parseable values")
        median_value = sample.median()
        unit = "ms" if median_value > 1e11 else "s"
        df_out[time_col] = pd.to_datetime(series, unit=unit, errors="coerce", utc=True).dt.tz_localize(None)
    else:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        if parsed.notna().sum() < max(10, int(0.5 * len(df_out))):
            parsed = pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="ms", errors="coerce", utc=True)
        df_out[time_col] = parsed.dt.tz_localize(None)
    df_out = df_out.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return df_out


def haversine_np(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def compute_prev_distance_km(df_in: pd.DataFrame) -> np.ndarray:
    lat_prev = df_in[LAT_COL].shift(1).to_numpy()
    lon_prev = df_in[LON_COL].shift(1).to_numpy()
    lat_cur = df_in[LAT_COL].to_numpy()
    lon_cur = df_in[LON_COL].to_numpy()
    distances = np.zeros(len(df_in), dtype=np.float32)
    valid = ~pd.isna(lat_prev) & ~pd.isna(lon_prev)
    distances[valid] = haversine_np(lat_prev[valid], lon_prev[valid], lat_cur[valid], lon_cur[valid]).astype(np.float32)
    return distances


def preprocess_earthquake_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = standardize_columns(df_in)
    required = [TIME_COL, LAT_COL, LON_COL, DEPTH_COL, MAG_COL]
    missing = [column for column in required if column not in df_out.columns]
    if missing:
        raise KeyError(f"Missing required columns after standardization: {missing}")
    if SIG_COL not in df_out.columns:
        df_out[SIG_COL] = 0.0
    for column in [LAT_COL, LON_COL, DEPTH_COL, MAG_COL, SIG_COL]:
        df_out[column] = pd.to_numeric(df_out[column], errors="coerce")
    df_out = df_out.drop_duplicates()
    df_out = df_out.dropna(subset=[TIME_COL, LAT_COL, LON_COL, DEPTH_COL, MAG_COL]).copy()
    df_out = df_out[df_out[MAG_COL] >= 0].copy()
    df_out = ensure_datetime_time(df_out, TIME_COL)
    df_out = df_out.drop(columns=["date"], errors="ignore")
    df_out[TIME_DIFF_COL] = df_out[TIME_COL].diff().dt.total_seconds().fillna(0).astype(np.float32)
    df_out[DIST_PREV_COL] = compute_prev_distance_km(df_out)
    for base_column in [LAT_COL, LON_COL, DEPTH_COL, MAG_COL, DIST_PREV_COL, TIME_DIFF_COL]:
        df_out[f"{base_column}_raw"] = df_out[base_column]
    return df_out.reset_index(drop=True)


def enforce_cleaning_guarantees(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    df_out[MAG_COL] = pd.to_numeric(df_out[MAG_COL], errors="coerce")
    df_out = df_out.dropna(subset=[MAG_COL])
    df_out = df_out[df_out[MAG_COL] >= 0].copy()
    df_out = ensure_datetime_time(df_out, TIME_COL)
    for column in [LAT_COL, LON_COL, DEPTH_COL, MAG_COL, SIG_COL]:
        df_out[column] = pd.to_numeric(df_out[column], errors="coerce")
    df_out = df_out.dropna(subset=[LAT_COL, LON_COL, DEPTH_COL, MAG_COL]).copy()
    if DIST_PREV_COL not in df_out.columns:
        df_out[DIST_PREV_COL] = compute_prev_distance_km(df_out)
    if TIME_DIFF_COL not in df_out.columns:
        df_out[TIME_DIFF_COL] = df_out[TIME_COL].diff().dt.total_seconds().fillna(0).astype(np.float32)
    for base_column in [LAT_COL, LON_COL, DEPTH_COL, MAG_COL, DIST_PREV_COL, TIME_DIFF_COL]:
        raw_name = f"{base_column}_raw"
        if raw_name not in df_out.columns:
            df_out[raw_name] = df_out[base_column]
    return df_out.reset_index(drop=True)


def find_csv_in_dir(dir_path: Path) -> Path:
    candidates = [path for path in dir_path.iterdir() if path.suffix.lower() == ".csv"]
    if not candidates:
        raise FileNotFoundError(f"No CSV file found in {dir_path}")
    candidates.sort(key=lambda path: (0 if "earth" in path.name.lower() else 1, len(path.name)))
    return candidates[0]


def load_raw_dataframe(raw_csv_path: str | None) -> pd.DataFrame:
    if raw_csv_path:
        csv_path = Path(raw_csv_path).expanduser()
        if not csv_path.exists():
            raise FileNotFoundError(f"Raw CSV path does not exist: {csv_path}")
        return pd.read_csv(csv_path)
    if kagglehub is None:
        raise RuntimeError("kagglehub is not installed and no local raw CSV path was provided.")
    download_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))
    csv_path = find_csv_in_dir(download_path)
    return pd.read_csv(csv_path)


def save_preprocessing_visualizations(df_in: pd.DataFrame, visualizations_dir: Path, sample_size: int, log_path: Path) -> None:
    if df_in.empty:
        return
    sample_n = min(sample_size, len(df_in))
    sample_df = df_in.sample(sample_n, random_state=42) if len(df_in) > sample_n else df_in.copy()
    figure = plt.figure(figsize=(8, 5))
    plt.hist(sample_df[MAG_COL], bins=50)
    plt.title("Magnitude Distribution")
    plt.xlabel(MAG_COL)
    plt.ylabel("Count")
    hist_path = visualizations_dir / "magnitude_distribution.png"
    figure.savefig(hist_path, bbox_inches="tight")
    plt.close(figure)
    figure = plt.figure(figsize=(8, 5))
    scatter = plt.scatter(sample_df[LON_COL], sample_df[LAT_COL], c=sample_df[MAG_COL], s=8)
    plt.title("Spatial Distribution")
    plt.xlabel(LON_COL)
    plt.ylabel(LAT_COL)
    plt.colorbar(scatter, label=MAG_COL)
    spatial_path = visualizations_dir / "spatial_distribution.png"
    figure.savefig(spatial_path, bbox_inches="tight")
    plt.close(figure)
    figure = plt.figure(figsize=(10, 5))
    temporal_series = df_in.set_index(TIME_COL).resample("MS").size()
    temporal_series.plot()
    plt.title("Monthly Earthquake Counts")
    plt.xlabel(TIME_COL)
    plt.ylabel("Count")
    temporal_path = visualizations_dir / "monthly_activity.png"
    figure.savefig(temporal_path, bbox_inches="tight")
    plt.close(figure)
    log_line(log_path, "Saved preprocessing visualizations")


def exact_binary_foreshock_labels(df_in: pd.DataFrame, mainshock_mag_threshold: float, lead_days: int, spatial_radius_km: float, log_path: Path) -> pd.DataFrame:
    out = df_in.copy().reset_index(drop=True)
    labels = np.zeros(len(out), dtype=np.uint8)
    times_ns = out[TIME_COL].astype("int64").to_numpy()
    lat = out[f"{LAT_COL}_raw"].to_numpy(dtype=np.float64)
    lon = out[f"{LON_COL}_raw"].to_numpy(dtype=np.float64)
    mag = out[f"{MAG_COL}_raw"].to_numpy(dtype=np.float32)
    mainshock_indices = np.where(mag >= mainshock_mag_threshold)[0]
    lead_ns = int(lead_days * 24 * 3600 * 1e9)
    for mainshock_index in tqdm(mainshock_indices, desc="Creating exact foreshock labels"):
        mainshock_time = times_ns[mainshock_index]
        start_index = np.searchsorted(times_ns, mainshock_time - lead_ns, side="left")
        if start_index >= mainshock_index:
            continue
        candidate_indices = np.arange(start_index, mainshock_index, dtype=np.int64)
        if candidate_indices.size == 0:
            continue
        lat_ms = lat[mainshock_index]
        lon_ms = lon[mainshock_index]
        lat_deg = spatial_radius_km / 111.0
        cos_lat = max(np.cos(np.radians(lat_ms)), 1e-6)
        lon_deg = spatial_radius_km / (111.0 * cos_lat)
        candidate_lat = lat[candidate_indices]
        candidate_lon = lon[candidate_indices]
        lon_diff = ((candidate_lon - lon_ms + 180.0) % 360.0) - 180.0
        bbox_mask = (candidate_lat >= lat_ms - lat_deg) & (candidate_lat <= lat_ms + lat_deg) & (np.abs(lon_diff) <= lon_deg)
        candidate_indices = candidate_indices[bbox_mask]
        if candidate_indices.size == 0:
            continue
        distances = haversine_np(lat[candidate_indices], lon[candidate_indices], lat_ms, lon_ms)
        labels[candidate_indices[distances <= spatial_radius_km]] = 1
    out[TARGET_COL] = labels.astype(np.int8)
    log_line(log_path, f"Generated labels with positive rate {float(out[TARGET_COL].mean()):.6f}")
    return out


def temporal_split(df_in: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(df_in)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_df = df_in.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_in.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_in.iloc[val_end:].copy().reset_index(drop=True)
    return train_df, val_df, test_df


def load_or_prepare_clean_dataframe(args: argparse.Namespace, paths: dict[str, Path], log_path: Path) -> pd.DataFrame:
    if paths["clean_parquet"].exists():
        df = pd.read_parquet(paths["clean_parquet"])
        df = enforce_cleaning_guarantees(df)
        df.to_parquet(paths["clean_parquet"], index=False)
        log_line(log_path, "Loaded cached cleaned dataframe")
        return df
    raw_df = load_raw_dataframe(args.raw_csv_path)
    if args.use_sample_for_debug and len(raw_df) > args.debug_n_rows:
        raw_df = raw_df.iloc[:args.debug_n_rows].copy()
        log_line(log_path, f"Debug mode enabled. Using first {len(raw_df):,} rows")
    df = preprocess_earthquake_dataframe(raw_df)
    df.to_parquet(paths["clean_parquet"], index=False)
    log_line(log_path, "Saved cleaned dataframe")
    return df


def load_or_prepare_labeled_dataframe(df_in: pd.DataFrame, args: argparse.Namespace, paths: dict[str, Path], log_path: Path) -> pd.DataFrame:
    if paths["labeled_parquet"].exists():
        df = pd.read_parquet(paths["labeled_parquet"])
        if TARGET_COL in df.columns and float(df[TARGET_COL].mean()) >= 0.0:
            log_line(log_path, "Loaded cached labeled dataframe")
            return df
    df = exact_binary_foreshock_labels(
        df_in,
        mainshock_mag_threshold=args.mainshock_mag_threshold,
        lead_days=args.lead_days,
        spatial_radius_km=args.spatial_radius_km,
        log_path=log_path,
    )
    df.to_parquet(paths["labeled_parquet"], index=False)
    log_line(log_path, "Saved labeled dataframe")
    return df


def load_or_prepare_splits(df_in: pd.DataFrame, args: argparse.Namespace, paths: dict[str, Path], feature_cols: list[str], log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    split_paths = [paths["train_parquet"], paths["val_parquet"], paths["test_parquet"], paths["scaler_path"]]
    if all(path.exists() for path in split_paths):
        train_df = pd.read_parquet(paths["train_parquet"])
        val_df = pd.read_parquet(paths["val_parquet"])
        test_df = pd.read_parquet(paths["test_parquet"])
        scaler = joblib.load(paths["scaler_path"])
        log_line(log_path, "Loaded cached train/val/test splits and scaler")
        return train_df, val_df, test_df, scaler
    train_df, val_df, test_df = temporal_split(df_in, args.train_ratio, args.val_ratio)
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    train_df.to_parquet(paths["train_parquet"], index=False)
    val_df.to_parquet(paths["val_parquet"], index=False)
    test_df.to_parquet(paths["test_parquet"], index=False)
    joblib.dump(scaler, paths["scaler_path"])
    log_line(log_path, "Saved train/val/test splits and scaler")
    return train_df, val_df, test_df, scaler


class SequenceWindowDataset(Dataset):
    def __init__(self, df_in: pd.DataFrame, feature_cols: list[str], target_col: str, window_size: int, stride: int):
        self.features = df_in[feature_cols].to_numpy(dtype=np.float32)
        self.labels = df_in[target_col].to_numpy(dtype=np.float32)
        self.window_size = window_size
        self.end_indices = np.arange(window_size, len(df_in), stride, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        end_index = int(self.end_indices[index])
        start_index = end_index - self.window_size
        x = self.features[start_index:end_index]
        y = self.labels[end_index]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

    @property
    def window_labels(self) -> np.ndarray:
        return self.labels[self.end_indices]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, bidirectional: bool):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(output_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = self.dropout(self.norm(output[:, -1, :]))
        return self.fc(last_hidden).squeeze(-1)


class TemporalConvEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding="same")
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding="same")
        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(batch)
        outputs = []
        start_index = 0
        for count in counts.tolist():
            sample = x[start_index:start_index + count]
            sample = sample.transpose(0, 1).unsqueeze(0)
            sample = F.relu(self.conv1(sample))
            sample = F.relu(self.conv2(sample))
            sample = self.norm(sample)
            outputs.append(sample.squeeze(0).transpose(0, 1))
            start_index += count
        return torch.cat(outputs, dim=0)


class TemporalGATClassifier(nn.Module):
    def __init__(self, input_dim: int, tcn_hidden: int, gat_hidden: int, heads: int, dropout: float, edge_dim: int = 2):
        super().__init__()
        self.temporal_encoder = TemporalConvEncoder(input_dim, tcn_hidden)
        self.gat1 = GATv2Conv(tcn_hidden, gat_hidden, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.gat2 = GATv2Conv(gat_hidden * heads, gat_hidden, heads=1, concat=True, dropout=dropout, edge_dim=edge_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(gat_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.temporal_encoder(data.x, data.batch)
        x = F.elu(self.gat1(x, data.edge_index, data.edge_attr))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, data.edge_index, data.edge_attr))
        pooled = torch.cat([global_mean_pool(x, data.batch), global_max_pool(x, data.batch)], dim=1)
        return self.classifier(pooled).squeeze(-1)


def compute_pos_weight(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    labels = labels.astype(np.float32)
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    raw_weight = negatives / max(positives, 1.0)
    pos_weight = min(raw_weight, 3.0)
    return torch.tensor([pos_weight], dtype=torch.float32, device=device)


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def tpr_at_target_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float = 0.20) -> dict[str, float]:
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        best_index = int(np.argmin(np.abs(fpr - target_fpr)))
        return {
            "target_fpr": float(target_fpr),
            "actual_fpr": float(fpr[best_index]),
            "tpr": float(tpr[best_index]),
            "threshold": float(thresholds[best_index]),
        }
    except Exception:
        return {
            "target_fpr": float(target_fpr),
            "actual_fpr": float("nan"),
            "tpr": float("nan"),
            "threshold": float("nan"),
        }


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auc_roc"] = float("nan")
    roc_point = tpr_at_target_fpr(y_true, y_prob, target_fpr=0.20)
    metrics["tpr_at_20_fpr"] = roc_point["tpr"]
    metrics["actual_fpr_at_selected_roc_point"] = roc_point["actual_fpr"]
    metrics["roc_threshold_at_20_fpr"] = roc_point["threshold"]
    return metrics


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    threshold_grid = np.round(np.arange(0.05, 0.91, 0.05), 2)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in threshold_grid:
        current_f1 = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def load_state_dict_safely(model: nn.Module, model_path: Path, device: torch.device) -> None:
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)


def predict_loader(model: nn.Module, data_loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            logits = model(x_batch)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(y_batch.numpy())
    logits = np.concatenate(all_logits) if all_logits else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    return labels, sigmoid_np(logits)


def predict_geo_loader(model: nn.Module, data_loader: GeoDataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            logits = model(batch)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(batch.y.detach().cpu().numpy())
    logits = np.concatenate(all_logits) if all_logits else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    return labels, sigmoid_np(logits)


def train_binary_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: dict[str, Any], model_path: Path, device: torch.device, log_path: Path, model_name: str, trial_index: int, pos_weight_tensor: torch.Tensor) -> tuple[list[dict[str, Any]], float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    best_val_f1 = -1.0
    best_threshold = 0.5
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        train_losses = []
        for x_batch, y_batch in tqdm(train_loader, desc=f"{model_name} trial {trial_index} epoch {epoch}/{config['epochs']}", leave=False):
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_y, val_prob = predict_loader(model, val_loader, device)
        best_threshold_epoch, best_f1_epoch = find_best_threshold(val_y, val_prob)
        val_metrics = classification_metrics(val_y, val_prob, threshold=best_threshold_epoch)
        epoch_record = {
            "trial_index": trial_index,
            "model": model_name,
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses) if train_losses else np.nan),
            "val_threshold": best_threshold_epoch,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_record)
        log_json(log_path, epoch_record)
        if best_f1_epoch > best_val_f1:
            best_val_f1 = best_f1_epoch
            best_threshold = best_threshold_epoch
            torch.save(model.state_dict(), model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(config["patience"]):
            log_line(log_path, f"Early stopping for {model_name} on trial {trial_index} at epoch {epoch}")
            break
    load_state_dict_safely(model, model_path, device)
    return history, best_threshold


def train_geo_binary_model(model: nn.Module, train_loader: GeoDataLoader, val_loader: GeoDataLoader, config: dict[str, Any], model_path: Path, device: torch.device, log_path: Path, model_name: str, trial_index: int, pos_weight_tensor: torch.Tensor) -> tuple[list[dict[str, Any]], float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    best_val_f1 = -1.0
    best_threshold = 0.5
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"{model_name} trial {trial_index} epoch {epoch}/{config['epochs']}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_y, val_prob = predict_geo_loader(model, val_loader, device)
        best_threshold_epoch, best_f1_epoch = find_best_threshold(val_y, val_prob)
        val_metrics = classification_metrics(val_y, val_prob, threshold=best_threshold_epoch)
        epoch_record = {
            "trial_index": trial_index,
            "model": model_name,
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses) if train_losses else np.nan),
            "val_threshold": best_threshold_epoch,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_record)
        log_json(log_path, epoch_record)
        if best_f1_epoch > best_val_f1:
            best_val_f1 = best_f1_epoch
            best_threshold = best_threshold_epoch
            torch.save(model.state_dict(), model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(config["patience"]):
            log_line(log_path, f"Early stopping for {model_name} on trial {trial_index} at epoch {epoch}")
            break
    load_state_dict_safely(model, model_path, device)
    return history, best_threshold


def measure_latency_per_event(model: nn.Module, example_batch: torch.Tensor, device: torch.device, n_runs: int = 30) -> tuple[float, float]:
    model.eval()
    x_batch = example_batch.to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(x_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        run_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(x_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            run_times.append(end - start)
    avg_batch_ms = float(np.mean(run_times) * 1000.0)
    return avg_batch_ms, avg_batch_ms / max(len(example_batch), 1)


def measure_geo_latency_per_event(model: nn.Module, example_batch: Data, device: torch.device, n_runs: int = 20) -> tuple[float, float]:
    model.eval()
    batch = example_batch.to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        run_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            run_times.append(end - start)
    avg_batch_ms = float(np.mean(run_times) * 1000.0)
    return avg_batch_ms, avg_batch_ms / max(int(batch.num_graphs), 1)


def build_temporal_encoding(delta_seconds: np.ndarray, graph_window_days: int) -> np.ndarray:
    delta_days = delta_seconds / 86400.0
    log_delta = np.log1p(np.clip(delta_seconds, a_min=0, a_max=None))
    sin_7 = np.sin(2 * np.pi * delta_days / 7.0)
    cos_7 = np.cos(2 * np.pi * delta_days / 7.0)
    sin_30 = np.sin(2 * np.pi * delta_days / 30.0)
    cos_30 = np.cos(2 * np.pi * delta_days / 30.0)
    return np.stack([log_delta, delta_days / max(graph_window_days, 1), sin_7, cos_7, sin_30, cos_30], axis=1).astype(np.float32)


def pairwise_haversine_km(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)[:, None]
    lon = np.radians(lon_deg)[:, None]
    dlat = lat.T - lat
    dlon = lon.T - lon
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def create_graph_samples(df_split: pd.DataFrame, feature_cols: list[str], config: dict[str, Any], desc: str) -> list[Data]:
    if not PYG_AVAILABLE:
        raise RuntimeError(f"PyTorch Geometric is unavailable: {PYG_IMPORT_ERROR}")
    feature_array = df_split[feature_cols].to_numpy(dtype=np.float32)
    labels = df_split[TARGET_COL].to_numpy(dtype=np.float32)
    times_ns = df_split[TIME_COL].astype("int64").to_numpy()
    lat = df_split[f"{LAT_COL}_raw"].to_numpy(dtype=np.float64)
    lon = df_split[f"{LON_COL}_raw"].to_numpy(dtype=np.float64)
    mag = df_split[f"{MAG_COL}_raw"].to_numpy(dtype=np.float32)
    graph_window_ns = int(config["graph_window_days"] * 24 * 3600 * 1e9)
    edge_temporal_ns = int(config["edge_temporal_days"] * 24 * 3600 * 1e9)
    graphs = []
    center_indices = np.arange(int(config["min_nodes"]), len(df_split), int(config["graph_stride"]), dtype=np.int64)
    for center_index in tqdm(center_indices, desc=f"Building {desc}"):
        center_time = times_ns[center_index]
        start_index = np.searchsorted(times_ns, center_time - graph_window_ns, side="left")
        candidate_indices = np.arange(start_index, center_index + 1, dtype=np.int64)
        if candidate_indices.size < int(config["min_nodes"]):
            continue
        center_distances = haversine_np(lat[candidate_indices], lon[candidate_indices], lat[center_index], lon[center_index])
        candidate_indices = candidate_indices[center_distances <= float(config["local_subgraph_radius_km"])]
        if candidate_indices.size < int(config["min_nodes"]):
            continue
        if candidate_indices.size > int(config["max_nodes"]):
            candidate_indices = candidate_indices[-int(config["max_nodes"]):]
        node_features = feature_array[candidate_indices]
        delta_seconds = ((center_time - times_ns[candidate_indices]) / 1e9).astype(np.float32)
        temporal_features = build_temporal_encoding(delta_seconds, int(config["graph_window_days"]))
        x_array = np.concatenate([node_features, temporal_features], axis=1).astype(np.float32)
        candidate_lat = lat[candidate_indices]
        candidate_lon = lon[candidate_indices]
        pair_distances = pairwise_haversine_km(candidate_lat, candidate_lon)
        pair_delta_ns = times_ns[candidate_indices][None, :] - times_ns[candidate_indices][:, None]
        forward_mask = (pair_delta_ns >= 0) & (pair_delta_ns <= edge_temporal_ns) & (pair_distances <= float(config["edge_radius_km"]))
        np.fill_diagonal(forward_mask, False)
        source_indices, target_indices = np.where(forward_mask)
        if len(source_indices) == 0:
            continue
        edge_index = np.vstack([np.concatenate([source_indices, target_indices]), np.concatenate([target_indices, source_indices])]).astype(np.int64)
        edge_distance_km_forward = pair_distances[source_indices, target_indices].astype(np.float32)
        edge_time_days_forward = (pair_delta_ns[source_indices, target_indices] / 86400e9).astype(np.float32)
        edge_attr_forward = np.stack([
            (edge_distance_km_forward / max(float(config["edge_radius_km"]), 1e-6)).astype(np.float32),
            (pair_delta_ns[source_indices, target_indices] / max(edge_temporal_ns, 1)).astype(np.float32),
        ], axis=1)
        edge_attr = np.vstack([edge_attr_forward, edge_attr_forward]).astype(np.float32)
        edge_distance_km = np.concatenate([edge_distance_km_forward, edge_distance_km_forward]).astype(np.float32)
        edge_time_days = np.concatenate([edge_time_days_forward, edge_time_days_forward]).astype(np.float32)
        graph = Data(
            x=torch.tensor(x_array, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y=torch.tensor([labels[center_index]], dtype=torch.float32),
        )
        graph.graph_time = int(center_time)
        graph.center_index = int(center_index)
        graph.node_lat = torch.tensor(candidate_lat.astype(np.float32))
        graph.node_lon = torch.tensor(candidate_lon.astype(np.float32))
        graph.node_mag = torch.tensor(mag[candidate_indices].astype(np.float32))
        graph.node_delta_days = torch.tensor((delta_seconds / 86400.0).astype(np.float32))
        graph.edge_distance_km = torch.tensor(edge_distance_km.astype(np.float32))
        graph.edge_time_days = torch.tensor(edge_time_days.astype(np.float32))
        graphs.append(graph)
        limit = config.get("max_graphs")
        if limit is not None and len(graphs) >= int(limit):
            break
    return graphs


def graph_cache_path(graphs_dir: Path, split_name: str, config: dict[str, Any]) -> Path:
    signature = hashlib.md5(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return graphs_dir / f"{split_name}_{signature}.pt"


def load_or_build_graphs(df_split: pd.DataFrame, split_name: str, graphs_dir: Path, feature_cols: list[str], config: dict[str, Any], log_path: Path | None) -> list[Data]:
    cache_config = dict(config)
    cache_path = graph_cache_path(graphs_dir, split_name, cache_config)
    if cache_path.exists():
        log_line(log_path, f"Loaded cached {split_name} graphs")
        return torch.load(cache_path)
    runtime_config = dict(config)
    limit_key = f"max_{split_name}_graphs"
    limit_value = runtime_config.get(limit_key)
    if limit_value is not None:
        runtime_config["max_graphs"] = int(limit_value)
    else:
        runtime_config.pop("max_graphs", None)
    graphs = create_graph_samples(df_split, feature_cols, runtime_config, f"{split_name} graphs")
    torch.save(graphs, cache_path)
    log_line(log_path, f"Saved {split_name} graphs")
    return graphs


def select_graph_for_visualization(graphs: list[Data]) -> Data | None:
    if not graphs:
        return None
    def score(graph: Data) -> tuple[int, int, float]:
        node_count = int(graph.x.shape[0])
        edge_count = int(graph.edge_index.shape[1] // 2)
        magnitude_range = float(torch.max(graph.node_mag).item() - torch.min(graph.node_mag).item()) if node_count > 1 else 0.0
        return node_count, edge_count, magnitude_range
    return max(graphs, key=score)


def visualize_graph_sample(graphs: list[Data], visualizations_dir: Path, log_path: Path | None) -> None:
    graph = select_graph_for_visualization(graphs)
    if graph is None:
        return
    longitude = graph.node_lon.detach().cpu().numpy()
    latitude = graph.node_lat.detach().cpu().numpy()
    magnitude = graph.node_mag.detach().cpu().numpy()
    delta_days = graph.node_delta_days.detach().cpu().numpy()
    edge_index = graph.edge_index.detach().cpu().numpy()
    edge_distance_km = graph.edge_distance_km.detach().cpu().numpy()
    edge_time_days = graph.edge_time_days.detach().cpu().numpy()
    unique_mask = edge_index[0] < edge_index[1]
    unique_sources = edge_index[0][unique_mask]
    unique_targets = edge_index[1][unique_mask]
    unique_distances = edge_distance_km[unique_mask]
    unique_times = edge_time_days[unique_mask]
    if unique_sources.size == 0:
        return
    edge_table = pd.DataFrame({
        "source_node": unique_sources.astype(int),
        "target_node": unique_targets.astype(int),
        "source_longitude": longitude[unique_sources],
        "source_latitude": latitude[unique_sources],
        "source_magnitude": magnitude[unique_sources],
        "target_longitude": longitude[unique_targets],
        "target_latitude": latitude[unique_targets],
        "target_magnitude": magnitude[unique_targets],
        "distance_km": unique_distances,
        "time_gap_days": unique_times,
    }).sort_values(["time_gap_days", "distance_km"], ascending=[False, False]).reset_index(drop=True)
    edge_table.to_csv(visualizations_dir / "spatiotemporal_graph_edges.csv", index=False)
    figure = plt.figure(figsize=(13, 10))
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(longitude, latitude, magnitude, c=delta_days, s=36 + 10 * magnitude, alpha=0.95)
    max_edges_to_draw = min(len(edge_table), 120)
    for row in edge_table.head(max_edges_to_draw).itertuples(index=False):
        axis.plot(
            [row.source_longitude, row.target_longitude],
            [row.source_latitude, row.target_latitude],
            [row.source_magnitude, row.target_magnitude],
            alpha=0.25,
            linewidth=0.8,
        )
    max_labels = min(len(edge_table), 16)
    for row in edge_table.head(max_labels).itertuples(index=False):
        x_mid = (row.source_longitude + row.target_longitude) / 2.0
        y_mid = (row.source_latitude + row.target_latitude) / 2.0
        z_mid = (row.source_magnitude + row.target_magnitude) / 2.0
        axis.text(x_mid, y_mid, z_mid, f"d={row.distance_km:.1f}km\nΔt={row.time_gap_days:.2f}d", fontsize=8)
    axis.set_title("3D Spatio-Temporal GAT Graph Sample")
    axis.set_xlabel(LON_COL)
    axis.set_ylabel(LAT_COL)
    axis.set_zlabel(MAG_COL)
    axis.view_init(elev=24, azim=-62)
    try:
        axis.set_box_aspect((max(np.ptp(longitude), 1e-6), max(np.ptp(latitude), 1e-6), max(np.ptp(magnitude), 1e-6)))
    except Exception:
        pass
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.08)
    colorbar.set_label("time from center (days)")
    graph_path = visualizations_dir / "spatiotemporal_graph_sample_3d.png"
    figure.savefig(graph_path, bbox_inches="tight", dpi=220)
    plt.close(figure)
    log_line(log_path, "Saved graph visualization")


def build_sequence_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], config: dict[str, Any], device: torch.device) -> tuple[SequenceWindowDataset, SequenceWindowDataset, SequenceWindowDataset, DataLoader, DataLoader, DataLoader]:
    train_dataset = SequenceWindowDataset(train_df, feature_cols, TARGET_COL, int(config["window_size"]), int(config["window_stride"]))
    val_dataset = SequenceWindowDataset(val_df, feature_cols, TARGET_COL, int(config["window_size"]), int(config["window_stride"]))
    test_dataset = SequenceWindowDataset(test_df, feature_cols, TARGET_COL, int(config["window_size"]), int(config["window_stride"]))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0, pin_memory=pin_memory)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_geo_loaders(train_graphs: list[Data], val_graphs: list[Data], test_graphs: list[Data], batch_size: int) -> tuple[GeoDataLoader, GeoDataLoader, GeoDataLoader]:
    train_loader = GeoDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = GeoDataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def build_full_sequence_loader(df_in: pd.DataFrame, feature_cols: list[str], config: dict[str, Any], device: torch.device) -> tuple[SequenceWindowDataset, DataLoader]:
    dataset = SequenceWindowDataset(df_in, feature_cols, TARGET_COL, int(config["window_size"]), int(config["window_stride"]))
    pin_memory = device.type == "cuda"
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0, pin_memory=pin_memory)
    return dataset, loader


def build_sequence_prediction_frame(df_in: pd.DataFrame, dataset: SequenceWindowDataset, probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    end_indices = dataset.end_indices.astype(np.int64)
    prediction_df = df_in.iloc[end_indices].copy().reset_index(drop=True)
    prediction_df["window_end_index"] = end_indices
    prediction_df["prediction_probability"] = probabilities.astype(np.float32)
    prediction_df["prediction"] = (probabilities >= threshold).astype(np.int8)
    return prediction_df


def build_graph_prediction_frame(df_in: pd.DataFrame, graphs: list[Data], probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    center_indices = np.array([int(graph.center_index) for graph in graphs], dtype=np.int64)
    graph_times = np.array([int(graph.graph_time) for graph in graphs], dtype=np.int64)
    prediction_df = df_in.iloc[center_indices].copy().reset_index(drop=True)
    prediction_df["graph_center_index"] = center_indices
    prediction_df["graph_time_ns"] = graph_times
    prediction_df["prediction_probability"] = probabilities.astype(np.float32)
    prediction_df["prediction"] = (probabilities >= threshold).astype(np.int8)
    return prediction_df


def instantiate_lstm_model(config: dict[str, Any], input_dim: int, device: torch.device) -> LSTMClassifier:
    return LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        bidirectional=bool(config["bidirectional"]),
    ).to(device)


def instantiate_gat_model(config: dict[str, Any], input_dim: int, device: torch.device) -> TemporalGATClassifier:
    return TemporalGATClassifier(
        input_dim=input_dim,
        tcn_hidden=int(config["tcn_hidden"]),
        gat_hidden=int(config["gat_hidden"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
    ).to(device)


def build_full_dataset_gat_config(config: dict[str, Any]) -> dict[str, Any]:
    full_config = dict(config)
    for key in ["max_train_graphs", "max_val_graphs", "max_test_graphs"]:
        if key in full_config:
            full_config[key] = None
    full_config["max_full_graphs"] = None
    return full_config


def run_best_models_on_full_dataset(best_lstm_candidate: dict[str, Any] | None, best_gat_candidate: dict[str, Any] | None, df_labeled: pd.DataFrame, feature_cols: list[str], device: torch.device, paths: dict[str, Path]) -> dict[str, Any]:
    final_summary: dict[str, Any] = {"mode": "full_dataset_inference_after_hyperparameter_search"}
    if best_lstm_candidate is not None:
        lstm_config = dict(best_lstm_candidate["result"]["config"])
        lstm_threshold = float(best_lstm_candidate["result"]["metrics"].get("threshold", 0.5))
        lstm_dataset, lstm_loader = build_full_sequence_loader(df_labeled, feature_cols, lstm_config, device)
        if len(lstm_dataset) > 0:
            lstm_model = instantiate_lstm_model(lstm_config, len(feature_cols), device)
            load_state_dict_safely(lstm_model, Path(best_lstm_candidate["result"]["model_path"]), device)
            y_full, p_full = predict_loader(lstm_model, lstm_loader, device)
            full_metrics = classification_metrics(y_full, p_full, threshold=lstm_threshold)
            example_x, _ = next(iter(lstm_loader))
            avg_batch_latency_ms, ms_per_event = measure_latency_per_event(lstm_model, example_x[: min(256, len(example_x))], device)
            full_metrics["avg_batch_latency_ms"] = avg_batch_latency_ms
            full_metrics["ms_per_event"] = ms_per_event
            prediction_df = build_sequence_prediction_frame(df_labeled, lstm_dataset, p_full, lstm_threshold)
            prediction_df.to_parquet(paths["full_lstm_predictions_path"], index=False)
            final_summary["lstm"] = {
                "source_trial_index": int(best_lstm_candidate["trial_index"]),
                "score": float(best_lstm_candidate["score"]),
                "config": lstm_config,
                "metrics": full_metrics,
                "predictions_rows": int(len(prediction_df)),
                "prediction_file": str(paths["full_lstm_predictions_path"]),
            }
            del lstm_model, lstm_loader, lstm_dataset, prediction_df
            gc.collect()
            torch.cuda.empty_cache()
    if best_gat_candidate is not None:
        gat_config = build_full_dataset_gat_config(best_gat_candidate["result"]["config"])
        gat_threshold = float(best_gat_candidate["result"]["metrics"].get("threshold", 0.5))
        full_graphs = load_or_build_graphs(df_labeled, "full", paths["graphs_dir"], feature_cols, gat_config, None)
        if full_graphs:
            gat_loader = GeoDataLoader(full_graphs, batch_size=int(gat_config["batch_size"]), shuffle=False)
            gat_model = instantiate_gat_model(gat_config, int(full_graphs[0].x.shape[1]), device)
            load_state_dict_safely(gat_model, Path(best_gat_candidate["result"]["model_path"]), device)
            y_full, p_full = predict_geo_loader(gat_model, gat_loader, device)
            full_metrics = classification_metrics(y_full, p_full, threshold=gat_threshold)
            avg_batch_latency_ms, ms_per_event = measure_geo_latency_per_event(gat_model, next(iter(gat_loader)), device)
            full_metrics["avg_batch_latency_ms"] = avg_batch_latency_ms
            full_metrics["ms_per_event"] = ms_per_event
            prediction_df = build_graph_prediction_frame(df_labeled, full_graphs, p_full, gat_threshold)
            prediction_df.to_parquet(paths["full_gat_predictions_path"], index=False)
            final_summary["gat"] = {
                "source_trial_index": int(best_gat_candidate["trial_index"]),
                "score": float(best_gat_candidate["score"]),
                "config": gat_config,
                "metrics": full_metrics,
                "graphs_built": int(len(full_graphs)),
                "predictions_rows": int(len(prediction_df)),
                "prediction_file": str(paths["full_gat_predictions_path"]),
            }
            del gat_model, gat_loader, full_graphs, prediction_df
            gc.collect()
            torch.cuda.empty_cache()
    return final_summary


def build_one_time_graph_visualization(train_df: pd.DataFrame, feature_cols: list[str], visualizations_dir: Path) -> bool:
    if not PYG_AVAILABLE or train_df.empty:
        return False
    visualization_config = dict(DEFAULT_HYPERPARAMETER_SET["gat"])
    visualization_config["max_graphs"] = 96
    sample_rows = min(len(train_df), 25000)
    start_index = max((len(train_df) - sample_rows) // 2, 0)
    visualization_df = train_df.iloc[start_index:start_index + sample_rows].copy().reset_index(drop=True)
    graphs = create_graph_samples(visualization_df, feature_cols, visualization_config, "visualization graph")
    if not graphs:
        return False
    visualize_graph_sample(graphs, visualizations_dir, None)
    del graphs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def score_against_targets(metrics: dict[str, float], proxy_ltss_improvement_pct: float | None) -> float:
    score = 0.0
    score += min(metrics.get("precision", 0.0) / IDEAL_TARGETS["precision"], 1.5)
    score += min(metrics.get("recall", 0.0) / IDEAL_TARGETS["recall"], 1.5)
    score += min(metrics.get("f1", 0.0) / IDEAL_TARGETS["f1"], 1.5)
    score += min(metrics.get("auc_roc", 0.0) / IDEAL_TARGETS["auc_roc"], 1.5)
    score += min(metrics.get("tpr_at_20_fpr", 0.0) / IDEAL_TARGETS["tpr_at_20_fpr"], 1.5)
    score += min(IDEAL_TARGETS["ms_per_event_max"] / max(metrics.get("ms_per_event", IDEAL_TARGETS["ms_per_event_max"]), 1e-8), 1.5)
    if proxy_ltss_improvement_pct is not None:
        score += min(proxy_ltss_improvement_pct / max(IDEAL_TARGETS["ltss_proxy_min_pct"], 1e-8), 1.5)
    return float(score)


def build_target_report(metrics: dict[str, float] | None, proxy_ltss_improvement_pct: float | None = None, require_proxy: bool = False) -> dict[str, Any]:
    if metrics is None:
        return {"checks": {}, "meets_all_targets": False}
    checks = {
        "precision": bool(metrics.get("precision", -np.inf) >= IDEAL_TARGETS["precision"]),
        "recall": bool(metrics.get("recall", -np.inf) >= IDEAL_TARGETS["recall"]),
        "f1": bool(metrics.get("f1", -np.inf) >= IDEAL_TARGETS["f1"]),
        "auc_roc": bool(metrics.get("auc_roc", -np.inf) >= IDEAL_TARGETS["auc_roc"]),
        "tpr_at_20_fpr": bool(metrics.get("tpr_at_20_fpr", -np.inf) >= IDEAL_TARGETS["tpr_at_20_fpr"]),
        "ms_per_event": bool(metrics.get("ms_per_event", np.inf) <= IDEAL_TARGETS["ms_per_event_max"]),
    }
    if require_proxy:
        checks["ltss_proxy_min_pct"] = proxy_ltss_improvement_pct is not None and proxy_ltss_improvement_pct >= IDEAL_TARGETS["ltss_proxy_min_pct"]
    return {
        "checks": checks,
        "proxy_ltss_improvement_pct": proxy_ltss_improvement_pct,
        "meets_all_targets": bool(checks) and all(checks.values()),
    }


def run_lstm_trial(trial_index: int, trial_config: dict[str, Any], train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], device: torch.device, models_dir: Path, log_path: Path) -> dict[str, Any] | None:
    lstm_config = dict(DEFAULT_HYPERPARAMETER_SET["lstm"])
    lstm_config.update(trial_config["lstm"])
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_sequence_loaders(train_df, val_df, test_df, feature_cols, lstm_config, device)
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        log_line(log_path, f"Skipping LSTM trial {trial_index} because one split has zero sequence windows")
        return None
    model = instantiate_lstm_model(lstm_config, len(feature_cols), device)
    pos_weight = compute_pos_weight(train_dataset.window_labels, device)
    model_path = models_dir / f"lstm_trial_{trial_index:03d}.pt"
    history, best_threshold = train_binary_model(model, train_loader, val_loader, lstm_config, model_path, device, log_path, "LSTM", trial_index, pos_weight)
    y_test, p_test = predict_loader(model, test_loader, device)
    metrics = classification_metrics(y_test, p_test, threshold=best_threshold)
    example_batch, _ = next(iter(test_loader))
    avg_batch_latency_ms, ms_per_event = measure_latency_per_event(model, example_batch[: min(256, len(example_batch))], device)
    metrics["avg_batch_latency_ms"] = avg_batch_latency_ms
    metrics["ms_per_event"] = ms_per_event
    result = {"config": lstm_config, "history": history, "metrics": metrics, "model_path": str(model_path)}
    log_json(log_path, {"trial_index": trial_index, "model": "LSTM", "test_metrics": metrics})
    del model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_gat_trial(trial_index: int, trial_config: dict[str, Any], train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], device: torch.device, graphs_dir: Path, models_dir: Path, visualizations_dir: Path, log_path: Path, graph_visual_saved: bool) -> tuple[dict[str, Any] | None, bool]:
    if not PYG_AVAILABLE:
        raise RuntimeError(f"PyTorch Geometric is required for GAT runs and is unavailable: {PYG_IMPORT_ERROR}")
    gat_config = dict(DEFAULT_HYPERPARAMETER_SET["gat"])
    gat_config.update(trial_config["gat"])
    train_graphs = load_or_build_graphs(train_df, "train", graphs_dir, feature_cols, gat_config, log_path)
    val_graphs = load_or_build_graphs(val_df, "val", graphs_dir, feature_cols, gat_config, log_path)
    test_graphs = load_or_build_graphs(test_df, "test", graphs_dir, feature_cols, gat_config, log_path)
    if not graph_visual_saved and train_graphs:
        visualize_graph_sample(train_graphs, visualizations_dir, log_path)
        graph_visual_saved = True
    if len(train_graphs) == 0 or len(val_graphs) == 0 or len(test_graphs) == 0:
        log_line(log_path, f"Skipping GAT trial {trial_index} because one split has zero graphs")
        return None, graph_visual_saved
    train_loader, val_loader, test_loader = build_geo_loaders(train_graphs, val_graphs, test_graphs, int(gat_config["batch_size"]))
    graph_labels = np.array([float(graph.y.item()) for graph in train_graphs], dtype=np.float32)
    pos_weight = compute_pos_weight(graph_labels, device)
    model = instantiate_gat_model(gat_config, int(train_graphs[0].x.shape[1]), device)
    model_path = models_dir / f"gat_trial_{trial_index:03d}.pt"
    history, best_threshold = train_geo_binary_model(model, train_loader, val_loader, gat_config, model_path, device, log_path, "GAT", trial_index, pos_weight)
    y_test, p_test = predict_geo_loader(model, test_loader, device)
    metrics = classification_metrics(y_test, p_test, threshold=best_threshold)
    avg_batch_latency_ms, ms_per_event = measure_geo_latency_per_event(model, next(iter(test_loader)), device)
    metrics["avg_batch_latency_ms"] = avg_batch_latency_ms
    metrics["ms_per_event"] = ms_per_event
    result = {"config": gat_config, "history": history, "metrics": metrics, "model_path": str(model_path)}
    log_json(log_path, {"trial_index": trial_index, "model": "GAT", "test_metrics": metrics})
    del model, train_loader, val_loader, test_loader, train_graphs, val_graphs, test_graphs
    gc.collect()
    torch.cuda.empty_cache()
    return result, graph_visual_saved


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    paths = build_paths(base_dir, args.project_path)
    device = ensure_cuda_device()
    set_seed(args.seed)
    print(f"DEVICE: {device}")
    print(f"PYG_AVAILABLE: {PYG_AVAILABLE}")
    print(json.dumps({
        "run_lstm": args.run_lstm,
        "run_gat": args.run_gat,
        "use_sample_for_debug": args.use_sample_for_debug,
        "debug_n_rows": args.debug_n_rows,
        "max_hp_runs": min(args.max_hp_runs, len(HYPERPARAMETER_SETS)),
        "mainshock_mag_threshold": args.mainshock_mag_threshold,
        "lead_days": args.lead_days,
        "spatial_radius_km": args.spatial_radius_km,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "targets": IDEAL_TARGETS,
    }, indent=2, default=str))
    df_clean = load_or_prepare_clean_dataframe(args, paths, None)
    print(f"Cleaned shape: {df_clean.shape}")
    print(f"Minimum magnitude after cleaning: {float(df_clean[MAG_COL].min()):.6f}")
    save_preprocessing_visualizations(df_clean, paths["visualizations_dir"], args.visual_sample_size, None)
    df_labeled = load_or_prepare_labeled_dataframe(df_clean, args, paths, None)
    print(f"Labeled shape: {df_labeled.shape}")
    print(f"Positive rate: {float(df_labeled[TARGET_COL].mean()):.6f}")
    feature_cols = [column for column in BASE_FEATURE_COLS if column in df_labeled.columns]
    train_df, val_df, test_df, _ = load_or_prepare_splits(df_labeled, args, paths, feature_cols, None)
    print(json.dumps({
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "positive_rates": {
            "train": float(train_df[TARGET_COL].mean()),
            "val": float(val_df[TARGET_COL].mean()),
            "test": float(test_df[TARGET_COL].mean()),
        },
        "feature_cols": feature_cols,
    }, indent=2, default=str))
    graph_visual_saved = False
    if args.run_gat:
        graph_visual_saved = build_one_time_graph_visualization(train_df, feature_cols, paths["visualizations_dir"])
    all_results = []
    best_result = None
    best_lstm_candidate = None
    best_gat_candidate = None
    max_runs = min(args.max_hp_runs, len(HYPERPARAMETER_SETS))
    for trial_index in range(1, max_runs + 1):
        trial_log_path = prepare_trial_log(paths["logs_dir"], trial_index)
        trial_config = HYPERPARAMETER_SETS[trial_index - 1]
        log_json(trial_log_path, {"trial_index": trial_index, "config": trial_config})
        lstm_result = run_lstm_trial(trial_index, trial_config, train_df, val_df, test_df, feature_cols, device, paths["models_dir"], trial_log_path) if args.run_lstm else None
        gat_result = None
        if args.run_gat:
            gat_result, graph_visual_saved = run_gat_trial(trial_index, trial_config, train_df, val_df, test_df, feature_cols, device, paths["graphs_dir"], paths["models_dir"], paths["visualizations_dir"], trial_log_path, graph_visual_saved)
        proxy_ltss_improvement_pct = None
        if lstm_result is not None and gat_result is not None:
            baseline_f1 = max(lstm_result["metrics"]["f1"], 1e-8)
            proxy_ltss_improvement_pct = 100.0 * (gat_result["metrics"]["f1"] - lstm_result["metrics"]["f1"]) / baseline_f1
        trial_summary = {
            "trial_index": trial_index,
            "config": trial_config,
            "lstm": lstm_result,
            "gat": gat_result,
            "proxy_ltss_improvement_pct": proxy_ltss_improvement_pct,
        }
        if lstm_result is not None:
            trial_summary["lstm_target_report"] = build_target_report(lstm_result["metrics"], None, False)
            trial_summary["lstm_score"] = score_against_targets(lstm_result["metrics"], None)
            lstm_candidate = {
                "score": float(trial_summary["lstm_score"]),
                "model": "LSTM",
                "trial_index": trial_index,
                "result": lstm_result,
                "target_report": trial_summary["lstm_target_report"],
                "proxy_ltss_improvement_pct": None,
            }
            if best_lstm_candidate is None or lstm_candidate["score"] > best_lstm_candidate["score"]:
                best_lstm_candidate = lstm_candidate
                paths["best_lstm_summary_path"].write_text(json.dumps(best_lstm_candidate, indent=2, default=str), encoding="utf-8")
        if gat_result is not None:
            trial_summary["gat_target_report"] = build_target_report(gat_result["metrics"], proxy_ltss_improvement_pct, args.run_lstm)
            trial_summary["gat_score"] = score_against_targets(gat_result["metrics"], proxy_ltss_improvement_pct)
            gat_candidate = {
                "score": float(trial_summary["gat_score"]),
                "model": "GAT",
                "trial_index": trial_index,
                "result": gat_result,
                "target_report": trial_summary["gat_target_report"],
                "proxy_ltss_improvement_pct": proxy_ltss_improvement_pct,
            }
            if best_gat_candidate is None or gat_candidate["score"] > best_gat_candidate["score"]:
                best_gat_candidate = gat_candidate
                paths["best_gat_summary_path"].write_text(json.dumps(best_gat_candidate, indent=2, default=str), encoding="utf-8")
        all_results.append(trial_summary)
        log_json(trial_log_path, trial_summary)
        candidates = []
        if best_lstm_candidate is not None and best_lstm_candidate["trial_index"] == trial_index:
            candidates.append(best_lstm_candidate)
        elif lstm_result is not None:
            candidates.append({
                "score": float(trial_summary.get("lstm_score", -np.inf)),
                "model": "LSTM",
                "trial_index": trial_index,
                "result": lstm_result,
                "target_report": trial_summary.get("lstm_target_report", {}),
                "proxy_ltss_improvement_pct": None,
            })
        if best_gat_candidate is not None and best_gat_candidate["trial_index"] == trial_index:
            candidates.append(best_gat_candidate)
        elif gat_result is not None:
            candidates.append({
                "score": float(trial_summary.get("gat_score", -np.inf)),
                "model": "GAT",
                "trial_index": trial_index,
                "result": gat_result,
                "target_report": trial_summary.get("gat_target_report", {}),
                "proxy_ltss_improvement_pct": proxy_ltss_improvement_pct,
            })
        if candidates:
            top_candidate = max(candidates, key=lambda item: item["score"])
            if best_result is None or top_candidate["score"] > best_result["score"]:
                best_result = top_candidate
                paths["best_summary_path"].write_text(json.dumps(best_result, indent=2, default=str), encoding="utf-8")
        stop_for_lstm = trial_summary.get("lstm_target_report", {}).get("meets_all_targets", False)
        stop_for_gat = trial_summary.get("gat_target_report", {}).get("meets_all_targets", False)
        if stop_for_gat or (not args.run_gat and stop_for_lstm):
            log_line(trial_log_path, f"Stopping hyperparameter search at trial {trial_index} because all hardcoded target metrics were met.")
            break
    paths["all_results_path"].write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    if best_result is not None:
        print(json.dumps(sanitize_for_log({"best_result": best_result}), indent=2, default=str))
    else:
        print("No valid model result was produced.")
    final_full_run = run_best_models_on_full_dataset(best_lstm_candidate if args.run_lstm else None, best_gat_candidate if args.run_gat else None, df_labeled, feature_cols, device, paths)
    paths["final_full_run_path"].write_text(json.dumps(final_full_run, indent=2, default=str), encoding="utf-8")
    print(json.dumps(sanitize_for_log({"final_full_run": final_full_run}), indent=2, default=str))


if __name__ == "__main__":
    main()
