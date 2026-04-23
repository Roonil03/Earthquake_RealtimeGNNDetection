# Optional: mount Google Drive when running in Colab
import argparse
import os
import sys


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Earthquake foreshock prediction training script")
    parser.add_argument("--project-path", default=os.environ.get("PROJECT_PATH"), help="Directory for cached artifacts and models")
    parser.add_argument("--raw-csv-path", default=os.environ.get("RAW_CSV_PATH"), help="Optional local earthquake CSV path")
    parser.add_argument("--run-lstm", type=_str2bool, default=True, help="Run LSTM baseline")
    parser.add_argument("--run-gat", type=_str2bool, default=True, help="Run GAT model")
    parser.add_argument("--run-hyperparam-tuning", type=_str2bool, default=False, help="Run Optuna tuning")
    parser.add_argument("--use-sample-for-debug", type=_str2bool, default=False, help="Use only the first N rows for debugging")
    parser.add_argument("--debug-n-rows", type=int, default=400_000, help="Number of rows to use in debug mode")
    return parser.parse_args()


ARGS = parse_args()
PROJECT_PATH = os.path.abspath(ARGS.project_path or "./Earthquake_GNN")
os.makedirs(PROJECT_PATH, exist_ok=True)
print("PROJECT_PATH:", PROJECT_PATH)


# Install packages only if missing.
# In recent Colab environments, most of these will already exist.

import importlib
import subprocess
import sys

def ensure_package(import_name: str, pip_name: str | None = None):
    try:
        importlib.import_module(import_name)
        print(f"[ok] {import_name}")
    except Exception:
        pip_target = pip_name or import_name
        print(f"[install] {pip_target}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_target])

ensure_package("kagglehub")
ensure_package("pyarrow")
ensure_package("joblib")
ensure_package("optuna")
ensure_package("sklearn", "scikit-learn")
ensure_package("tqdm")
ensure_package("torch")
try:
    importlib.import_module("torch_geometric")
    print("[ok] torch_geometric")
except Exception:
    print("[install] torch-geometric")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch-geometric"])



import os
import gc
import math
import time
import json
import random
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)


from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as GeoDataLoader
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
    PYG_AVAILABLE = True
except Exception as e:
    print("PyG import failed:", e)
    PYG_AVAILABLE = False

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
PIN_MEMORY = True

print("DEVICE:", DEVICE)
print("PYG_AVAILABLE:", PYG_AVAILABLE)



# -------------------------------
# Global configuration
# -------------------------------

# Dataset loading
RAW_CSV_PATH = ARGS.raw_csv_path
KAGGLE_DATASET_ID = "alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023"

# Cache files
CLEAN_PARQUET_PATH = os.path.join(PROJECT_PATH, "earthquake_cleaned.parquet")
LABELED_PARQUET_PATH = os.path.join(PROJECT_PATH, "earthquake_labeled_m55_lead30_r100.parquet")
TRAIN_PARQUET_PATH = os.path.join(PROJECT_PATH, "train_df.parquet")
VAL_PARQUET_PATH = os.path.join(PROJECT_PATH, "val_df.parquet")
TEST_PARQUET_PATH = os.path.join(PROJECT_PATH, "test_df.parquet")
SCALER_PATH = os.path.join(PROJECT_PATH, "feature_scaler.joblib")
LSTM_MODEL_PATH = os.path.join(PROJECT_PATH, "best_lstm_model.pt")
GAT_MODEL_PATH = os.path.join(PROJECT_PATH, "best_gat_model.pt")

# Canonical columns
TIME_COL = "time"
LAT_COL = "latitude"
LON_COL = "longitude"
DEPTH_COL = "depth"
MAG_COL = "magnitudo"
SIG_COL = "significance"
TARGET_COL = "target"
TIME_DIFF_COL = "time_diff"
DIST_PREV_COL = "dist_prev"

# Labeling parameters
MAINSHOCK_MAG_THRESHOLD = 5.5
LEAD_DAYS = 30
SPATIAL_RADIUS_KM = 100.0

# Split
TRAIN_RATIO = 0.85
VAL_RATIO = 0.075

# Features
BASE_FEATURE_COLS = [
    LAT_COL,
    LON_COL,
    DEPTH_COL,
    MAG_COL,
    DIST_PREV_COL,
    TIME_DIFF_COL,
    SIG_COL,
]

# Runtime toggles
RUN_EDA = False
RUN_LSTM = ARGS.run_lstm
RUN_GAT = ARGS.run_gat
RUN_HYPERPARAM_TUNING = ARGS.run_hyperparam_tuning

# Fast debug toggles
USE_SAMPLE_FOR_DEBUG = ARGS.use_sample_for_debug
DEBUG_N_ROWS = ARGS.debug_n_rows

# LSTM parameters
# WINDOW_SIZE = 32
# WINDOW_STRIDE = 4
# LSTM_HIDDEN_DIM = 128
# LSTM_NUM_LAYERS = 2
# LSTM_DROPOUT = 0.25
# LSTM_BIDIRECTIONAL = True
# LSTM_BATCH_SIZE = 512
# LSTM_EPOCHS = 12
# LSTM_PATIENCE = 3
# LSTM_LR = 1e-3
# LSTM_WEIGHT_DECAY = 1e-5

LSTM_BIDIRECTIONAL = True
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 1
LSTM_DROPOUT = 0.10
LSTM_BATCH_SIZE = 1024
LSTM_EPOCHS = 8
LSTM_PATIENCE = 2
LSTM_LR = 5e-4
LSTM_WEIGHT_DECAY = 1e-4
WINDOW_SIZE = 24
WINDOW_STRIDE = 8

# GAT parameters
# GRAPH_WINDOW_DAYS = 90
# GRAPH_EDGE_TEMPORAL_DAYS = 90
# LOCAL_SUBGRAPH_RADIUS_KM = 300.0
# MAX_NODES_PER_GRAPH = 128
# MIN_NODES_PER_GRAPH = 8
# GRAPH_STRIDE = 8
# MAX_TRAIN_GRAPHS = 15000
# MAX_VAL_GRAPHS = 3000
# MAX_TEST_GRAPHS = 3000
GRAPH_WINDOW_DAYS = 60
GRAPH_EDGE_TEMPORAL_DAYS = 60
LOCAL_SUBGRAPH_RADIUS_KM = 200.0
MAX_NODES_PER_GRAPH = 96
MIN_NODES_PER_GRAPH = 12
GRAPH_STRIDE = 12
MAX_TRAIN_GRAPHS = 12000
MAX_VAL_GRAPHS = 2500
MAX_TEST_GRAPHS = 2500

# GAT_TCN_HIDDEN = 64
# GAT_HIDDEN = 64
# GAT_HEADS = 4
# GAT_DROPOUT = 0.25
# GAT_BATCH_SIZE = 128
# GAT_EPOCHS = 15
# GAT_PATIENCE = 4
# GAT_LR = 2e-3
# GAT_WEIGHT_DECAY = 1e-5
GAT_TCN_HIDDEN = 32
GAT_HIDDEN = 48
GAT_HEADS = 2
GAT_DROPOUT = 0.35
GAT_BATCH_SIZE = 64
GAT_EPOCHS = 10
GAT_PATIENCE = 3
GAT_LR = 1e-3
GAT_WEIGHT_DECAY = 5e-4

# Threshold search
# THRESH_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
THRESH_GRID = np.round(np.arange(0.05, 0.91, 0.05), 2)

# Graph dataset cache
TRAIN_GRAPHS_PATH = os.path.join(PROJECT_PATH, "train_graphs.pt")
VAL_GRAPHS_PATH = os.path.join(PROJECT_PATH, "val_graphs.pt")
TEST_GRAPHS_PATH = os.path.join(PROJECT_PATH, "test_graphs.pt")

# Benchmark targets from metricGaps.md
METRIC_GAPS_TARGETS = {
    "precision_overall": 0.851,
    "precision_best_site": 0.926,
    "recall_overall": 0.838,
    "recall_best_site": 0.924,
    "f1_overall": 0.839,
    "f1_best_site": 0.923,
    "auc_overall": 0.758,
    "auc_best_site": 0.817,
    "ltss_proxy_min_pct": 4.0,       # ETAS-style paper proxy threshold
    "tpr_at_20_fpr_target": 0.90,
    "efficiency_ms_per_event_max": 100.0,
}

print(json.dumps({
    "USE_SAMPLE_FOR_DEBUG": USE_SAMPLE_FOR_DEBUG,
    "RUN_LSTM": RUN_LSTM,
    "RUN_GAT": RUN_GAT,
    "RUN_HYPERPARAM_TUNING": RUN_HYPERPARAM_TUNING,
    "TRAIN_RATIO": TRAIN_RATIO,
    "VAL_RATIO": VAL_RATIO,
    "MAINSHOCK_MAG_THRESHOLD": MAINSHOCK_MAG_THRESHOLD,
    "LEAD_DAYS": LEAD_DAYS,
    "SPATIAL_RADIUS_KM": SPATIAL_RADIUS_KM,
}, indent=2))



def find_csv_in_dir(dir_path: str) -> str:
    candidates = []
    for name in os.listdir(dir_path):
        if name.lower().endswith(".csv"):
            candidates.append(os.path.join(dir_path, name))
    if not candidates:
        raise FileNotFoundError(f"No CSV found in: {dir_path}")

    # Prefer filenames containing earthquake-related words
    priority = sorted(
        candidates,
        key=lambda p: (
            0 if "earth" in os.path.basename(p).lower() else 1,
            len(os.path.basename(p))
        )
    )
    return priority[0]

def load_raw_dataframe(raw_csv_path: str | None = None) -> pd.DataFrame:
    if raw_csv_path is not None and os.path.exists(raw_csv_path):
        print("Loading raw CSV from:", raw_csv_path)
        return pd.read_csv(raw_csv_path)

    import kagglehub

    print(f"Downloading dataset from Kaggle: {KAGGLE_DATASET_ID}")
    download_path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    print("Downloaded to:", download_path)

    csv_path = find_csv_in_dir(download_path)
    print("Using CSV:", csv_path)
    return pd.read_csv(csv_path)

if os.path.exists(CLEAN_PARQUET_PATH):
    df_raw = pd.read_parquet(CLEAN_PARQUET_PATH)
    print("Loaded cached cleaned dataframe:", CLEAN_PARQUET_PATH)
else:
    df_raw = load_raw_dataframe(RAW_CSV_PATH)

if USE_SAMPLE_FOR_DEBUG and len(df_raw) > DEBUG_N_ROWS:
    df_raw = df_raw.iloc[:DEBUG_N_ROWS].copy()
    print(f"Debug mode enabled. Using first {len(df_raw):,} rows")

print("Raw shape:", df_raw.shape)
display(df_raw.head())
print("Columns:", list(df_raw.columns))



COLUMN_ALIASES = {
    TIME_COL: [TIME_COL, "timestamp", "event_time", "datetime"],
    LAT_COL: [LAT_COL, "lat"],
    LON_COL: [LON_COL, "lon", "lng"],
    DEPTH_COL: [DEPTH_COL, "depth_km"],
    MAG_COL: [MAG_COL, "magnitude", "mag"],
    SIG_COL: [SIG_COL, "sig"],
    "date": ["date"],
}

def resolve_column(df_in: pd.DataFrame, canonical_name: str, aliases: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df_in.columns}
    for name in aliases:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None

def standardize_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = resolve_column(df_out, canonical, aliases)
        if found is not None and found != canonical:
            rename_map[found] = canonical
    df_out = df_out.rename(columns=rename_map)
    return df_out

def ensure_datetime_time(df_in: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    df_out = df_in.copy()

    if time_col not in df_out.columns:
        raise KeyError(f"Column '{time_col}' not found in dataframe")

    col = df_out[time_col]

    if is_datetime64_any_dtype(col):
        if getattr(col.dt, "tz", None) is not None:
            df_out[time_col] = col.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            df_out[time_col] = col

    elif is_numeric_dtype(col):
        sample = col.dropna()
        if len(sample) == 0:
            raise ValueError(f"Column '{time_col}' contains no valid values")
        median_val = sample.median()
        unit = "ms" if median_val > 1e11 else "s"
        df_out[time_col] = pd.to_datetime(col, unit=unit, errors="coerce", utc=True).dt.tz_localize(None)

    else:
        parsed = pd.to_datetime(col, errors="coerce", utc=True)
        if parsed.notna().sum() < max(10, int(0.5 * len(df_out))):
            parsed = pd.to_datetime(pd.to_numeric(col, errors="coerce"), unit="ms", errors="coerce", utc=True)
        df_out[time_col] = parsed.dt.tz_localize(None)

    df_out = df_out.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return df_out

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def compute_prev_distance_km(df_in: pd.DataFrame) -> np.ndarray:
    lat_prev = df_in[LAT_COL].shift(1).to_numpy()
    lon_prev = df_in[LON_COL].shift(1).to_numpy()
    lat_cur = df_in[LAT_COL].to_numpy()
    lon_cur = df_in[LON_COL].to_numpy()

    dist_prev = np.zeros(len(df_in), dtype=np.float32)
    valid = ~pd.isna(lat_prev) & ~pd.isna(lon_prev)
    dist_prev[valid] = haversine_np(
        lat_prev[valid], lon_prev[valid], lat_cur[valid], lon_cur[valid]
    ).astype(np.float32)
    return dist_prev

def preprocess_earthquake_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = standardize_columns(df_in)

    required = [TIME_COL, LAT_COL, LON_COL, DEPTH_COL, MAG_COL]
    missing = [c for c in required if c not in df_out.columns]
    if missing:
        raise KeyError(f"Missing required columns after schema normalization: {missing}")

    if SIG_COL not in df_out.columns:
        df_out[SIG_COL] = 0.0

    for col in [LAT_COL, LON_COL, DEPTH_COL, MAG_COL, SIG_COL]:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    df_out = df_out.drop_duplicates()
    df_out = df_out.dropna(subset=[TIME_COL, LAT_COL, LON_COL, DEPTH_COL, MAG_COL]).copy()
    df_out = df_out[df_out[MAG_COL] >= 0].copy()
    df_out = ensure_datetime_time(df_out, TIME_COL)

    # Remove optional 'date' column safely
    df_out = df_out.drop(columns=["date"], errors="ignore")

    # Temporal + spatial engineered features
    df_out[TIME_DIFF_COL] = df_out[TIME_COL].diff().dt.total_seconds().fillna(0).astype(np.float32)
    df_out[DIST_PREV_COL] = compute_prev_distance_km(df_out)

    # Preserve raw values for graph construction and later diagnostics
    for base_col in [LAT_COL, LON_COL, DEPTH_COL, MAG_COL, DIST_PREV_COL, TIME_DIFF_COL]:
        df_out[f"{base_col}_raw"] = df_out[base_col]

    return df_out.reset_index(drop=True)

if os.path.exists(CLEAN_PARQUET_PATH):
    df = pd.read_parquet(CLEAN_PARQUET_PATH)
    print("Loaded cached cleaned parquet")
else:
    df = preprocess_earthquake_dataframe(df_raw)
    df.to_parquet(CLEAN_PARQUET_PATH, index=False)
    print("Saved cleaned parquet to:", CLEAN_PARQUET_PATH)

print("Cleaned shape:", df.shape)
display(df.head())
print(df.dtypes[[TIME_COL, LAT_COL, LON_COL, DEPTH_COL, MAG_COL, SIG_COL]])



if RUN_EDA:
    import matplotlib.pyplot as plt

    sample_n = min(5000, len(df))
    sample_df = df.sample(sample_n, random_state=SEED) if len(df) > sample_n else df

    print("Basic stats:")
    display(sample_df[[LAT_COL, LON_COL, DEPTH_COL, MAG_COL, SIG_COL, DIST_PREV_COL, TIME_DIFF_COL]].describe())

    plt.figure(figsize=(8, 5))
    sample_df[MAG_COL].hist(bins=50)
    plt.title("Magnitude distribution")
    plt.show()
else:
    print("RUN_EDA = False, skipping plots.")



def exact_binary_foreshock_labels(
    df_in: pd.DataFrame,
    mainshock_mag_threshold: float = MAINSHOCK_MAG_THRESHOLD,
    lead_days: int = LEAD_DAYS,
    spatial_radius_km: float = SPATIAL_RADIUS_KM,
) -> pd.DataFrame:
    out = df_in.copy().reset_index(drop=True)
    y = np.zeros(len(out), dtype=np.uint8)

    times_ns = out[TIME_COL].astype("int64").to_numpy()
    lat = out[f"{LAT_COL}_raw"].to_numpy(dtype=np.float64)
    lon = out[f"{LON_COL}_raw"].to_numpy(dtype=np.float64)
    mag = out[f"{MAG_COL}_raw"].to_numpy(dtype=np.float32)

    mainshock_idx = np.where(mag >= mainshock_mag_threshold)[0]
    lead_ns = int(lead_days * 24 * 3600 * 1e9)

    for ms_idx in tqdm(mainshock_idx, desc="Creating exact foreshock labels"):
        ms_time = times_ns[ms_idx]
        start_idx = np.searchsorted(times_ns, ms_time - lead_ns, side="left")
        if start_idx >= ms_idx:
            continue

        candidate_idx = np.arange(start_idx, ms_idx, dtype=np.int64)
        if candidate_idx.size == 0:
            continue

        lat_ms = lat[ms_idx]
        lon_ms = lon[ms_idx]

        lat_deg = spatial_radius_km / 111.0
        cos_lat = max(np.cos(np.radians(lat_ms)), 1e-6)
        lon_deg = spatial_radius_km / (111.0 * cos_lat)

        # Exact longitude distance with dateline-safe wrapping
        cand_lat = lat[candidate_idx]
        cand_lon = lon[candidate_idx]
        lon_diff = ((cand_lon - lon_ms + 180.0) % 360.0) - 180.0

        bbox_mask = (
            (cand_lat >= lat_ms - lat_deg) &
            (cand_lat <= lat_ms + lat_deg) &
            (np.abs(lon_diff) <= lon_deg)
        )

        candidate_idx = candidate_idx[bbox_mask]
        if candidate_idx.size == 0:
            continue

        dists = haversine_np(
            lat[candidate_idx],
            lon[candidate_idx],
            lat_ms,
            lon_ms
        )
        positive_idx = candidate_idx[dists <= spatial_radius_km]
        y[positive_idx] = 1

    out[TARGET_COL] = y.astype(np.int8)
    return out

if os.path.exists(LABELED_PARQUET_PATH):
    df = pd.read_parquet(LABELED_PARQUET_PATH)
    print("Loaded cached labeled parquet")
else:
    if TARGET_COL in df.columns:
        print(f"'{TARGET_COL}' already exists. Keeping existing labels.")
    else:
        df = exact_binary_foreshock_labels(df)
    df.to_parquet(LABELED_PARQUET_PATH, index=False)
    print("Saved labeled parquet to:", LABELED_PARQUET_PATH)

print(df[TARGET_COL].value_counts(dropna=False))
print("Positive rate:", float(df[TARGET_COL].mean()))



def temporal_split(df_in: pd.DataFrame, train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO):
    n = len(df_in)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df_in.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_in.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_in.iloc[val_end:].copy().reset_index(drop=True)

    return train_df, val_df, test_df

if all(os.path.exists(p) for p in [TRAIN_PARQUET_PATH, VAL_PARQUET_PATH, TEST_PARQUET_PATH, SCALER_PATH]):
    train_df = pd.read_parquet(TRAIN_PARQUET_PATH)
    val_df = pd.read_parquet(VAL_PARQUET_PATH)
    test_df = pd.read_parquet(TEST_PARQUET_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Loaded cached train/val/test splits and scaler")
else:
    train_df, val_df, test_df = temporal_split(df)

    feature_cols = [c for c in BASE_FEATURE_COLS if c in train_df.columns]
    scaler = StandardScaler()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    train_df.to_parquet(TRAIN_PARQUET_PATH, index=False)
    val_df.to_parquet(VAL_PARQUET_PATH, index=False)
    test_df.to_parquet(TEST_PARQUET_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)

print("Split sizes:", len(train_df), len(val_df), len(test_df))
print("Positive rate train:", float(train_df[TARGET_COL].mean()))
print("Positive rate val  :", float(val_df[TARGET_COL].mean()))
print("Positive rate test :", float(test_df[TARGET_COL].mean()))

MODEL_FEATURE_COLS = [c for c in BASE_FEATURE_COLS if c in train_df.columns]
print("MODEL_FEATURE_COLS:", MODEL_FEATURE_COLS)



class SequenceWindowDataset(Dataset):
    def __init__(self, df_in: pd.DataFrame, feature_cols: list[str], target_col: str, window_size: int = WINDOW_SIZE, stride: int = WINDOW_STRIDE):
        self.features = df_in[feature_cols].to_numpy(dtype=np.float32)
        self.labels = df_in[target_col].to_numpy(dtype=np.float32)
        self.window_size = window_size
        self.end_indices = np.arange(window_size, len(df_in), stride, dtype=np.int64)

    def __len__(self):
        return len(self.end_indices)

    def __getitem__(self, idx):
        end_idx = int(self.end_indices[idx])
        start_idx = end_idx - self.window_size
        x = self.features[start_idx:end_idx]
        y = self.labels[end_idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

    @property
    def window_labels(self):
        return self.labels[self.end_indices]

train_seq_ds = SequenceWindowDataset(train_df, MODEL_FEATURE_COLS, TARGET_COL, WINDOW_SIZE, WINDOW_STRIDE)
val_seq_ds = SequenceWindowDataset(val_df, MODEL_FEATURE_COLS, TARGET_COL, WINDOW_SIZE, WINDOW_STRIDE)
test_seq_ds = SequenceWindowDataset(test_df, MODEL_FEATURE_COLS, TARGET_COL, WINDOW_SIZE, WINDOW_STRIDE)

train_loader = DataLoader(
    train_seq_ds,
    batch_size=LSTM_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
val_loader = DataLoader(
    val_seq_ds,
    batch_size=LSTM_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)
test_loader = DataLoader(
    test_seq_ds,
    batch_size=LSTM_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

print("Sequence dataset sizes:", len(train_seq_ds), len(val_seq_ds), len(test_seq_ds))
print("Train window positive rate:", float(train_seq_ds.window_labels.mean()))



class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        bidirectional: bool = LSTM_BIDIRECTIONAL,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden).squeeze(-1)
        return logits

def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    labels = labels.astype(np.float32)
    pos = labels.sum()
    neg = len(labels) - pos
    # pos_weight = neg / max(pos, 1.0)
    raw = neg / max(pos, 1.0)
    pos_weight = min(raw, 3.0)
    return torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE)

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def tpr_at_target_fpr(y_true, y_prob, target_fpr=0.20):
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        idx = int(np.argmin(np.abs(fpr - target_fpr)))
        return {
            "target_fpr": float(target_fpr),
            "actual_fpr": float(fpr[idx]),
            "tpr": float(tpr[idx]),
            "threshold": float(thresholds[idx]),
        }
    except Exception:
        return {
            "target_fpr": float(target_fpr),
            "actual_fpr": np.nan,
            "tpr": np.nan,
            "threshold": np.nan,
        }

def classification_metrics(y_true, y_prob, threshold=0.5):
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
        metrics["auc_roc"] = np.nan

    roc_point = tpr_at_target_fpr(y_true, y_prob, target_fpr=0.20)
    metrics["tpr_at_20_fpr"] = roc_point["tpr"]
    metrics["actual_fpr_at_selected_roc_point"] = roc_point["actual_fpr"]
    metrics["roc_threshold_at_20_fpr"] = roc_point["threshold"]
    return metrics

def find_best_threshold(y_true, y_prob, grid=THRESH_GRID):
    best_threshold = 0.5
    best_f1 = -1.0
    for thr in grid:
        cur_f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = float(thr)
    return best_threshold, best_f1

@torch.no_grad()
def predict_loader(model, data_loader):
    model.eval()
    all_logits, all_y = [], []
    for xb, yb in data_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(yb.numpy())
    logits = np.concatenate(all_logits) if all_logits else np.array([])
    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_prob = sigmoid_np(logits)
    return y_true, y_prob

def train_binary_model(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    weight_decay,
    patience,
    model_path,
    pos_weight_tensor,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    best_val_f1 = -1.0
    best_threshold = 0.5
    epochs_without_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_y, val_prob = predict_loader(model, val_loader)
        best_thr_epoch, best_f1_epoch = find_best_threshold(val_y, val_prob)
        val_metrics = classification_metrics(val_y, val_prob, threshold=best_thr_epoch)

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses) if train_losses else np.nan),
            "val_threshold": best_thr_epoch,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_record)

        improved = best_f1_epoch > best_val_f1
        if improved:
            best_val_f1 = best_f1_epoch
            best_threshold = best_thr_epoch
            torch.save(model.state_dict(), model_path)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        print({
            "epoch": epoch,
            "train_loss": round(epoch_record["train_loss"], 5),
            "val_f1": round(val_metrics["f1"], 5),
            "val_precision": round(val_metrics["precision"], 5),
            "val_recall": round(val_metrics["recall"], 5),
            "val_auc_roc": round(val_metrics["auc_roc"], 5) if not pd.isna(val_metrics["auc_roc"]) else None,
            "best_threshold": best_thr_epoch,
        })

        if epochs_without_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return history, best_threshold

def measure_latency_per_event(model, example_batch, n_runs=30):
    model.eval()
    xb = example_batch.to(DEVICE)
    with torch.no_grad():
        for _ in range(5):
            _ = model(xb)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(xb)
            end = time.perf_counter()
            times.append(end - start)

    avg_batch_ms = float(np.mean(times) * 1000.0)
    ms_per_event = avg_batch_ms / max(len(example_batch), 1)
    return avg_batch_ms, ms_per_event



lstm_history = []
lstm_val_threshold = 0.5
lstm_test_metrics = None

if RUN_LSTM:
    lstm_model = LSTMClassifier(input_dim=len(MODEL_FEATURE_COLS)).to(DEVICE)
    lstm_pos_weight = compute_pos_weight(train_seq_ds.window_labels)

    lstm_history, lstm_val_threshold = train_binary_model(
        model=lstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=LSTM_EPOCHS,
        lr=LSTM_LR,
        weight_decay=LSTM_WEIGHT_DECAY,
        patience=LSTM_PATIENCE,
        model_path=LSTM_MODEL_PATH,
        pos_weight_tensor=lstm_pos_weight,
    )

    y_test_lstm, p_test_lstm = predict_loader(lstm_model, test_loader)
    lstm_test_metrics = classification_metrics(y_test_lstm, p_test_lstm, threshold=lstm_val_threshold)

    # Latency
    xb_example, _ = next(iter(test_loader))
    avg_batch_ms, ms_per_event = measure_latency_per_event(lstm_model, xb_example[: min(256, len(xb_example))])
    lstm_test_metrics["avg_batch_latency_ms"] = avg_batch_ms
    lstm_test_metrics["ms_per_event"] = ms_per_event

    print("\nLSTM test metrics")
    print(json.dumps(lstm_test_metrics, indent=2))
else:
    print("RUN_LSTM = False")



def build_temporal_encoding(delta_seconds: np.ndarray) -> np.ndarray:
    # delta_seconds is relative to the most recent event in the graph
    delta_days = delta_seconds / 86400.0
    log_delta = np.log1p(np.clip(delta_seconds, a_min=0, a_max=None))

    sin_7 = np.sin(2 * np.pi * delta_days / 7.0)
    cos_7 = np.cos(2 * np.pi * delta_days / 7.0)
    sin_30 = np.sin(2 * np.pi * delta_days / 30.0)
    cos_30 = np.cos(2 * np.pi * delta_days / 30.0)

    enc = np.stack([
        log_delta,
        delta_days / max(GRAPH_WINDOW_DAYS, 1),
        sin_7,
        cos_7,
        sin_30,
        cos_30,
    ], axis=1).astype(np.float32)
    return enc

def pairwise_haversine_km(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)[:, None]
    lon = np.radians(lon_deg)[:, None]

    dlat = lat.T - lat
    dlon = lon.T - lon

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

def create_graph_samples(
    df_split: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET_COL,
    graph_window_days: int = GRAPH_WINDOW_DAYS,
    edge_temporal_days: int = GRAPH_EDGE_TEMPORAL_DAYS,
    edge_radius_km: float = SPATIAL_RADIUS_KM,
    local_subgraph_radius_km: float = LOCAL_SUBGRAPH_RADIUS_KM,
    max_nodes: int = MAX_NODES_PER_GRAPH,
    min_nodes: int = MIN_NODES_PER_GRAPH,
    stride: int = GRAPH_STRIDE,
    max_graphs: int | None = None,
    desc: str = "graphs",
):
    if not PYG_AVAILABLE:
        raise RuntimeError("PyTorch Geometric is not available.")

    feature_arr = df_split[feature_cols].to_numpy(dtype=np.float32)
    labels = df_split[target_col].to_numpy(dtype=np.float32)
    times_ns = df_split[TIME_COL].astype("int64").to_numpy()
    lat = df_split[f"{LAT_COL}_raw"].to_numpy(dtype=np.float64)
    lon = df_split[f"{LON_COL}_raw"].to_numpy(dtype=np.float64)

    window_ns = int(graph_window_days * 24 * 3600 * 1e9)
    edge_temporal_ns = int(edge_temporal_days * 24 * 3600 * 1e9)

    graphs = []
    center_indices = np.arange(min_nodes, len(df_split), stride, dtype=np.int64)

    for center_idx in tqdm(center_indices, desc=f"Building {desc}"):
        center_time = times_ns[center_idx]
        start_idx = np.searchsorted(times_ns, center_time - window_ns, side="left")
        cand = np.arange(start_idx, center_idx + 1, dtype=np.int64)

        if cand.size < min_nodes:
            continue

        # Localize around the current event for a tractable subgraph
        d_center = haversine_np(lat[cand], lon[cand], lat[center_idx], lon[center_idx])
        cand = cand[d_center <= local_subgraph_radius_km]

        if cand.size < min_nodes:
            continue

        # Keep the most recent nodes if the local graph is still too large
        if cand.size > max_nodes:
            cand = cand[-max_nodes:]

        node_features = feature_arr[cand]
        delta_seconds = ((center_time - times_ns[cand]) / 1e9).astype(np.float32)
        temporal_enc = build_temporal_encoding(delta_seconds)
        x_np = np.concatenate([node_features, temporal_enc], axis=1).astype(np.float32)

        # Pairwise exact graph edges
        cand_lat = lat[cand]
        cand_lon = lon[cand]
        pair_d = pairwise_haversine_km(cand_lat, cand_lon)
        pair_dt_ns = times_ns[cand][None, :] - times_ns[cand][:, None]

        # Earlier -> later edges only, then mirror for undirected message passing
        forward_mask = (
            (pair_dt_ns >= 0) &
            (pair_dt_ns <= edge_temporal_ns) &
            (pair_d <= edge_radius_km)
        )
        np.fill_diagonal(forward_mask, False)

        src, dst = np.where(forward_mask)
        if len(src) == 0:
            continue

        edge_index_np = np.vstack([
            np.concatenate([src, dst]),
            np.concatenate([dst, src]),
        ]).astype(np.int64)

        edge_attr_forward = np.stack([
            (pair_d[src, dst] / max(edge_radius_km, 1e-6)).astype(np.float32),
            (pair_dt_ns[src, dst] / max(edge_temporal_ns, 1)).astype(np.float32),
        ], axis=1)
        edge_attr_np = np.vstack([edge_attr_forward, edge_attr_forward]).astype(np.float32)

        data = Data(
            x=torch.tensor(x_np, dtype=torch.float32),
            edge_index=torch.tensor(edge_index_np, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr_np, dtype=torch.float32),
            y=torch.tensor([labels[center_idx]], dtype=torch.float32),
        )
        data.graph_time = int(center_time)
        data.center_index = int(center_idx)
        data.num_nodes_local = int(x_np.shape[0])
        graphs.append(data)

        if max_graphs is not None and len(graphs) >= max_graphs:
            break

    return graphs

def maybe_load_or_build_graphs(df_split, save_path, max_graphs, desc):
    if os.path.exists(save_path):
        print("Loading cached graphs:", save_path)
        return torch.load(save_path, weights_only=False)
    graphs = create_graph_samples(
        df_split=df_split,
        feature_cols=MODEL_FEATURE_COLS,
        target_col=TARGET_COL,
        max_graphs=max_graphs,
        desc=desc,
    )
    torch.save(graphs, save_path)
    print("Saved graphs to:", save_path)
    return graphs

def validate_graph_samples(graphs, n_check=20):
    if len(graphs) == 0:
        print("No graphs available to validate.")
        return

    checks = []
    for g in graphs[: min(n_check, len(graphs))]:
        x_ok = g.x.ndim == 2 and g.x.shape[0] > 0 and g.x.shape[1] > 0
        edge_ok = g.edge_index.ndim == 2 and g.edge_index.shape[0] == 2 and g.edge_index.shape[1] > 0
        edge_attr_ok = g.edge_attr.ndim == 2 and g.edge_attr.shape[0] == g.edge_index.shape[1]
        target_ok = g.y.numel() == 1
        checks.append(all([x_ok, edge_ok, edge_attr_ok, target_ok]))

    print(f"Validated {len(checks)} graphs. All structurally valid:", bool(np.all(checks)))



train_graphs = []
val_graphs = []
test_graphs = []
gat_test_metrics = None

if RUN_GAT:
    if not PYG_AVAILABLE:
        print("RUN_GAT=True, but PyG is unavailable. Skipping GAT.")
        RUN_GAT = False
    else:
        train_graphs = maybe_load_or_build_graphs(train_df, TRAIN_GRAPHS_PATH, MAX_TRAIN_GRAPHS, "train graphs")
        val_graphs = maybe_load_or_build_graphs(val_df, VAL_GRAPHS_PATH, MAX_VAL_GRAPHS, "val graphs")
        test_graphs = maybe_load_or_build_graphs(test_df, TEST_GRAPHS_PATH, MAX_TEST_GRAPHS, "test graphs")

        validate_graph_samples(train_graphs)

        print("Graph counts:", len(train_graphs), len(val_graphs), len(test_graphs))
        if len(train_graphs) == 0 or len(val_graphs) == 0 or len(test_graphs) == 0:
            print("One or more graph splits are empty. Skipping GAT.")
            RUN_GAT = False
else:
    print("RUN_GAT = False")



class TemporalConvEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding="same")
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding="same")
        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, batch):
        # x is concatenated node features from a PyG batch; nodes remain grouped by graph
        counts = torch.bincount(batch)
        outputs = []
        start = 0
        for count in counts.tolist():
            xi = x[start:start + count]                      # [nodes, feat]
            xi = xi.transpose(0, 1).unsqueeze(0)            # [1, feat, nodes]
            xo = F.relu(self.conv1(xi))
            xo = F.relu(self.conv2(xo))
            xo = self.norm(xo)
            xo = xo.squeeze(0).transpose(0, 1)              # [nodes, hidden]
            outputs.append(xo)
            start += count
        return torch.cat(outputs, dim=0)

class TemporalGATClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        tcn_hidden: int = GAT_TCN_HIDDEN,
        gat_hidden: int = GAT_HIDDEN,
        heads: int = GAT_HEADS,
        dropout: float = GAT_DROPOUT,
        edge_dim: int = 2,
    ):
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

    def forward(self, data):
        x = self.temporal_encoder(data.x, data.batch)
        x = F.elu(self.gat1(x, data.edge_index, data.edge_attr))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, data.edge_index, data.edge_attr))
        pooled = torch.cat([global_mean_pool(x, data.batch), global_max_pool(x, data.batch)], dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return logits

@torch.no_grad()
def predict_geo_loader(model, data_loader):
    model.eval()
    all_logits, all_y = [], []
    for batch in data_loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(batch.y.detach().cpu().numpy())
    logits = np.concatenate(all_logits) if all_logits else np.array([])
    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_prob = sigmoid_np(logits)
    return y_true, y_prob

def train_geo_binary_model(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    weight_decay,
    patience,
    model_path,
    pos_weight_tensor,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    best_val_f1 = -1.0
    best_threshold = 0.5
    epochs_without_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"GAT epoch {epoch}/{epochs}", leave=False):
            batch = batch.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_y, val_prob = predict_geo_loader(model, val_loader)
        best_thr_epoch, best_f1_epoch = find_best_threshold(val_y, val_prob)
        val_metrics = classification_metrics(val_y, val_prob, threshold=best_thr_epoch)

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses) if train_losses else np.nan),
            "val_threshold": best_thr_epoch,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if best_f1_epoch > best_val_f1:
            best_val_f1 = best_f1_epoch
            best_threshold = best_thr_epoch
            torch.save(model.state_dict(), model_path)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        print({
            "epoch": epoch,
            "train_loss": round(history[-1]["train_loss"], 5),
            "val_f1": round(val_metrics["f1"], 5),
            "val_precision": round(val_metrics["precision"], 5),
            "val_recall": round(val_metrics["recall"], 5),
            "val_auc_roc": round(val_metrics["auc_roc"], 5) if not pd.isna(val_metrics["auc_roc"]) else None,
            "best_threshold": best_thr_epoch,
        })

        if epochs_without_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    return history, best_threshold

def measure_geo_latency_per_event(model, example_batch, n_runs=20):
    model.eval()
    batch = example_batch.to(DEVICE)

    with torch.no_grad():
        for _ in range(3):
            _ = model(batch)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(batch)
            end = time.perf_counter()
            times.append(end - start)

    avg_batch_ms = float(np.mean(times) * 1000.0)
    num_events = int(batch.num_graphs)
    ms_per_event = avg_batch_ms / max(num_events, 1)
    return avg_batch_ms, ms_per_event



gat_history = []
gat_val_threshold = 0.5

if RUN_GAT:
    train_geo_loader = GeoDataLoader(train_graphs, batch_size=GAT_BATCH_SIZE, shuffle=True)
    val_geo_loader = GeoDataLoader(val_graphs, batch_size=GAT_BATCH_SIZE, shuffle=False)
    test_geo_loader = GeoDataLoader(test_graphs, batch_size=GAT_BATCH_SIZE, shuffle=False)

    graph_labels_train = np.array([float(g.y.item()) for g in train_graphs], dtype=np.float32)
    gat_pos_weight = compute_pos_weight(graph_labels_train)

    input_dim_gat = train_graphs[0].x.shape[1]
    gat_model = TemporalGATClassifier(input_dim=input_dim_gat).to(DEVICE)

    gat_history, gat_val_threshold = train_geo_binary_model(
        model=gat_model,
        train_loader=train_geo_loader,
        val_loader=val_geo_loader,
        epochs=GAT_EPOCHS,
        lr=GAT_LR,
        weight_decay=GAT_WEIGHT_DECAY,
        patience=GAT_PATIENCE,
        model_path=GAT_MODEL_PATH,
        pos_weight_tensor=gat_pos_weight,
    )

    y_test_gat, p_test_gat = predict_geo_loader(gat_model, test_geo_loader)
    gat_test_metrics = classification_metrics(y_test_gat, p_test_gat, threshold=gat_val_threshold)

    example_graph_batch = next(iter(test_geo_loader))
    avg_batch_ms, ms_per_event = measure_geo_latency_per_event(gat_model, example_graph_batch)
    gat_test_metrics["avg_batch_latency_ms"] = avg_batch_ms
    gat_test_metrics["ms_per_event"] = ms_per_event

    print("\nGAT test metrics")
    print(json.dumps(gat_test_metrics, indent=2))
else:
    print("GAT section skipped.")



if RUN_HYPERPARAM_TUNING:
    import optuna

    def objective_lstm(trial):
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 192])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

        model = LSTMClassifier(
            input_dim=len(MODEL_FEATURE_COLS),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        ).to(DEVICE)

        tmp_path = os.path.join(PROJECT_PATH, f"tmp_lstm_trial_{trial.number}.pt")
        pos_weight = compute_pos_weight(train_seq_ds.window_labels)

        _, best_thr = train_binary_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=5,
            lr=lr,
            weight_decay=weight_decay,
            patience=2,
            model_path=tmp_path,
            pos_weight_tensor=pos_weight,
        )

        y_val, p_val = predict_loader(model, val_loader)
        metrics = classification_metrics(y_val, p_val, threshold=best_thr)
        return metrics["f1"]

    study_lstm = optuna.create_study(direction="maximize")
    study_lstm.optimize(objective_lstm, n_trials=10)

    print("Best LSTM params:", study_lstm.best_params)
    print("Best LSTM F1:", study_lstm.best_value)

    if RUN_GAT:
        def objective_gat(trial):
            tcn_hidden = trial.suggest_categorical("tcn_hidden", [32, 64, 96])
            gat_hidden = trial.suggest_categorical("gat_hidden", [32, 64, 96])
            heads = trial.suggest_categorical("heads", [2, 4, 8])
            dropout = trial.suggest_float("dropout", 0.1, 0.4)
            lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

            train_geo_loader = GeoDataLoader(train_graphs, batch_size=GAT_BATCH_SIZE, shuffle=True)
            val_geo_loader = GeoDataLoader(val_graphs, batch_size=GAT_BATCH_SIZE, shuffle=False)

            input_dim = train_graphs[0].x.shape[1]
            model = TemporalGATClassifier(
                input_dim=input_dim,
                tcn_hidden=tcn_hidden,
                gat_hidden=gat_hidden,
                heads=heads,
                dropout=dropout,
            ).to(DEVICE)

            tmp_path = os.path.join(PROJECT_PATH, f"tmp_gat_trial_{trial.number}.pt")
            pos_weight = compute_pos_weight(np.array([float(g.y.item()) for g in train_graphs], dtype=np.float32))

            _, best_thr = train_geo_binary_model(
                model=model,
                train_loader=train_geo_loader,
                val_loader=val_geo_loader,
                epochs=6,
                lr=lr,
                weight_decay=weight_decay,
                patience=2,
                model_path=tmp_path,
                pos_weight_tensor=pos_weight,
            )

            y_val, p_val = predict_geo_loader(model, val_geo_loader)
            metrics = classification_metrics(y_val, p_val, threshold=best_thr)
            return metrics["f1"]

        study_gat = optuna.create_study(direction="maximize")
        study_gat.optimize(objective_gat, n_trials=10)

        print("Best GAT params:", study_gat.best_params)
        print("Best GAT F1:", study_gat.best_value)
else:
    print("RUN_HYPERPARAM_TUNING = ARGS.run_hyperparam_tuning")



def safe_metric_value(metrics_dict, key):
    if metrics_dict is None:
        return np.nan
    return metrics_dict.get(key, np.nan)

def compare_model_to_targets(model_name: str, metrics_dict: dict | None):
    rows = []

    metric_rows = [
        ("precision", METRIC_GAPS_TARGETS["precision_overall"], METRIC_GAPS_TARGETS["precision_best_site"], True),
        ("recall", METRIC_GAPS_TARGETS["recall_overall"], METRIC_GAPS_TARGETS["recall_best_site"], True),
        ("f1", METRIC_GAPS_TARGETS["f1_overall"], METRIC_GAPS_TARGETS["f1_best_site"], True),
        ("auc_roc", METRIC_GAPS_TARGETS["auc_overall"], METRIC_GAPS_TARGETS["auc_best_site"], True),
        ("tpr_at_20_fpr", METRIC_GAPS_TARGETS["tpr_at_20_fpr_target"], np.nan, True),
        ("ms_per_event", METRIC_GAPS_TARGETS["efficiency_ms_per_event_max"], np.nan, False),
    ]

    for metric_name, overall_target, best_target, higher_is_better in metric_rows:
        value = safe_metric_value(metrics_dict, metric_name)
        if pd.isna(value):
            met_overall = None
            delta_overall = np.nan
        else:
            if higher_is_better:
                met_overall = bool(value >= overall_target)
                delta_overall = float(value - overall_target)
            else:
                met_overall = bool(value <= overall_target)
                delta_overall = float(overall_target - value)

        rows.append({
            "model": model_name,
            "metric": metric_name,
            "value": value,
            "overall_target": overall_target,
            "best_site_target": best_target,
            "delta_vs_overall_target": delta_overall,
            "meets_overall_target": met_overall,
        })

    return pd.DataFrame(rows)

comparison_frames = []

if lstm_test_metrics is not None:
    comparison_frames.append(compare_model_to_targets("LSTM", lstm_test_metrics))

if gat_test_metrics is not None:
    comparison_frames.append(compare_model_to_targets("GAT", gat_test_metrics))

if comparison_frames:
    benchmark_comparison_df = pd.concat(comparison_frames, ignore_index=True)
    display(benchmark_comparison_df)
else:
    benchmark_comparison_df = pd.DataFrame()
    print("No model metrics available yet.")

proxy_improvement_df = None
if lstm_test_metrics is not None and gat_test_metrics is not None:
    proxy_improvement_f1_pct = 100.0 * (
        (gat_test_metrics["f1"] - lstm_test_metrics["f1"]) / max(lstm_test_metrics["f1"], 1e-8)
    )
    proxy_improvement_df = pd.DataFrame([{
        "comparison": "GAT vs LSTM F1 improvement (%)",
        "value": proxy_improvement_f1_pct,
        "metricGaps_ltss_proxy_reference_pct": METRIC_GAPS_TARGETS["ltss_proxy_min_pct"],
        "note": "This is NOT ETAS LTSS; shown only as an internal comparison.",
    }])
    display(proxy_improvement_df)
else:
    print("Proxy improvement table skipped because both LSTM and GAT metrics are not available.")

