# ==========================================
# CELL 1 — Install / imports
# ==========================================
# If running in Colab, uncomment:
# !pip install -q torch torchvision torchaudio torch-geometric scikit-learn tqdm optuna

import os
import math
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ==========================================
# CELL 2 — Config
# ==========================================
TARGET_COL = "label"
TIME_COL = "time"
LAT_COL = "latitude_raw" if "latitude_raw" in df.columns else "latitude"
LON_COL = "longitude_raw" if "longitude_raw" in df.columns else "longitude"
MAG_COL = "magnitudo_raw" if "magnitudo_raw" in df.columns else "magnitudo"
DEPTH_COL = "depth_raw" if "depth_raw" in df.columns else "depth"
DIST_PREV_COL = "dist_prev_raw" if "dist_prev_raw" in df.columns else "dist_prev"
TIME_DIFF_COL = "time_diff_raw" if "time_diff_raw" in df.columns else "time_diff"
SIG_COL = "significance" if "significance" in df.columns else None

BASE_FEATURES = [LAT_COL, LON_COL, DEPTH_COL, MAG_COL, DIST_PREV_COL, TIME_DIFF_COL]
if SIG_COL is not None:
    BASE_FEATURES.append(SIG_COL)
BASE_FEATURES = [c for c in BASE_FEATURES if c in df.columns]

SEQ_LEN = 32
BATCH_SIZE = 256
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_LR = 1e-3
LSTM_EPOCHS = 10

GRAPH_LOOKBACK_DAYS = 90
SPATIAL_THRESHOLD_KM = 100.0
MAX_GRAPH_NODES = 128
GRAPH_BATCH_SIZE = 64
GAT_HIDDEN = 64
GAT_HEADS = 4
GAT_LAYERS = 2
TEMPORAL_EMBED_DIM = 16
TCN_CHANNELS = 64
GAT_DROPOUT = 0.2
GAT_LR = 1e-3
GAT_EPOCHS = 10

TUNE_TRIALS = 10

print("Base features:", BASE_FEATURES)

# ==========================================
# CELL 3 — Time fix + label creation + split
# ==========================================
def ensure_datetime_time(df_in: pd.DataFrame, time_col: str = TIME_COL) -> pd.DataFrame:
    df_out = df_in.copy()
    if not np.issubdtype(df_out[time_col].dtype, np.datetime64):
        # Notebook bug fix: source data is epoch milliseconds
        df_out[time_col] = pd.to_datetime(df_out[time_col], unit="ms", errors="coerce")
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


def build_binary_foreshock_labels(
    df_in: pd.DataFrame,
    mainshock_mag_threshold: float = 5.5,
    lead_days: int = 30,
    spatial_radius_km: float = 100.0,
) -> pd.DataFrame:
    """
    Label event i as 1 if a mainshock occurs within the next `lead_days`
    and within `spatial_radius_km` of the event. This is a practical working label.
    Replace with your catalog's official foreshock labels if available.
    """
    out = df_in.copy().reset_index(drop=True)
    out[TARGET_COL] = 0

    mainshock_idx = out.index[out[MAG_COL] >= mainshock_mag_threshold].to_numpy()
    times = out[TIME_COL].values.astype("datetime64[ns]")
    lat = out[LAT_COL].to_numpy()
    lon = out[LON_COL].to_numpy()

    for ms_idx in tqdm(mainshock_idx, desc="Creating labels"):
        ms_time = times[ms_idx]
        start_time = ms_time - np.timedelta64(lead_days, "D")
        candidate_mask = (times < ms_time) & (times >= start_time)
        candidate_idx = np.where(candidate_mask)[0]
        if len(candidate_idx) == 0:
            continue
        dists = haversine_np(lat[candidate_idx], lon[candidate_idx], lat[ms_idx], lon[ms_idx])
        positive_idx = candidate_idx[dists <= spatial_radius_km]
        out.loc[positive_idx, TARGET_COL] = 1
    return out


def temporal_split(df_in: pd.DataFrame, train_ratio=0.85, val_ratio=0.075):
    n = len(df_in)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_df = df_in.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_in.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_in.iloc[val_end:].copy().reset_index(drop=True)
    return train_df, val_df, test_df


df = ensure_datetime_time(df)
if TARGET_COL not in df.columns:
    df = build_binary_foreshock_labels(df, mainshock_mag_threshold=5.5, lead_days=30, spatial_radius_km=100.0)

# Recompute time_diff correctly if needed
if TIME_DIFF_COL in df.columns:
    df[TIME_DIFF_COL] = df[TIME_COL].diff().dt.total_seconds().fillna(0)

train_df, val_df, test_df = temporal_split(df, 0.85, 0.075)
print(len(train_df), len(val_df), len(test_df))
print(train_df[TARGET_COL].mean(), val_df[TARGET_COL].mean(), test_df[TARGET_COL].mean())

# ==========================================
# CELL 4 — Scale features using train only
# ==========================================
scaler = StandardScaler()
train_df[BASE_FEATURES] = scaler.fit_transform(train_df[BASE_FEATURES])
val_df[BASE_FEATURES] = scaler.transform(val_df[BASE_FEATURES])
test_df[BASE_FEATURES] = scaler.transform(test_df[BASE_FEATURES])

# ==========================================
# CELL 5 — Create sequence windows for LSTM
# ==========================================
class EarthquakeSequenceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_cols: List[str], seq_len: int = 32):
        self.frame = frame.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.X = []
        self.y = []
        vals = self.frame[self.feature_cols].to_numpy(dtype=np.float32)
        labels = self.frame[TARGET_COL].to_numpy(dtype=np.float32)

        for end_idx in range(seq_len, len(self.frame)):
            start_idx = end_idx - seq_len
            self.X.append(vals[start_idx:end_idx])
            self.y.append(labels[end_idx])

        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

train_seq_ds = EarthquakeSequenceDataset(train_df, BASE_FEATURES, SEQ_LEN)
val_seq_ds = EarthquakeSequenceDataset(val_df, BASE_FEATURES, SEQ_LEN)
test_seq_ds = EarthquakeSequenceDataset(test_df, BASE_FEATURES, SEQ_LEN)

train_seq_loader = DataLoader(train_seq_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_seq_loader = DataLoader(val_seq_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_seq_loader = DataLoader(test_seq_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Sequence windows:", len(train_seq_ds), len(val_seq_ds), len(test_seq_ds))

# ==========================================
# CELL 6 — Implement LSTM classifier
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1),
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        logits = self.classifier(h).squeeze(-1)
        return logits

lstm_model = LSTMClassifier(
    input_dim=len(BASE_FEATURES),
    hidden_dim=LSTM_HIDDEN,
    num_layers=LSTM_LAYERS,
    dropout=LSTM_DROPOUT,
    bidirectional=False,
).to(DEVICE)
print(lstm_model)

# ==========================================
# CELL 7 — Metrics, LTSS, and training utils
# ==========================================
def focal_bce_loss(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    focal = alpha * (1 - pt).pow(gamma) * bce
    return focal.mean()


def compute_class_weight(y: np.ndarray):
    pos = max(y.sum(), 1)
    neg = max(len(y) - y.sum(), 1)
    return torch.tensor([neg / pos], dtype=torch.float32, device=DEVICE)


def find_best_threshold(y_true, y_prob):
    best_thr, best_f1 = 0.5, -1
    for thr in np.arange(0.1, 0.91, 0.05):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1


def compute_ltss(model_metric: float, baseline_metric: float):
    # Skill score form relative to baseline.
    denom = max(1.0 - baseline_metric, 1e-8)
    return (model_metric - baseline_metric) / denom


def simple_rate_baseline_prob(frame: pd.DataFrame, window_size: int = 32):
    # A lightweight baseline proxy when ETAS outputs are unavailable.
    # Replace with ETAS probabilities once available.
    mags = frame[MAG_COL].to_numpy()
    baseline = np.zeros(len(frame), dtype=np.float32)
    for i in range(window_size, len(frame)):
        window = mags[i - window_size:i]
        baseline[i] = 1 / (1 + np.exp(-(window.mean() - 4.0)))
    return baseline


def evaluate_predictions(y_true, y_prob, threshold=0.5, baseline_prob=None):
    y_pred = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    out = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc,
    }

    if baseline_prob is not None:
        baseline_pred = (baseline_prob >= threshold).astype(int)
        baseline_f1 = f1_score(y_true, baseline_pred, zero_division=0)
        out["ltss"] = compute_ltss(f1, baseline_f1)
        out["baseline_f1"] = baseline_f1
    else:
        out["ltss"] = np.nan
    return out


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in tqdm(loader, leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_lstm(model, loader):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        labels.append(yb.numpy())
    return np.concatenate(labels), np.concatenate(probs)

# ==========================================
# CELL 8 — Train baseline model (LSTM)
# ==========================================
train_targets = train_seq_ds.y
pos_weight = compute_class_weight(train_targets)
criterion = lambda logits, y: F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
optimizer = torch.optim.AdamW(lstm_model.parameters(), lr=LSTM_LR, weight_decay=1e-4)

best_state = None
best_val_f1 = -1
best_thr = 0.5
history = []

val_baseline_prob_full = simple_rate_baseline_prob(val_df, window_size=SEQ_LEN)[SEQ_LEN:]

for epoch in range(1, LSTM_EPOCHS + 1):
    train_loss = train_epoch(lstm_model, train_seq_loader, optimizer, criterion)
    y_val, p_val = predict_lstm(lstm_model, val_seq_loader)
    thr, _ = find_best_threshold(y_val, p_val)
    metrics = evaluate_predictions(y_val, p_val, threshold=thr, baseline_prob=val_baseline_prob_full)
    history.append({"epoch": epoch, "train_loss": train_loss, **metrics, "threshold": thr})
    print(f"[LSTM] epoch={epoch} loss={train_loss:.4f} f1={metrics['f1']:.4f} auc={metrics['auc_roc']:.4f} thr={thr:.2f}")

    if metrics["f1"] > best_val_f1:
        best_val_f1 = metrics["f1"]
        best_thr = thr
        best_state = {k: v.cpu().clone() for k, v in lstm_model.state_dict().items()}

if best_state is not None:
    lstm_model.load_state_dict(best_state)

history_df = pd.DataFrame(history)
display(history_df)

# ==========================================
# CELL 9 — Compute LSTM metrics on test set
# ==========================================
y_test_lstm, p_test_lstm = predict_lstm(lstm_model, test_seq_loader)
test_baseline_prob_full = simple_rate_baseline_prob(test_df, window_size=SEQ_LEN)[SEQ_LEN:]
lstm_test_metrics = evaluate_predictions(y_test_lstm, p_test_lstm, threshold=best_thr, baseline_prob=test_baseline_prob_full)
print("LSTM test metrics:", lstm_test_metrics)

# ==========================================
# CELL 10 — Graph utilities and graph dataset creation
# ==========================================
def temporal_encoding(delta_seconds: np.ndarray, dim: int = 16) -> np.ndarray:
    delta_seconds = np.asarray(delta_seconds, dtype=np.float32).reshape(-1, 1)
    pe = np.zeros((len(delta_seconds), dim), dtype=np.float32)
    for i in range(0, dim, 2):
        div_term = 10000 ** (i / dim)
        pe[:, i] = np.sin(delta_seconds[:, 0] / div_term)
        if i + 1 < dim:
            pe[:, i + 1] = np.cos(delta_seconds[:, 0] / div_term)
    return pe


def build_graph_for_event(frame: pd.DataFrame, end_idx: int, lookback_days=90, max_nodes=128, spatial_km=100.0):
    current_time = frame.iloc[end_idx][TIME_COL]
    start_time = current_time - pd.Timedelta(days=lookback_days)
    sub = frame.iloc[: end_idx + 1].copy()
    sub = sub[sub[TIME_COL].between(start_time, current_time)].copy()
    sub = sub.tail(max_nodes).reset_index(drop=True)

    if len(sub) < 2:
        return None

    x_num = sub[BASE_FEATURES].to_numpy(dtype=np.float32)
    delta_sec = (current_time - sub[TIME_COL]).dt.total_seconds().to_numpy(dtype=np.float32)
    t_enc = temporal_encoding(delta_sec, TEMPORAL_EMBED_DIM)
    x = np.concatenate([x_num, t_enc], axis=1)

    lat = sub[LAT_COL].to_numpy()
    lon = sub[LON_COL].to_numpy()
    times = sub[TIME_COL].to_numpy()
    mags = sub[MAG_COL].to_numpy(dtype=np.float32)

    edge_src, edge_dst, edge_attr = [], [], []
    for j in range(len(sub)):
        for i in range(j):
            dt_days = (times[j] - times[i]) / np.timedelta64(1, 'D')
            if dt_days < 0 or dt_days > lookback_days:
                continue
            dist = haversine_np(lat[i], lon[i], lat[j], lon[j])
            if dist <= spatial_km:
                edge_src.extend([i, j])
                edge_dst.extend([j, i])
                mag_diff = abs(mags[j] - mags[i])
                attr = [dist / spatial_km, dt_days / lookback_days, mag_diff]
                edge_attr.extend([attr, attr])

    if len(edge_src) == 0:
        edge_src = list(range(len(sub) - 1))
        edge_dst = list(range(1, len(sub)))
        edge_attr = [[0.0, 0.0, 0.0] for _ in edge_src]

    y = np.array([sub.iloc[-1][TARGET_COL]], dtype=np.float32)
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
    )
    return data


class EventGraphDataset(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame, stride: int = 1, lookback_days=90, max_nodes=128, spatial_km=100.0):
        self.graphs = []
        for idx in tqdm(range(1, len(frame), stride), desc="Building graphs"):
            g = build_graph_for_event(frame, idx, lookback_days=lookback_days, max_nodes=max_nodes, spatial_km=spatial_km)
            if g is not None:
                self.graphs.append(g)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

# IMPORTANT:
# For full 3.3M rows, build graphs in chunks or use stride > 1 first.
GRAPH_STRIDE = 10
train_graph_ds = EventGraphDataset(train_df, stride=GRAPH_STRIDE, lookback_days=GRAPH_LOOKBACK_DAYS, max_nodes=MAX_GRAPH_NODES, spatial_km=SPATIAL_THRESHOLD_KM)
val_graph_ds = EventGraphDataset(val_df, stride=GRAPH_STRIDE, lookback_days=GRAPH_LOOKBACK_DAYS, max_nodes=MAX_GRAPH_NODES, spatial_km=SPATIAL_THRESHOLD_KM)
test_graph_ds = EventGraphDataset(test_df, stride=GRAPH_STRIDE, lookback_days=GRAPH_LOOKBACK_DAYS, max_nodes=MAX_GRAPH_NODES, spatial_km=SPATIAL_THRESHOLD_KM)

train_graph_loader = PyGDataLoader(train_graph_ds, batch_size=GRAPH_BATCH_SIZE, shuffle=True)
val_graph_loader = PyGDataLoader(val_graph_ds, batch_size=GRAPH_BATCH_SIZE, shuffle=False)
test_graph_loader = PyGDataLoader(test_graph_ds, batch_size=GRAPH_BATCH_SIZE, shuffle=False)

print(len(train_graph_ds), len(val_graph_ds), len(test_graph_ds))

# ==========================================
# CELL 11 — Implement Graph Attention Network + multi-head attention + temporal convolution + classification head
# ==========================================
class TemporalConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        return F.relu(x)


class GATTemporalNet(nn.Module):
    def __init__(
        self,
        input_dim,
        edge_dim=3,
        hidden_dim=64,
        heads=4,
        num_layers=2,
        tcn_channels=64,
        dropout=0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = hidden_dim if layer_idx == 0 else hidden_dim * heads
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim * heads))

        self.temporal_conv = TemporalConvBlock(hidden_dim * heads, kernel_size=3, dilation=1, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear((hidden_dim * heads) * 2, hidden_dim * heads),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.input_proj(x)

        for gat, norm in zip(self.gat_layers, self.norms):
            residual = x
            x = gat(x, edge_index, edge_attr)
            x = norm(x)
            x = F.elu(x)
            if residual.shape == x.shape:
                x = x + residual

        # temporal conv over node dimension per graph after sorting by batch order
        dense_list = []
        batch_ids = batch.unique(sorted=True)
        max_nodes = max([(batch == b).sum().item() for b in batch_ids])
        feat_dim = x.size(-1)

        for b in batch_ids:
            xb = x[batch == b]
            if xb.size(0) < max_nodes:
                pad = torch.zeros(max_nodes - xb.size(0), feat_dim, device=xb.device)
                xb = torch.cat([xb, pad], dim=0)
            dense_list.append(xb.unsqueeze(0))

        dense_x = torch.cat(dense_list, dim=0)      # [B, N, F]
        dense_x = dense_x.transpose(1, 2)           # [B, F, N]
        dense_x = self.temporal_conv(dense_x)
        dense_x = dense_x.transpose(1, 2)           # [B, N, F]

        mean_pool = dense_x.mean(dim=1)
        max_pool, _ = dense_x.max(dim=1)
        graph_emb = torch.cat([mean_pool, max_pool], dim=1)
        logits = self.classifier(graph_emb).squeeze(-1)
        return logits

sample_input_dim = train_graph_ds[0].x.shape[1]
gat_model = GATTemporalNet(
    input_dim=sample_input_dim,
    edge_dim=3,
    hidden_dim=GAT_HIDDEN,
    heads=GAT_HEADS,
    num_layers=GAT_LAYERS,
    tcn_channels=TCN_CHANNELS,
    dropout=GAT_DROPOUT,
).to(DEVICE)
print(gat_model)

# ==========================================
# CELL 12 — GNN loss functions and prediction helpers
# ==========================================
def make_bce_criterion_from_graphs(graph_dataset):
    y = np.array([g.y.item() for g in graph_dataset.graphs], dtype=np.float32)
    pos_weight = compute_class_weight(y)
    return lambda logits, y_true: F.binary_cross_entropy_with_logits(logits, y_true, pos_weight=pos_weight)


def train_graph_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, leave=False):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_graph(model, loader):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        labels.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs)

# ==========================================
# CELL 13 — Train GAT + temporal model
# ==========================================
gat_criterion = make_bce_criterion_from_graphs(train_graph_ds)
gat_optimizer = torch.optim.AdamW(gat_model.parameters(), lr=GAT_LR, weight_decay=1e-4)

best_gat_state = None
best_gat_f1 = -1
best_gat_thr = 0.5
gat_history = []

# baseline proxy aligned to graph count
val_graph_baseline_prob = np.array([0.5] * len(val_graph_ds), dtype=np.float32)

for epoch in range(1, GAT_EPOCHS + 1):
    train_loss = train_graph_epoch(gat_model, train_graph_loader, gat_optimizer, gat_criterion)
    y_val_g, p_val_g = predict_graph(gat_model, val_graph_loader)
    thr, _ = find_best_threshold(y_val_g, p_val_g)
    metrics = evaluate_predictions(y_val_g, p_val_g, threshold=thr, baseline_prob=val_graph_baseline_prob)
    gat_history.append({"epoch": epoch, "train_loss": train_loss, **metrics, "threshold": thr})
    print(f"[GAT] epoch={epoch} loss={train_loss:.4f} f1={metrics['f1']:.4f} auc={metrics['auc_roc']:.4f} thr={thr:.2f}")

    if metrics["f1"] > best_gat_f1:
        best_gat_f1 = metrics["f1"]
        best_gat_thr = thr
        best_gat_state = {k: v.cpu().clone() for k, v in gat_model.state_dict().items()}

if best_gat_state is not None:
    gat_model.load_state_dict(best_gat_state)

gat_history_df = pd.DataFrame(gat_history)
display(gat_history_df)

# ==========================================
# CELL 14 — Compute GAT metrics on test set
# ==========================================
y_test_gat, p_test_gat = predict_graph(gat_model, test_graph_loader)
test_graph_baseline_prob = np.array([0.5] * len(test_graph_ds), dtype=np.float32)
gat_test_metrics = evaluate_predictions(y_test_gat, p_test_gat, threshold=best_gat_thr, baseline_prob=test_graph_baseline_prob)
print("GAT test metrics:", gat_test_metrics)

# ==========================================
# CELL 15 — Hyperparameter tuning (LSTM and GAT)
# ==========================================
# Install optuna if needed:
# !pip install -q optuna
import optuna


def objective_lstm(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.5])
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)

    model = LSTMClassifier(
        input_dim=len(BASE_FEATURES),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=False,
    ).to(DEVICE)

    criterion = lambda logits, y: F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(3):
        train_epoch(model, train_seq_loader, optimizer, criterion)

    y_val, p_val = predict_lstm(model, val_seq_loader)
    thr, _ = find_best_threshold(y_val, p_val)
    metrics = evaluate_predictions(y_val, p_val, threshold=thr, baseline_prob=val_baseline_prob_full)
    return metrics["f1"]


def objective_gat(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    heads = trial.suggest_categorical("heads", [2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 2, 3)
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.4])
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)

    model = GATTemporalNet(
        input_dim=sample_input_dim,
        edge_dim=3,
        hidden_dim=hidden_dim,
        heads=heads,
        num_layers=num_layers,
        tcn_channels=TCN_CHANNELS,
        dropout=dropout,
    ).to(DEVICE)

    criterion = make_bce_criterion_from_graphs(train_graph_ds)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(3):
        train_graph_epoch(model, train_graph_loader, optimizer, criterion)

    y_val, p_val = predict_graph(model, val_graph_loader)
    thr, _ = find_best_threshold(y_val, p_val)
    metrics = evaluate_predictions(y_val, p_val, threshold=thr, baseline_prob=val_graph_baseline_prob)
    return metrics["f1"]

# Run tuning
lstm_study = optuna.create_study(direction="maximize", study_name="lstm_foreshock")
lstm_study.optimize(objective_lstm, n_trials=TUNE_TRIALS)
print("Best LSTM params:", lstm_study.best_params)
print("Best LSTM F1:", lstm_study.best_value)

gat_study = optuna.create_study(direction="maximize", study_name="gat_temporal_foreshock")
gat_study.optimize(objective_gat, n_trials=TUNE_TRIALS)
print("Best GAT params:", gat_study.best_params)
print("Best GAT F1:", gat_study.best_value)

# ==========================================
# CELL 16 — Final comparison table
# ==========================================
comparison_df = pd.DataFrame([
    {"model": "LSTM", **lstm_test_metrics},
    {"model": "GAT+Temporal", **gat_test_metrics},
])
display(comparison_df)
