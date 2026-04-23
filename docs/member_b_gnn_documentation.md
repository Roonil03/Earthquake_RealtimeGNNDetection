# Member B Documentation: Model Design, Baseline, Graph Input Format, and Dataset Split

## Scope
This document covers:
1. Finalized **GAT + temporal model** design
2. **LSTM baseline** design
3. **Required libraries**
4. Validation of the current preprocessing / graph-readiness in the notebook
5. Proper **GNN input format**
6. **Train / validation / test** split strategy for the preprocessed earthquake data

---

## 1) Finalized GAT + Temporal Model Design

### Objective
Detect whether an incoming earthquake event belongs to a **foreshock sequence** or to **background seismicity**, using a dynamic spatio-temporal graph built from earthquake catalogs.

### Recommended architecture
A **spatio-temporal Graph Attention Network (ST-GAT)** with event-level temporal encoding:

#### Input
Each graph snapshot contains:
- **Nodes** = earthquake events inside a sliding temporal window
- **Edges** = event pairs satisfying:
  - spatial distance <= **100 km**
  - temporal difference <= **90 days**
  - edge direction should be **past -> future** to preserve causality

#### Node features
Use these per event:
- latitude_norm
- longitude_norm
- depth_norm
- magnitude_norm
- significance_norm
- log1p(time_since_prev_seconds)
- log1p(distance_from_prev_km)

Keep the following raw values in parallel for graph construction and evaluation:
- event_time_utc
- latitude_raw
- longitude_raw
- depth_raw
- magnitude_raw

#### Edge features
For each edge `(j -> i)` where event `j` happened before event `i`, store:
- delta_t_seconds
- delta_t_days
- haversine_distance_km
- delta_depth_km
- delta_magnitude
- same_region flag (optional)
- normalized versions of the above for training

#### Temporal encoding
Use one of these:
- **Time2Vec** on event timestamps or relative age inside the window
- sinusoidal positional encoding on event order
- learned age embedding from `log1p(delta_t_seconds)`

Recommended practical choice:
- event age embedding from `log1p(age_seconds)`
- edge time embedding from `log1p(delta_t_seconds)`

#### GAT block
Recommended stack:
- Input projection -> hidden size **64**
- **2 to 3 GAT layers**
- **4 attention heads**
- hidden size per head: **16**
- activation: **GELU** or **ReLU**
- dropout: **0.2**
- residual connections enabled
- layer normalization after each block

#### Temporal aggregation
After each GAT layer, combine graph output with temporal modeling using either:
- **Temporal Convolution (TCN)** over events sorted by time inside each graph, or
- gated temporal MLP using age encodings

Recommended practical choice:
- **TCN with kernel sizes 3 and 5**
- dilation `[1, 2]`
- dropout `0.2`

#### Readout / prediction head
Two supported settings:

**A. Event-level binary classification**
- Output: probability that the current event belongs to a foreshock sequence

**B. Graph/window-level binary classification**
- Output: probability that the current window contains a foreshock pattern preceding a mainshock

Recommended for the current synopsis:
- start with **event-level binary classification**
- optionally extend to multitask:
  - foreshock/background classification
  - time-to-mainshock regression
  - expected mainshock magnitude regression

#### Loss
- `BCEWithLogitsLoss` for binary classification
- use **class weights** or **focal loss** because foreshocks are rare
- multitask extension:
  - `L_total = L_cls + 0.3 * L_time + 0.2 * L_mag`

#### Suggested hyperparameters
- node hidden dim: **64**
- output dim before classifier: **64**
- heads: **4**
- layers: **2**
- batch size: **16-64** graph windows depending on memory
- optimizer: **AdamW**
- lr: **1e-3**
- weight decay: **1e-4**
- epochs: **30-80**
- early stopping patience: **8**

---

## 2) LSTM Baseline Structure

### Purpose
Provide a simpler temporal baseline that ignores explicit graph structure.

### Sequence construction
For each target event or target window:
- sort preceding events by time
- use the last **N events** or all events in the last **90 days**
- recommended fixed sequence length: **64** or **128**
- pad shorter sequences and mask them

### Input features per timestep
Use the same event-level features as much as possible:
- magnitude_norm
- depth_norm
- latitude_norm
- longitude_norm
- significance_norm
- log1p(time_since_prev_seconds)
- log1p(distance_from_prev_km)

Optional:
- bearing change
- local density in last 1 / 7 / 30 days

### Recommended baseline
- 2-layer LSTM
- hidden size **128**
- dropout **0.2**
- bidirectional = **False** for real-time causality
- final hidden state -> MLP classifier
- output = sigmoid logit for binary classification

### Baseline head
- Linear(128 -> 64)
- ReLU
- Dropout(0.2)
- Linear(64 -> 1)

### Loss / optimization
- `BCEWithLogitsLoss`
- AdamW, lr `1e-3`
- same split and same evaluation metrics as GNN

### Why this is a fair baseline
It uses:
- the same earthquake catalog
- the same core engineered features
- the same target label definition
but does **not** use spatial graph structure explicitly.

---

## 3) Required Libraries

### Core ML
- `torch`
- `torchvision` (optional, not required unless reused elsewhere)
- `torchaudio` (not required here)
- `torchmetrics`

### Graph learning
- `torch-geometric`
- `torch-scatter`
- `torch-sparse`
- `torch-cluster`
- `networkx` (for debugging / graph inspection)

### Data handling
- `pandas`
- `numpy`
- `scikit-learn`
- `pyarrow` (recommended for Parquet)
- `joblib`

### Geospatial / scientific
- `scipy`
- `geopy` (optional, good for validation; Haversine is enough in production)
- `haversine` (optional utility)
- `numba` (recommended for fast custom pairwise graph construction)

### Visualization / analysis
- `matplotlib`
- `seaborn`
- `plotly` (optional for interactive graph inspection)

### Experiment management
- `tqdm`
- `pyyaml`
- `tensorboard` or `wandb` (optional but useful)
- `rich` (optional for cleaner logs)

### Real-time / deployment
- `fastapi`
- `uvicorn`
- `pydantic`
- `requests`

### Recommended install block
```bash
pip install torch torchmetrics pandas numpy scikit-learn scipy matplotlib seaborn pyarrow joblib tqdm networkx numba fastapi uvicorn pydantic requests
pip install torch-geometric
```

If PyG wheels are needed separately on a local machine, install `torch-scatter`, `torch-sparse`, and related packages according to the installed PyTorch and CUDA versions.

---

## 4) Validation of the Current Notebook and Graph Readiness

## What is already good
The notebook already does:
- duplicate removal
- removal of negative magnitudes
- chronological sorting intent
- creation of `time_diff`
- creation of `dist_prev`
- normalization of several useful features

These are useful first preprocessing steps.

## Critical issue that must be fixed first
The notebook converts:
```python
df['time'] = pd.to_datetime(df['time'])
```

But the raw `time` column is in **Unix epoch milliseconds**.  
Because `unit='ms'` was not provided, pandas interprets the values incorrectly, which is why the displayed `time` becomes around **1970-01-01** instead of the real catalog years around **1990-2023**.

### Why this matters
This breaks:
- event ordering reliability
- `time_diff`
- any 7-day / 14-day / 30-day lead-time labeling
- graph edge creation based on temporal thresholds
- train/validation/test temporal split

### Correct fix
Use either:
```python
df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
```

or, since the dataset already has a parsed `date` column:
```python
df['time'] = pd.to_datetime(df['date'], utc=True)
```

Recommended choice:
- use `date` as the authoritative timestamp if it is already correct
- then drop the original raw `time` column or rename it to `time_ms_raw`

## Additional validation findings

### A. `time_diff` is currently not trustworthy
Because of the timestamp parsing bug, the current `time_diff` values are effectively wrong for seismic modeling.

### B. `dist_prev` is useful, but not sufficient for graph construction
`dist_prev` only measures distance to the immediately previous event in the globally sorted catalog.
That is **not enough** for a GNN, because the graph needs **pairwise edges** between all relevant past events inside a local spatial-temporal neighborhood.

### C. Raw columns were saved after cleaning, which is good
The notebook preserves:
- `latitude_raw`
- `longitude_raw`
- `depth_raw`
- `magnitudo_raw`
- `dist_prev_raw`
- `time_diff_raw`

That is helpful, but `time_diff_raw` should be recomputed after fixing time parsing.

### D. Significance should be inspected for scale and outliers
Keep it, but consider:
- standardization
- optional clipping at high percentiles

### E. Missing graph objects
The notebook currently contains **no actual graph tensor construction yet**:
- no `edge_index`
- no `edge_attr`
- no node label tensor
- no PyG `Data` objects

So the notebook is **graph-ready only after corrections**, not graph-complete yet.

## Conclusion
In its current state, the notebook preprocessing is **not yet valid for final GNN training** because the time conversion is wrong and graph tensors have not been created yet.  
After fixing timestamp parsing and recomputing temporal features, the data can be used properly for both the GNN and the LSTM baseline.

---

## 5) Proper GNN Input Format

Use **PyTorch Geometric** style graph samples.

### Recommended sample unit
A single training sample should be a **graph window**:
- target event at time `t`
- context = all events within the previous **90 days**
- edges only from earlier events to later events
- spatial threshold <= **100 km**

### PyG object
```python
Data(
    x=node_features,           # [num_nodes, num_node_features]
    edge_index=edge_index,     # [2, num_edges]
    edge_attr=edge_features,   # [num_edges, num_edge_features]
    y=label,                   # [1] or [num_nodes]
    event_time=event_time,     # optional metadata
    event_id=event_id,         # optional metadata
    target_node_mask=mask      # optional for event-level prediction
)
```

### Required tensors

#### `x` node feature matrix
Shape:
```text
[num_nodes, F]
```

Recommended feature order:
1. latitude_norm
2. longitude_norm
3. depth_norm
4. magnitude_norm
5. significance_norm
6. log1p(time_since_prev_seconds)_norm
7. log1p(distance_from_prev_km)_norm
8. age_within_window_norm

#### `edge_index`
Directed COO tensor:
```text
[2, num_edges]
```
with `edge_index[0] = source`, `edge_index[1] = destination`.

Use only causal edges:
- source event time < destination event time

#### `edge_attr`
Shape:
```text
[num_edges, E]
```

Recommended feature order:
1. delta_t_days_norm
2. distance_km_norm
3. delta_depth_norm
4. delta_magnitude_norm

#### `y`
Choose one of:

**Event-level**
```text
[num_nodes]
```
Binary label for each event or only for the target node using a mask.

**Window-level**
```text
[1]
```
Binary label for the full graph window.

### Metadata to preserve outside normalized tensors
Keep a side table with:
- original timestamp
- original latitude/longitude
- original magnitude
- catalog row index
- split membership
- region if available

This is necessary for:
- debugging
- interpretation
- false alarm analysis
- mapping outputs back to real earthquake events

---

## 6) Dataset Split Strategy

### Important rule
Use a **temporal split**, not a random split.

Why:
- earthquake forecasting is time-dependent
- random shuffling leaks future seismic patterns into training
- temporal split better reflects real-time deployment

### Requested ratio
- **Train:** 85%
- **Validation:** 7.5%
- **Test:** 7.5%

### Split method
1. Sort by corrected event time ascending
2. Compute split indices on the sorted dataframe
3. Assign earliest 85% to train
4. Next 7.5% to validation
5. Final 7.5% to test

### Expected counts from the cleaned row count in the notebook
Cleaned dataset size shown in the notebook:
- **3,340,460 rows**

Requested split:
- Train = `3,340,460 * 0.85` = **2,839,391**
- Validation = `3,340,460 * 0.075` = **250,534**
- Test = remaining = **250,535**

### Split note
The exact validation / test counts may differ by 1 due to rounding.  
The counts above are the clean integer split that sums exactly to the dataset size.

---

## 7) Notebook-Ready Split Code

```python
# Fix timestamp first
if 'date' in df.columns:
    df['time'] = pd.to_datetime(df['date'], utc=True)
else:
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)

# Re-sort after fixing timestamps
df = df.sort_values('time').reset_index(drop=True)

# Recompute time_diff in seconds
df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(0)

# Optional: stabilize skewed temporal / distance features
df['time_diff_log'] = np.log1p(df['time_diff'])
df['dist_prev_log'] = np.log1p(df['dist_prev'])

n = len(df)
train_end = int(n * 0.85)
val_end = train_end + int(n * 0.075)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print("Total:", n)
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

train_df.to_parquet("train_earthquake.parquet", index=False)
val_df.to_parquet("val_earthquake.parquet", index=False)
test_df.to_parquet("test_earthquake.parquet", index=False)
```

---

## 8) Recommended Next Step for Member B

1. Fix timestamp parsing
2. Recompute temporal features
3. Freeze node and edge feature schema
4. Build graph windows as PyG `Data` objects
5. Train LSTM baseline first
6. Train ST-GAT model
7. Compare on:
   - Precision
   - Recall
   - F1
   - AUC-ROC
   - LTSS
   - inference latency

---

## 9) Final Verdict on the Notebook

### Will the current notebook work properly for the GNN?
**Not yet.**

### What must be fixed before it will work properly?
- correct timestamp parsing
- recompute `time_diff`
- build real graph edges instead of only `dist_prev`
- define labels explicitly
- create temporal split before modeling
- export graph-ready tensors / PyG objects

Once these are done, the notebook will be on the correct path for both:
- the **GAT + temporal model**
- the **LSTM baseline**
