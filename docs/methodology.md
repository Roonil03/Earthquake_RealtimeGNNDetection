# Summarized Methodology and other Documentation

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
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn scipy matplotlib seaborn pyarrow joblib tqdm networkx numba fastapi uvicorn pydantic requests torchmetrics
pip install torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
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
| Paper                                             |                                          Closest metric to your project |                                                                                                                                                                         Exact figure to beat | How relevant it is                                                                                                                                           |
| ------------------------------------------------- | ----------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Convertito et al. (2024), PreD-Net**            | Precision / Recall / F1 / AUC on precursor-vs-background classification |                                                                                                           **Test-total:** Precision **0.851**, Recall **0.838**, F1 **0.839**, AUC **0.758** | **Best direct benchmark** for your foreshock/precursor classification task ([PMC][1])                                                                        |
| **Convertito et al. (2024), The Geysers test**    |                                           Precision / Recall / F1 / AUC |                                                                                                                           Precision **0.926**, Recall **0.924**, F1 **0.923**, AUC **0.817** | Strongest single-site target in the list, but easier than cross-region generalization ([PMC][1])                                                             |
| **Convertito et al. (2024), Cooper Basin test**   |                                           Precision / Recall / F1 / AUC |                                                                                                                           Precision **0.852**, Recall **0.819**, F1 **0.831**, AUC **0.773** | Good medium-strength benchmark ([PMC][1])                                                                                                                    |
| **Convertito et al. (2024), Hengill test**        |                                           Precision / Recall / F1 / AUC |                                                                                                                           Precision **0.829**, Recall **0.803**, F1 **0.817**, AUC **0.762** | Useful out-of-site comparison ([PMC][1])                                                                                                                     |
| **Convertito et al. (2024), Basel external test** |                                           Precision / Recall / F1 / AUC |                                                                                                                           Precision **0.782**, Recall **0.783**, F1 **0.765**, AUC **0.684** | Most relevant **external generalization floor** to beat ([PMC][1])                                                                                           |
| **Zlydenko et al. (2023), FERN vs ETAS**          |                                                   Improvement over ETAS |                                            **>4%** improvement in information gain per earthquake; with added low-magnitude events, **4–12%** IGPE gain; about **0.1 bits/event** on average | Best proxy for your **LTSS-style** ETAS comparison ([Nature][2])                                                                                             |
| **Zlydenko et al. (2023), short-term forecast**   |                                               ROC-style operating point |                                                                                                             In Region C at **20% FPR**, ETAS gives **80% TPR** while FERN+ gives **90% TPR** | Strongest accessible ROC-style baseline against ETAS in your list ([Nature][2])                                                                              |
| **Zlydenko et al. (2023), efficiency**            |                                                Computational efficiency |                                                                                                                     **>1000-fold runtime improvement** over ETAS-style short-term prediction | Best efficiency target in the list ([Nature][2])                                                                                                             |
| **Wang et al. (2020), LSTM**                      |               Accuracy baseline for spatio-temporal sequence prediction |  Two-dimensional LSTM: **74.81%** overall accuracy, **68.56%** true-positive accuracy, **81.31%** true-negative accuracy; decomposed model: **85.12%** overall, **77.07%** TP, **93.49%** TN | Good **LSTM baseline** to beat, though not a foreshock-sequence benchmark with your exact metrics ([Case School of Engineering][3])                          |
| **Saad et al. (2021)**                            |                Precision / Recall / F1 for EEW parameter classification | Location classifier: Precision **94.79%**, Recall **94.58%**, F1 **94.60%**; 10-fold average accuracies: Location **89.85%**, Magnitude **91.72%**, Depth **92.49%**, Origin time **91.52%** | Useful for EEW-style classification quality, but **not foreshock detection** ([Scribd][4])                                                                   |
| **SeismoQuakeGNN (2025)**                         |                        Accuracy / MSE / (R^2) for earthquake prediction |                                                    SeismoQuakeGNN: **98.00% accuracy**, **0.07 MSE**, **88.00% (R^2)**; LSTM baseline: **97.45% accuracy**, **0.1245 MSE**, **77.19% (R^2)** | Helpful as a **GNN-vs-LSTM** performance reference, but authors warn the 98% accuracy is on predefined magnitude classes, not your foreshock task ([PMC][5]) |
| **Zhang et al. (2022), STGNN**                    |                                            Localization / magnitude MAE |                                                   Oklahoma: latitude MAE **3.574 km**, longitude MAE **3.697 km**, magnitude MAE **0.154**; better than GNN baseline magnitude MAE **0.195** | Strong evidence that graph models help, but metrics are **not your target metrics** ([PMC][6])                                                               |
| **Vikraman (2016)**                               |                                                           Accuracy only |                                                                                                   **99% classification accuracy** for foreshock/mainshock/aftershock waveform classification | Tempting headline number, but it is waveform classification and not a catalog-graph foreshock benchmark with precision/recall/AUC breakdown ([Paperzz][7])   |

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10844195/ "
            Deep learning forecasting of large induced earthquakes via precursory signals - PMC
        "
[2]: https://www.nature.com/articles/s41598-023-38033-9 "A neural encoder for earthquake rate forecasting | Scientific Reports"
[3]: https://engineering.case.edu/sites/default/files/tetc0417.pdf "TETC2699169.pdf"
[4]: https://www.scribd.com/document/778209578/Deep-Learning-Approach-for-Earthquake-Parameters-Classification-in-Earthquake-Early-Warning-System?utm_source=chatgpt.com "Deep Learning Approach For Earthquake Parameters Classification in Earthquake Early Warning System | PDF"
[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12706585/ "
            SeismoQuakeGNN: a hybrid framework for spatio-temporal earthquake prediction with transformer-enhanced models - PMC
        "
[6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10078111/?utm_source=chatgpt.com "Spatiotemporal Graph Convolutional Networks for Earthquake Source Characterization - PMC"
[7]: https://paperzz.com/doc/8165519/a-deep-neural-network-to-identify-foreshocks-in-real-time?utm_source=chatgpt.com "A Deep Neural Network to identify foreshocks in real time"


| Your metric                   |                                                                                                                                                                        Best paper-derived target to beat |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Precision                     |                                                                                                                       **0.851** on an overall precursor test set; **0.926** on the best single-site test |
| Recall                        |                                                                                                                                                **0.838** overall; **0.924** on the best single-site test |
| F1-score                      |                                                                                                                                                **0.839** overall; **0.923** on the best single-site test |
| AUC-ROC                       |                                                                                                                                                **0.758** overall; **0.817** on the best single-site test |
| LTSS / ETAS improvement proxy |                                                                                                                                  **>4%** improvement over ETAS, with strong cases in the **4–12%** range |
| ROC operating point vs ETAS   |                                                                                                                                                                         At **20% FPR**, beat **90% TPR** |
| Computational efficiency      | Aim to be comfortably under your **<100 ms/event** target and, ideally, show a very large gain versus ETAS-style simulation baselines; the closest number in your list is **>1000× runtime improvement** |


## List of Papers:

(1) Graph Convolution Networks for Seismic Events Classification Using Raw Waveform Data from Multiple Stations
- Korea University Pure page: https://pure.korea.ac.kr/en/publications/graph-convolution-networks-for-seismic-events-classification-usin/
- DOI: https://doi.org/10.1109/LGRS.2021.3127874

(2) Real-Time Seismic Intensity Prediction Using Self-Supervised Contrastive GNN for Earthquake Early Warning
- DOI: https://doi.org/10.1109/TGRS.2024.3373643
- arXiv: https://arxiv.org/abs/2306.14336
- IEEE Xplore link provided: https://ieeexplore.ieee.org/document/10459332

(3) A Deep Neural Network to identify foreshocks in real time
- arXiv: https://arxiv.org/abs/1611.08655

(4) Machine learning predicts meter-scale laboratory earthquakes
- Nature Communications page: https://www.nature.com/articles/s41467-025-64542-4
- DOI: https://doi.org/10.1038/s41467-025-64542-4

(5) Earthquake Prediction Based on Spatio-Temporal Data Mining: An LSTM Network Approach
- DOI: https://doi.org/10.1109/TETC.2017.2699169
- Accessible PDF used above: https://engineering.case.edu/sites/default/files/tetc0417.pdf

(6) Development of a Long Short-Term Memory (LSTM)-Based Statistical Model for Earthquake Forecasting in Central Asia
- DOI: https://doi.org/10.1109/ACCESS.2025.3610168
- DBLP record: https://dblp.org/rec/journals/access/NurtasAYVN25

(7) Seismic and Geospatial Feature Integration for Earthquake Magnitude Prediction Using Machine Learning
- ResearchGate metadata page: https://www.researchgate.net/publication/399962406_Seismic_and_Geospatial_Feature_Integration_for_Earthquake_Magnitude_Prediction_Using_Machine_Learning
- DOI: https://doi.org/10.1109/CISCON66933.2025.11337796

(8) Deep Learning Approach for Earthquake Parameters Classification in Earthquake Early Warning System
- DOI: https://doi.org/10.1109/LGRS.2020.2998580
- Code repo referencing the paper: https://github.com/omarmohamed15/Deep-learning-for-earthquake-parameters-classification-in-EEW

(9) Application of Artificial Intelligence in predicting earthquakes: state-of-the-art and future challenges
- DOI: https://doi.org/10.1109/ACCESS.2020.3029859
- NTU repository landing page: https://irep.ntu.ac.uk/id/eprint/41423/

(10) Advances in Deep Learning for Earthquake Monitoring and Forecasting: Techniques, Applications, and Future Directions
- DOI: https://doi.org/10.1109/ICSESS62520.2024.10719051
- ResearchGate full text page: https://www.researchgate.net/publication/385199228_Advances_in_Deep_Learning_for_Earthquake_Monitoring_and_Forecasting_Techniques_Applications_and_Future_Directions

(11) Automated Seismic Source Characterisation Using Deep Graph Neural Networks
- DOI (AGU): https://doi.org/10.1029/2020GL088690
- CaltechAUTHORS landing page: https://authors.library.caltech.edu/records/yjdma-6as79

(12) Spatiotemporally explicit earthquake prediction using deep neural network
- ScienceDirect landing page: https://www.sciencedirect.com/science/article/pii/S0267726121000853
- DOI: https://doi.org/10.1016/j.soildyn.2021.106663

(13) Spatiotemporal Graph Convolutional Networks for Earthquake Source Characterization
- DOI: https://doi.org/10.1029/2022JB024401
- OSTI metadata page: https://www.osti.gov/pages/biblio/1896917

(14) Earthquake Phase Association with Graph Neural Networks
- Journal landing page: https://pubs.geoscienceworld.org/ssa/bssa/article/113/2/524/619845/Earthquake-Phase-Association-with-Graph-Neural
- arXiv: https://arxiv.org/abs/2209.07086

(15) Deep learning forecasting of large induced earthquakes via precursory signals
- Nature Scientific Reports landing page: https://www.nature.com/articles/s41598-024-52935-2
- DOI: https://doi.org/10.1038/s41598-024-52935-2

(16) SeismoQuakeGNN: a hybrid framework for spatio-temporal earthquake prediction with transformer-enhanced models
- Frontiers landing page: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1690476/full
- DOI: https://doi.org/10.3389/frai.2025.1690476

(17) Spatio-Temporal Graph Structure Learning for Earthquake Detection
- arXiv: https://arxiv.org/abs/2503.11215

(18) EPBench: A Benchmark for Short-term Earthquake Prediction with Neural Networks
- arXiv: https://arxiv.org/abs/2505.15588

(19) Ranking Earthquake Prediction Algorithms: A Comprehensive Review of Machine Learning and Deep Learning Methods
- ScienceDirect landing page: https://www.sciencedirect.com/science/article/pii/S0267726125005330
- DOI: https://doi.org/10.1016/j.soildyn.2025.109740

(20) Introduction to Graph Neural Networks: A Starting Point for Machine Learning Engineers
- arXiv: https://arxiv.org/abs/2412.19419

(21) Time Series Foundation Models and Deep Learning Architectures for Earthquake Temporal and Spatial Nowcasting
- arXiv: https://arxiv.org/abs/2408.11990

(22) Double difference earthquake location with graph neural networks
- Springer Nature landing page: https://link.springer.com/article/10.1186/s40623-025-02251-4
- DOI: https://doi.org/10.1186/s40623-025-02251-4
- arXiv preprint: https://arxiv.org/abs/2410.19323

(23) ScienceDirect link provided without accessible metadata (403 here)
- URL as provided: https://www.sciencedirect.com/science/article/pii/S009830042500124

(24) INSTANCE – the Italian seismic dataset for machine learning
- ESSD landing page: https://essd.copernicus.org/articles/13/5509/2021/
- DOI: https://doi.org/10.5194/essd-13-5509-2021

(25) Enhancing Earthquake Forecasting (PhD thesis)
- University of Bristol landing page: https://research-information.bris.ac.uk/en/studentTheses/enhancing-earthquake-forecasting/

(26) California earthquake dataset for machine learning and cloud computing
- arXiv: https://arxiv.org/abs/2502.11500

(27) A neural encoder for earthquake rate forecasting
- Nature Scientific Reports landing page: https://www.nature.com/articles/s41598-023-38033-9
- DOI: https://doi.org/10.1038/s41598-023-38033-9

(28) SwissRe Historical Earthquake Statistics Dataset
- GitHub repository: https://github.com/SwissRe/Historical-Earthquake-Statistics-Dataset

(29) All the Earthquakes Dataset: from 1990-2023 (project primary dataset reference)
- Kaggle landing page (dynamic site): https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023

(30) USGS real-time feeds and catalog APIs (for operational ingestion)
- Event Web Service docs: https://earthquake.usgs.gov/fdsnws/event/1/
- Feeds & Notifications: https://earthquake.usgs.gov/earthquakes/feed/v1.0/
- ComCat documentation: https://earthquake.usgs.gov/data/comcat/

## Division of what the papers do:
### Why GNNs are the RIGHT choice?
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1611.08655)
- [van den Ende & Ampuero (2020) — Deep GNNs for seismic source characterization](https://doi.org/10.1029/2020GL088690)
- [Zhang et al. (2022) — Spatio-temporal GCNs](https://doi.org/10.1029/2022JB024401)
- [McBrearty & Beroza (2023) — GNN for phase association](https://arxiv.org/abs/2209.07086)
- [Leema et al. (2025) — SeismoQuakeGNN (Transformer + GNN)](https://doi.org/10.3389/frai.2025.1690476)
- [Piriyasatit et al. (2025) — Graph structure learning](https://arxiv.org/abs/2503.11215)
- [Double-difference earthquake location with GNNs (Springer, 2025)](https://doi.org/10.1186/s40623-025-02251-4)
- [Real-Time Seismic Intensity Prediction using Contrastive GNN (IEEE 2024)](https://arxiv.org/abs/2306.14336)
- [Graph Convolution Networks for Seismic Event Classification (Korea Univ.)](https://doi.org/10.1109/LGRS.2021.3127874)

### Why GNNs outperform LSTM / traditional models?
- [Leema et al. (2025) — hybrid GNN + Transformer](https://doi.org/10.3389/frai.2025.1690476)
- [Zhang et al. (2022) — spatio-temporal GCN](https://doi.org/10.1029/2022JB024401)
- [Contrastive GNN (IEEE 2024)](https://arxiv.org/abs/2306.14336)
- [Time Series Foundation Models (arXiv 2024)](https://arxiv.org/abs/2408.11990)
- [Zlydenko et al. (2023) — neural encoder for earthquake rates](https://doi.org/10.1038/s41598-023-38033-9)

### Previous ML (NON-GNN) Earthquake Prediction:
- [Wang et al. (2020) — LSTM prediction](https://engineering.case.edu/sites/default/files/tetc0417.pdf)
- [Nurtas et al. (2025) — LSTM forecasting](https://doi.org/10.1109/ACCESS.2025.3610168)
- [Yousefzadeh et al. (2021) — DNN spatiotemporal prediction](https://doi.org/10.1016/j.soildyn.2021.106663)
- [Convertito et al. (2024) — precursory signals DL](https://doi.org/10.1038/s41598-024-52935-2)
- [Saad et al. (2021) — parameter classification](https://doi.org/10.1109/LGRS.2020.2998580)
- [Padmashree et al. (2025) — geospatial + ML](https://doi.org/10.1109/CISCON66933.2025.11337796)
- [Al Banna et al. (2020) — survey](https://doi.org/10.1109/ACCESS.2020.3029859)
- [Wan et al. (2024) — survey](https://doi.org/10.1109/ICSESS62520.2024.10719051)
- [Stockman (2024) — forecasting thesis](https://research-information.bris.ac.uk/en/studentTheses/enhancing-earthquake-forecasting/)
- [Scientific Reports review (2025) — ranking ML models](https://doi.org/10.1016/j.soildyn.2025.109740)

### Foreshock / Precursory Signal Learning:
- [Norisugi et al. (2025) — lab-scale earthquake prediction](https://doi.org/10.1038/s41467-025-64542-4)
- [Convertito et al. (2024) — precursory signals](https://doi.org/10.1038/s41598-024-52935-2)
- [EPBench (2025) — benchmark for short-term prediction](https://arxiv.org/abs/2505.15588)

### Real-Time / Early Warning Systems:
- [Saad et al. (2021) — early warning classification](https://doi.org/10.1109/LGRS.2020.2998580)
- [Contrastive GNN (IEEE 2024) — real-time intensity prediction](https://doi.org/10.1109/TGRS.2024.3373643)
- [TGRS 2024 paper](https://ieeexplore.ieee.org/document/10459332)
- [Time-series nowcasting (arXiv 2024)](https://arxiv.org/abs/2408.11990)

### General GNN / ML Foundations:
- [Intro to GNN (arXiv 2024)](https://arxiv.org/abs/2412.19419)
- [Time Series Foundation Models (2024)](https://arxiv.org/abs/2408.11990)
# Tuning Experiments to Try

## Phase 1: Fast baseline sweep
1. LSTM hidden size: 64, 128, 256
2. LSTM layers: 1, 2, 3
3. Dropout: 0.1, 0.3, 0.5
4. Sequence length: 16, 32, 64
5. Learning rate: 1e-3, 5e-4, 1e-4
6. Batch size: 128, 256, 512
7. Loss: BCE vs focal loss

## Phase 2: Core GAT sweep
1. Hidden channels: 64, 128, 256
2. Heads: 2, 4, 8
3. GAT layers: 2, 3, 4
4. Dropout: 0.1, 0.2, 0.4
5. Temporal encoding dim: 8, 16, 32
6. TCN kernel size: 3, 5
7. TCN dilation schedule: [1,2], [1,2,4]

## Phase 3: Graph construction sweep
1. Spatial threshold: 50 km, 100 km, 150 km
2. Temporal threshold: 30 days, 60 days, 90 days
3. Max nodes per graph: 64, 128, 256
4. Edge features:
   - distance only
   - distance + delta time
   - distance + delta time + magnitude difference

## Phase 4: Imbalance and threshold sweep
1. `pos_weight`: inverse frequency, clipped inverse frequency, square-root inverse frequency
2. Focal gamma: 1.0, 2.0, 3.0
3. Decision threshold: 0.3 to 0.8 in steps of 0.05

## Phase 5: Readout and head ablation
1. Global mean pooling
2. Global max pooling
3. Attention pooling
4. MLP head depth: 1, 2, 3 layers

## Recommended order
- First optimize LSTM to establish a solid baseline.
- Then optimize graph construction.
- Then tune GAT architecture.
- Then tune threshold and calibration for best precision at fixed recall.
- Finally compare 7-day, 14-day, and 30-day lead-time settings separately.

## What to log for every run
- Train/validation loss
- Precision
- Recall
- F1
- ROC-AUC
- LTSS
- Inference latency per sample
- Number of parameters
- Peak GPU memory
# Hyperparameter Tuning Features for Earthquake Foreshock Detection

## Purpose
This document lists the tuning knobs that matter most for the LSTM baseline and the GAT + temporal model.

## 1. Data and labeling features
- Sequence window length for LSTM (`SEQ_LEN`)
- Graph time horizon in days (`GRAPH_LOOKBACK_DAYS`)
- Maximum nodes per graph (`MAX_GRAPH_NODES`)
- Mainshock magnitude threshold
- Foreshock lead-time bucket (7, 14, 30 days)
- Spatial edge threshold in km
- Temporal edge threshold in days
- Positive class weighting strategy
- Normalization strategy (`StandardScaler`, `RobustScaler`)

## 2. LSTM tuning features
- Hidden size
- Number of LSTM layers
- Dropout
- Bidirectional vs unidirectional LSTM
- Sequence aggregation method (last state, mean pool, attention pool)
- Learning rate
- Batch size
- Optimizer (`Adam`, `AdamW`)
- Weight decay
- Gradient clipping threshold
- Epoch count and early stopping patience

## 3. GAT + temporal model tuning features
- Number of GAT layers
- Hidden channel size
- Number of attention heads
- Dropout on attention and hidden layers
- Residual connections
- Edge attribute dimension
- Temporal encoding dimension
- Temporal convolution kernel size
- Temporal convolution dilation
- Readout type (mean, max, attention pooling)
- Classification head depth
- Batch size for graph mini-batches
- Neighbor sampling / subgraph sampling strategy
- Learning rate and scheduler

## 4. Loss and class imbalance tuning
- `BCEWithLogitsLoss` with `pos_weight`
- Focal loss gamma and alpha
- Label smoothing
- Threshold tuning for final decision boundary

## 5. Evaluation tuning features
- Probability threshold for precision/recall tradeoff
- Lead-time specific thresholds
- Magnitude-threshold specific calibration
- Calibration method (Platt scaling, isotonic regression)
- LTSS baseline definition and metric basis

## 6. Efficiency tuning features
- Mixed precision training
- Number of workers in data loaders
- Pin memory
- Graph batch size
- Max graph nodes per sample
- Sparse adjacency / neighbor cutoffs
