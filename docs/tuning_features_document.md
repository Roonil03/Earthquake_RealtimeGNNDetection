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
