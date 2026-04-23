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
