# Visualization summary
This folder was generated from the uploaded training/test logs.
## Best trial per model (ranked by F1, then AUC, then latency)
| model   |   trial_index |   precision |   recall |       f1 |   accuracy |   balanced_accuracy |   auc_roc |   tpr_at_20_fpr |   ms_per_event |
|:--------|--------------:|------------:|---------:|---------:|-----------:|--------------------:|----------:|----------------:|---------------:|
| GAT     |             3 |   0.261056  | 0.816964 | 0.395676 |    0.7205  |            0.762649 |  0.846648 |        0.78125  |     0.330715   |
| LSTM    |             4 |   0.0730012 | 0.98239  | 0.135903 |    0.09375 |            0.503316 |  0.537301 |        0.252256 |     0.00473887 |
## Generated charts
- 01_* : SeismoQuakeGNN-style epoch curves
- 02_* : across-trial metric summaries
- 03_* : PreD-Net-style benchmark bars for Precision/Recall/F1/AUC
- 04_* : Wang-style grouped accuracy bars
- 05_* : Zlydenko-style operating-point comparison at fixed FPR
- 06_* : latency vs F1 Pareto frontier
- 07_* : ranked trials
- 08_* : per-model test-metric heatmaps
- 09_* : pairplot-style hyperparameter scatter matrices
- 10_* : AUC vs TPR@20%FPR

## Important limitations
- Full ROC curves cannot be reconstructed because the logs contain only AUC and a fixed operating point, not raw prediction scores.
- Actual-vs-predicted magnitude traces cannot be recreated because the logs do not include per-sample predictions.
- Spatial seismicity maps and event timelines cannot be recreated because the logs do not include latitude/longitude/time per event.
