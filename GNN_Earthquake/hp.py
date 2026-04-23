from __future__ import annotations

LSTM_SPACE = {
    "window_size": [16, 24, 32, 40],
    "window_stride": [4, 8],
    "hidden_dim": [64, 96, 128, 160, 192],
    "num_layers": [1, 2],
    "dropout": [0.1, 0.2, 0.3],
    "batch_size": [256, 512, 1024],
    "epochs": [6, 8, 10],
    "patience": [2, 3],
    "lr": [3e-4, 5e-4, 1e-3],
    "weight_decay": [1e-5, 5e-5, 1e-4],
    "bidirectional": [True],
}

GAT_SPACE = {
    "graph_window_days": [30, 45, 60, 90],
    "edge_temporal_days": [15, 30, 45, 60],
    "edge_radius_km": [50.0, 75.0, 100.0, 125.0],
    "local_subgraph_radius_km": [100.0, 150.0, 200.0, 250.0],
    "max_nodes": [64, 96, 128],
    "min_nodes": [8, 12, 16],
    "graph_stride": [6, 8, 12],
    "max_train_graphs": [8000, 12000, 15000],
    "max_val_graphs": [2000, 2500, 3000],
    "max_test_graphs": [2000, 2500, 3000],
    "tcn_hidden": [32, 48, 64],
    "gat_hidden": [32, 48, 64, 96],
    "heads": [2, 4],
    "dropout": [0.15, 0.25, 0.35],
    "batch_size": [32, 64, 96],
    "epochs": [6, 8, 10],
    "patience": [2, 3, 4],
    "lr": [5e-4, 1e-3, 2e-3],
    "weight_decay": [1e-5, 5e-5, 1e-4, 5e-4],
}


def _decode_index(index: int, space: dict[str, list]) -> dict:
    remaining = index
    config = {}
    for key, values in space.items():
        config[key] = values[remaining % len(values)]
        remaining //= len(values)
    return config


def _normalize_gat(config: dict) -> dict:
    config = dict(config)
    if config["min_nodes"] >= config["max_nodes"]:
        config["min_nodes"] = 8 if config["max_nodes"] > 8 else 4
    if config["local_subgraph_radius_km"] < config["edge_radius_km"]:
        config["local_subgraph_radius_km"] = config["edge_radius_km"]
    return config


def _build_hyperparameter_sets(limit: int = 200) -> list[dict]:
    sets = []
    for index in range(limit):
        lstm = _decode_index(index, LSTM_SPACE)
        gat = _normalize_gat(_decode_index(index * 7, GAT_SPACE))
        sets.append(
            {
                "trial_index": index + 1,
                "lstm": lstm,
                "gat": gat,
            }
        )
    return sets


HYPERPARAMETER_SETS = _build_hyperparameter_sets(20)
DEFAULT_HYPERPARAMETER_SET = HYPERPARAMETER_SETS[0]
