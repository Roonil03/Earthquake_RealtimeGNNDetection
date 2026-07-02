# Earthquake Realtime GNN Detection

An advanced Earthquake Foreshock Prediction pipeline leveraging Spatio-Temporal Graph Attention Networks (GATv2) and Bidirectional LSTMs. This repository has been structured for reproducibility, enabling researchers to process raw seismic catalogs into complex spatial subgraphs and train scalable neural networks for early warning systems.

## Documentation & Methodology

Please refer to the [Methodology](docs/methodology.md) for a comprehensive deep-dive into the spatio-temporal graphing approach, metric optimizations, and model architectures.

>  **Research Paper**: [Placeholder Link to Published Paper](to be updated soon)

## Repository Structure

The core pipeline has been formally encapsulated into two interactive Jupyter Notebooks:

- **[Earthquake_Prediction_Notebook.ipynb](Earthquake_Prediction_Notebook.ipynb)**: The primary evaluation notebook. This runs the final pipeline end-to-end on the complete dataset using the optimal configuration of hyperparameters to produce final metrics and visualizations.
- **[Earthquake_Prediction_Tuning.ipynb](Earthquake_Prediction_Tuning.ipynb)**: The hyperparameter simulation notebook. This encapsulates the 20-trial grid sweep used to discover the best parameters.

## Execution Requirements

To run the notebooks locally, install the automated Python environment using `uv`:

```bash
uv venv
uv pip install -r requirements.txt
uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
uv pip install torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
uv pip install jupyter ipykernel
```

*Note: The notebooks use `kagglehub` to download the raw datasets natively into the execution cell. Ensure you have an active internet connection when running for the first time.*

## Contributors

- [**Roonil03**](https://github.com/Roonil03)
- [**Aaryan Paranjape**](https://github.com/galactonebulose)
- [**SUPERSUPERSUPERuser**](https://github.com/SUPERSUPERSUPERuser)
