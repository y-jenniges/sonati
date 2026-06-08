# MaSTNeT — Masked Spatio-Temporal Neighbourhood Transformer

`mastnet` is a Python package for imputing missing values in sparse spatiotemporal data. It implements **MaSTNeT**, a transformer that attends separately to the k-nearest spatial and temporal neighbours of each query point and returns both a reconstructed value and a per-point uncertainty estimate.

The model was developed for oceanographic in-situ observations but applies to any point-cloud data with latitude, longitude, depth, and time coordinates.

## Installation

```bash
pip install mastnet-imputer
```

Or directly from this repository:

```bash
pip install git+https://github.com/y-jenniges/mastnet-imputer.git
```

For local development (editable install):

```bash
git clone https://github.com/y-jenniges/mastnet-imputer.git
pip install -e mastnet-imputer/
```

Requires Python ≥ 3.11 and PyTorch ≥ 2.0.

## Quick start

```python
import numpy as np
from mastnet import MaSTNetImputer

# X      — (n_samples, n_features), NaN marks missing values
# coords — (n_samples, 4) as [latitude (°), longitude (°), depth (m), time]
imputer = MaSTNetImputer(
    n_neighbours_space=30,
    n_neighbours_time=30,
    loss_name="hetero",   # heteroscedastic NLL — also trains a variance head
    max_epochs=100,
    device="auto",        # uses CUDA when available
)
X_filled = imputer.fit_transform(X, coords)
```

Fit and transform can be called separately, e.g. to impute data not seen during training:

```python
imputer.fit(X_train, coords_train)
X_test_filled = imputer.transform(X_test, coords_test)
```

## Coordinate format

`coords` must have exactly four columns in this order:

| Column | Meaning | Unit |
|--------|---------|------|
| 0 | Latitude | degrees (−90 … 90) |
| 1 | Longitude | degrees (−180 … 180) |
| 2 | Depth | metres |
| 3 | Time | numeric year, or month 1–12 with `cyclic_time=True` |

Latitude and longitude are mapped internally to Cartesian unit-sphere coordinates. Depth and time are MinMax-scaled. Features are scaled independently per variable on observed values only. All inverse transforms are applied automatically before returning imputed values.

## Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_neighbours_space` | 30 | Neighbours per query in the spatial graph |
| `n_neighbours_time` | 30 | Neighbours per query in the temporal graph |
| `d_model` | 64 | Transformer embedding dimension |
| `nhead` | 4 | Attention heads (must divide `d_model`) |
| `n_spatial_layers` | 3 | Cross-attention layers over spatial neighbourhood |
| `n_temporal_layers` | 3 | Cross-attention layers over temporal neighbourhood |
| `dim_feedforward` | 128 | Feedforward sublayer width |
| `dropout` | 0.1 | Dropout rate (training only) |
| `use_masks` | `True` | Append binary observation mask to each token |
| `fill_strategy` | `'zero'` | Fill unobserved positions with `'zero'` or `'mean'` |
| `loss_name` | `'hetero'` | `'hetero'` (Gaussian NLL + variance head) or `'mse'` |
| `max_epochs` | 100 | Training epoch limit |
| `lr` | 1e-3 | Adam learning rate |
| `batch_size` | 64 | Mini-batch size |
| `mask_ratio` | 0.5 | Fraction of observed values hidden during training |
| `patience` | 10 | Early-stopping patience (epochs) |
| `val_frac` | 0.2 | Internal validation fraction |
| `cyclic_time` | `False` | Cyclic sine/cosine encoding for the time column |
| `device` | `'auto'` | `'cpu'`, `'cuda'`, or `'auto'` |
| `seed` | 42 | Random seed |

## Running the tests

```bash
pip install pytest
pytest tests/
```

## Citation

If you use this package, please cite:

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO},
}
```

## License

MIT
