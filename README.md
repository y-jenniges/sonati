# SONATI - Spatio-temporal Ocean Neighbourhood Asymmetric Transformer for Imputation

`sonati` is a Python package for imputing missing values in sparse spatiotemporal data. It implements **SONATI**, a transformer that attends separately to the k-nearest temporal, then spatial, neighbours of each query point and returns both a reconstructed value and a per-point uncertainty estimate.

The model was developed for oceanographic in-situ observations but applies to any point-cloud data with latitude, longitude, depth, and time coordinates.

## Installation

```bash
pip install sonati
```

Or directly from this repository:

```bash
pip install git+https://github.com/y-jenniges/sonati.git
```

For local development (editable install):

```bash
git clone https://github.com/y-jenniges/sonati.git
pip install -e sonati/
```

Requires Python ≥ 3.11 and PyTorch ≥ 2.0.

## Quick start

```python
import numpy as np
from sonati import SONATIImputer

# X      - (n_samples, n_features), NaN marks missing values
# coords - (n_samples, 4) as [latitude (°), longitude (°), depth (m), time]
imputer = SONATIImputer(
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

When `transform()` or `predict_with_uncertainty()` is called on the training data itself, the k-NN graph built during `fit()` is reused automatically instead of being rebuilt - worth on large datasets, where the graph build can take long.

To obtain per-point uncertainty estimates alongside imputed values, use `predict_with_uncertainty()`. It runs `n_inference` forward passes with dropout active (MC Dropout) to decompose uncertainty into aleatoric (data noise, from the variance head) and epistemic (model uncertainty, from the spread across passes) components:

```python
imputer = SONATIImputer(n_inference=50, loss_name="hetero")
imputer.fit(X_train, coords_train)
X_filled, aleatoric_std, epistemic_std = imputer.predict_with_uncertainty(X_test, coords_test)
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

| Parameter | Default | Description                                                                                                            |
|-----------|---------|------------------------------------------------------------------------------------------------------------------------|
| `n_neighbours_space` | 30 | Neighbours per query in the spatial graph                                                                              |
| `n_neighbours_time` | 30 | Neighbours per query in the temporal graph                                                                             |
| `d_model` | 64 | Transformer embedding dimension                                                                                        |
| `nhead` | 4 | Attention heads (must divide `d_model`)                                                                                |
| `n_spatial_layers` | 3 | Cross-attention layers over spatial neighbourhood                                                                      |
| `n_temporal_layers` | 3 | Cross-attention layers over temporal neighbourhood                                                                     |
| `dim_feedforward` | 128 | Feedforward sublayer width                                                                                             |
| `dropout` | 0.1 | Dropout rate (training only)                                                                                           |
| `fill_strategy` | `'zero'` | Fill unobserved positions with `'zero'` or `'mean'`                                                                    |
| `loss_name` | `'hetero'` | `'hetero'` (Gaussian NLL + variance head) or `'mse'`                                                                   |
| `max_epochs` | 100 | Training epoch limit                                                                                                   |
| `lr` | 1e-3 | Adam learning rate                                                                                                     |
| `batch_size` | 64 | Mini-batch size                                                                                                        |
| `masking` | `None` | Masking strategies applied during training. `None` defaults to `[RandomMask(ratio=0.5), CylindricalMask(radius=0.1, prob=1.0)]` |
| `val_method` | `'random'` | Internal validation split: `'random'` or `'block'` (geographic)                                                        |
| `val_indices` | `None` | Explicit validation row indices; overrides `val_method`/`val_frac`                                                     |
| `grad_clip_norm` | 1.0 | Maximum gradient norm (clipped before each optimiser step); stabilises heteroscedastic training, set to `0` to disable |
| `n_jobs` | `None` | Parallel jobs for the spatial k-NN graph build (sklearn convention: `None`=1, `-1`=all cores)                          |
| `patience` | 10 | Early-stopping patience (epochs)                                                                                       |
| `val_frac` | 0.2 | Internal validation fraction                                                                                           |
| `cyclic_time` | `False` | Cyclic sine/cosine encoding for the time column                                                                        |
| `device` | `'auto'` | `'cpu'`, `'cuda'`, or `'auto'`                                                                                         |
| `seed` | 42 | Random seed                                                                                                            |
| `n_inference` | 50 | MC Dropout forward passes used by `predict_with_uncertainty()`                                                         |

## Masking strategies

Available masking strategies for the `masking` argument: 

| Strategy | Hides | 
|----------|-------|
| `RandomMask(ratio=0.5)` | A random fraction of features, independently per point (row-wise) |
| `PerFeatureMask(ratio=0.5)` | A random fraction of points, independently per feature (column-wise) |
| `CylindricalMask(radius=0.1, prob=1.0)` | Every feature of every point inside a spatial cylinder of radius `radius` |
| `CopyMask(prob=1.0)` | Apply missingness of a randomly drawn training row, independently per point [1] |

[1] Gorla A, Wang R, Liu Z, An U, Sankararaman S. CACTI: Leveraging Copy Masking and Contextual Information to Improve Tabular Data Imputation. Proc Mach Learn Res. 2025 Jul;267:20187-20225. PMID: 41959945; PMCID: PMC13060770.

## Precomputed graphs for hyperparameter tuning

The k-NN graph depends only on the coordinates and the neighbour counts - not on feature values or model hyperparameters. On large datasets the graph build can dominate a trial's runtime, so for tuning it can be built once with the largest `k` under consideration and reused; `fit()` slices the first `n_neighbours_space` / `n_neighbours_time` columns per trial:

```python
from sonati import SONATIImputer, precompute_graph

graph = precompute_graph(coords, n_neighbours_space=60, n_neighbours_time=60)

for k in (15, 30, 45, 60):
    imputer = SONATIImputer(n_neighbours_space=k, n_neighbours_time=k)
    imputer.fit(X, coords, graph=graph)   # no k-NN build in the trial
```

## Examples

The `examples/` directory contains runnable end-to-end scripts:

- `real_world_test.py` - air-quality (Open-Meteo API) and traffic (METR-LA) imputation with public data downloaded on run.
<!-- - `jenniges_reproduction.py` - reproduces the sparse, time-resolved ocean imputation from the [companion research project](https://github.com/y-jenniges/time_imputation) on the preprocessed COMFORT dataset (dataset + train/test split hosted on Zenodo). This is a workflow, not a bit-exact, reproduction: an independently trained model agrees with the research pipeline only within predicted uncertainty, which the script quantifies via `agreement_within_uncertainty()`. Pass `--data-path` to run against a local CSV, `--split-path` to use the published split as the exact early-stopping monitor, and `--reference-path` to log agreement against a reference imputation.
-->

## Running the tests

```bash
pip install pytest
pytest tests/
```

## Citation

If you use this package, please cite:

```bibtex
@article{jenniges_inprep,
  title   = {{SONATI}: Spatio-temporal ocean neighbourhood asymmetric transformer for imputation of sparse, tabular data with uncertainty},
  author  = {Yvonne Jenniges and Boris Peter Koch and Maike Sonnewald and Sebastian Maneth},
  journal = {Environmental Data Science},
  year    = {in prep.},
}
```

## License

MIT
