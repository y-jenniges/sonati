# SONATI - Spatio-temporal Ocean Neighbourhood Asymmetric Transformer for Imputation

`sonati` is a Python package for imputing missing values in sparse spatio-temporal data. It implements **SONATI**, 
a transformer that attends separately to the k-nearest temporal, then j-nearest spatial neighbours of each query point and 
returns a filled value and a per-point aleatoric and epistemic uncertainty estimate.

The model was developed for oceanographic in-situ observations but applies to any point-cloud data 
with latitude, longitude, depth and time coordinates.

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

# X           - (n_samples, n_features), NaN marks missing values
# coords      - (n_samples, 4) as [latitude (°), longitude (°), depth (m), time]
# val_indices - row indices held out for early stopping (required), e.g. random or block hold-out
rng = np.random.default_rng(0)
val_indices = rng.choice(len(X), size=len(X) // 5, replace=False)

imputer = SONATIImputer(
    val_indices=val_indices,
    n_neighbours_space=30,
    n_neighbours_time=30,
    loss_name="hetero",     # heteroscedastic NLL - also trains a variance head
    max_epochs=100,
    device="auto",          # uses CUDA when available
)
X_filled = imputer.fit_transform(X, coords)
```

## Coordinate format

`coords` must have exactly four columns in this order:

| Column | Meaning | Unit |
|--------|---------|------|
| 0 | Latitude | Degrees (−90 ... 90) |
| 1 | Longitude | Degrees (−180 ... 180) |
| 2 | Depth | Metres |
| 3 | Time | Numeric year, or month 1–12 with `cyclic_time=True` |

Latitude and longitude are mapped internally to Cartesian unit-sphere coordinates. 
Depth, time and features are MinMax-scaled.
All inverse transforms are applied automatically before returning imputed values.

## Uncertainty estimates

`predict_with_uncertainty()` returns per-point epistemic uncertainty from `n_inference` forward passes with dropout 
active (MC Dropout). 
Aleatoric uncertainty requires `loss_name='hetero'` (or `'beta_nll'`); 
with `'mse'`, the variance head is untrained and `aleatoric_std` is returned as NaN.

```python
# All data goes into the imputer; test rows are hidden from training via test_idx
imputer = SONATIImputer(val_indices=val_indices, test_idx=test_idx, n_inference=50, loss_name="hetero")
imputer.fit(X, coords)
X_filled, aleatoric_std, epistemic_std = imputer.predict_with_uncertainty(X, coords)
```

## Graph building and reuse

The k-NN graphs depends only on the spatio-temporal location and is built in `fit()`.

- **Training data:** `transform()` and `predict_with_uncertainty()` on the training data itself reuse the graph 
  from `fit()` instead of rebuilding it, which saves time on large datasets (typical imputation use case).
- **Other data:** The graph is built from the passed rows only. `X_test` must therefore contain more than 
  `max(n_neighbours_space, n_neighbours_time)` rows, and points can only attend to other rows of `X_test`, 
  not to the training observations. 
  To keep held-out rows in the graph, pass them as `test_idx` at construction instead (recommended).
- **Hyperparameter tuning:** The graph build can dominate a trial's runtime. Build it once with the largest `k` 
  under consideration; `fit()` slices the first `n_neighbours_space` / `n_neighbours_time` columns per trial:

```python
from sonati import SONATIImputer, precompute_graph

graph = precompute_graph(coords, n_neighbours_space=60, n_neighbours_time=60)

for k in (15, 30, 45, 60):
    imputer = SONATIImputer(val_indices, n_neighbours_space=k, n_neighbours_time=k)
    imputer.fit(X, coords, graph=graph)  # no k-NN build in the trial
```

## Masking strategies

Strategies for the `masking` argument (default: `[RandomMask(ratio=0.5), CylindricalMask(radius=0.1, prob=1.0)]`):

| Strategy | Hides | 
|----------|-------|
| `RandomMask(ratio=0.5)` | A random fraction of features, independently per point (row-wise) |
| `PerFeatureMask(ratio=0.5)` | A random fraction of points, independently per feature (column-wise) |
| `CylindricalMask(radius=0.1, prob=1.0)` | Every feature of every point inside a spatial cylinder of radius `radius` |
| `CopyMask(prob=1.0)` | Apply missingness of a randomly drawn training row, independently per point [1] |

[1] Gorla A, Wang R, Liu Z, An U, Sankararaman S. CACTI: 
Leveraging Copy Masking and Contextual Information to Improve Tabular Data Imputation. 
Proc Mach Learn Res. 2025 Jul;267:20187-20225. PMID: 41959945; PMCID: PMC13060770.

## Hyperparameters

**Setup options** - set according to your data and hardware:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `val_indices` | Required | Row indices of `X` held out as internal validation set for early stopping |
| `test_idx` | `None` | Row indices withheld entirely from training (coordinates still used for the k-NN graph) |
| `cyclic_time` | `False` | Cyclic sine/cosine encoding for the time column |
| `max_epochs` / `patience` | 100 / 10 | Training epoch limit / early-stopping patience (epochs) |
| `n_inference` | 50 | MC Dropout forward passes used by `predict_with_uncertainty()` for epistemic uncertainties |
| `n_jobs` | `None` | Parallel jobs for the spatial k-NN graph build (sklearn convention: `None`=1, `-1`=all cores) |
| `device` | `'auto'` | `'cpu'`, `'cuda'`, or `'auto'` |

**Common hyperparameters to tune:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_neighbours_space` | 30 | Neighbours per query in the spatial graph |
| `n_neighbours_time` | 30 | Neighbours per query in the temporal graph |
| `d_model` | 64 | Transformer embedding dimension |
| `nhead` | 4 | Attention heads (must divide `d_model`) |
| `n_spatial_layers` / `n_temporal_layers` | 3 / 3 | Cross-attention layers over the spatial / temporal neighbourhood |
| `loss_name` | `'hetero'` | `'hetero'` (Gaussian NLL + variance head), `'beta_nll'` (beta-NLL, [2]) or `'mse'` |
| `masking` | `None` | Training masking strategies, see [Masking strategies](#masking-strategies) |

All remaining options (optimiser, loss options, checkpointing, DataLoader settings, ...) are documented 
in the `SONATIImputer` docstring.

[2] Maximilian Seitzer, Arash Tavakoli, Dimitrije Antic, Georg Martius: 
On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks. 
In International Conference on Learning Representations. 2022.

## Examples

The `examples/` directory contains runnable end-to-end scripts:

- `real_world_test.py`: Air-quality (Open-Meteo API) and traffic (METR-LA) imputation with public data downloaded on run.
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
