"""Smoke tests — verify the package installs and runs end-to-end."""

import numpy as np
from mastnet import MaSTNetImputer


def _dummy_data(n: int = 300, n_features: int = 3, seed: int = 0):
    # Create random data with nan
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(float)
    X[rng.random((n, n_features)) < 0.25] = np.nan

    # Add coordinates
    coords = np.column_stack([
        rng.uniform(-90, 90, n),
        rng.uniform(-180, 180, n),
        rng.uniform(0, 1000, n),
        rng.integers(2000, 2020, n).astype(float),
    ])
    return X, coords


# Hyperparameters for a quick, low-capacity test
_FAST = dict(max_epochs=2, n_neighbours_space=10, n_neighbours_time=10, patience=2, d_model=16, dim_feedforward=32)


def test_import():
    # Check import
    import mastnet
    assert hasattr(mastnet, "__version__")
    assert mastnet.MaSTNetImputer is MaSTNetImputer


def test_fit_transform():
    # Check fit_transform and output shapes and if nans were imputed
    X, coords = _dummy_data()
    X_filled = MaSTNetImputer(**_FAST).fit_transform(X, coords)
    assert X_filled.shape == X.shape
    assert not np.isnan(X_filled).any()


def test_fit_then_transform():
    # Check fit, then transform
    X, coords = _dummy_data()
    imputer = MaSTNetImputer(**_FAST)
    imputer.fit(X, coords)
    X_filled = imputer.transform(X, coords)
    assert X_filled.shape == X.shape
    assert not np.isnan(X_filled).any()


def test_observed_values_unchanged():
    # Check if observed values remain unchanged
    X, coords = _dummy_data()
    imputer = MaSTNetImputer(**_FAST)
    X_filled = imputer.fit_transform(X, coords)
    observed = ~np.isnan(X)
    np.testing.assert_array_equal(X_filled[observed], X[observed])


def test_cyclic_time():
    # Random data with cyclic time
    rng = np.random.default_rng(1)
    n = 300
    X = rng.standard_normal((n, 2)).astype(float)
    X[rng.random((n, 2)) < 0.3] = np.nan
    coords = np.column_stack([
        rng.uniform(-90, 90, n),
        rng.uniform(-180, 180, n),
        rng.uniform(0, 500, n),
        rng.integers(1, 13, n).astype(float),  # months 1-12
    ])
    X_filled = MaSTNetImputer(**_FAST, cyclic_time=True).fit_transform(X, coords)
    assert not np.isnan(X_filled).any()
