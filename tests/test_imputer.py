"""Functional tests for MaSTNetImputer."""

import numpy as np
import pytest
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


# Minimal hyperparameters for fast tests; verbose=False keeps test output clean
_FAST = dict(
    max_epochs=2, n_neighbours_space=10, n_neighbours_time=10,
    patience=2, d_model=16, dim_feedforward=32, n_inference=2,
    verbose=False,
)


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


def test_predict_with_uncertainty():
    # Check if prediction with and without uncertainty work as intended
    X, coords = _dummy_data()
    imputer = MaSTNetImputer(**_FAST)
    imputer.fit(X, coords)
    X_filled, aleatoric_std, epistemic_std = imputer.predict_with_uncertainty(X, coords)
    assert X_filled.shape == X.shape
    assert aleatoric_std.shape == X.shape
    assert epistemic_std.shape == X.shape
    assert not np.isnan(X_filled).any()

    # With n_inference=1, epistemic std must be zero (single deterministic pass)
    imputer_det = MaSTNetImputer(**{**_FAST, "n_inference": 1})
    imputer_det.fit(X, coords)
    _, _, epistemic_det = imputer_det.predict_with_uncertainty(X, coords)
    assert (epistemic_det == 0).all()


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


def test_val_method_block():
    # Check if block validation works
    X, coords = _dummy_data()
    X_filled = MaSTNetImputer(**_FAST, val_method="block").fit_transform(X, coords)
    assert X_filled.shape == X.shape
    assert not np.isnan(X_filled).any()


def test_val_indices():
    # Explicit validation indices take precedence over val_method
    X, coords = _dummy_data()
    val_idx = np.arange(0, len(X), 5)
    imputer = MaSTNetImputer(**_FAST, val_indices=val_idx, val_method="block")

    train_idx, split_val = imputer._make_train_val_split(coords)
    assert np.array_equal(np.sort(split_val), np.sort(val_idx))
    assert len(np.intersect1d(train_idx, split_val)) == 0
    assert len(train_idx) + len(split_val) == len(X)

    X_filled = imputer.fit_transform(X, coords)
    assert not np.isnan(X_filled).any()


def test_fill_strategy_mean():
    # Test mean filling
    X, coords = _dummy_data()
    X_filled = MaSTNetImputer(**_FAST, fill_strategy="mean").fit_transform(X, coords)
    assert X_filled.shape == X.shape
    assert not np.isnan(X_filled).any()


def test_sklearn_api():
    # Check sklearn compatibility
    from sklearn.base import clone
    imputer = MaSTNetImputer(**_FAST)
    params = imputer.get_params()
    assert "max_epochs" in params
    assert "verbose" in params
    cloned = clone(imputer)
    assert cloned.get_params() == params


def test_not_fitted_error():
    # Check compatibility with NotFittedError
    from sklearn.exceptions import NotFittedError
    imputer = MaSTNetImputer(**_FAST)
    X, coords = _dummy_data()
    with pytest.raises(NotFittedError):
        imputer.transform(X, coords)
