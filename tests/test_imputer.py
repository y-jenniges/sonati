"""Functional tests for MaSTNetImputer."""

import logging

import numpy as np
import pytest
import torch

from mastnet import MaSTNetImputer, precompute_graph


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


def test_logvar_clamped():
    # An extreme logvar head must be clamped so exp(logvar) cannot overflow
    from mastnet._model import LOGVAR_CLAMP, MaSTNeT
    model = MaSTNeT(n_features=2, coord_dim=5, d_model=16, dim_feedforward=32)
    with torch.no_grad():
        model.logvar_head.bias.fill_(1e4)

    B, K = 4, 5
    batch = {
        "query_features": torch.randn(B, 2),
        "query_mask": torch.ones(B, 2, dtype=torch.bool),
        "query_coords": torch.rand(B, 5),
    }
    for scope in ("space", "time"):
        batch[scope] = {
            "features": torch.randn(B, K, 2),
            "mask": torch.ones(B, K, 2, dtype=torch.bool),
            "coords": torch.rand(B, K, 5),
            "rel_positions": torch.rand(B, K, 5),
        }

    _, pred_logvar = model(batch)
    assert (pred_logvar <= LOGVAR_CLAMP).all()
    assert (pred_logvar >= -LOGVAR_CLAMP).all()
    assert torch.isfinite(pred_logvar.exp()).all()


def test_fit_aborts_on_nonfinite_loss(monkeypatch, caplog):
    # A non-finite epoch loss must abort training instead of burning patience
    from mastnet import _training

    monkeypatch.setattr(_training, "_train_epoch", lambda *a, **k: float("nan"))
    monkeypatch.setattr(_training, "_eval_epoch", lambda *a, **k: float("nan"))

    model = torch.nn.Linear(1, 1)
    with caplog.at_level(logging.INFO, logger="mastnet._training"):
        _training.fit(
            model, train_loader=[], val_loader=[],
            max_epochs=10, lr=1e-3, mask_ratio=0.5, patience=10,
            device=torch.device("cpu"), verbose=False,
        )

    epoch_lines = [r for r in caplog.records if "| train=" in r.getMessage()]
    assert len(epoch_lines) == 1  # aborted after the first epoch
    assert any("Non-finite loss" in r.getMessage() for r in caplog.records)


def test_precomputed_graph():
    # A graph built once with a larger k can be sliced and reused by fit
    X, coords = _dummy_data()
    graph = precompute_graph(coords, n_neighbours_space=15, n_neighbours_time=15)
    assert graph["space"].shape == (len(X), 15)

    imputer = MaSTNetImputer(**_FAST)  # n_neighbours=10 < 15
    X_filled = imputer.fit_transform(X, coords, graph=graph)
    assert X_filled.shape == X.shape
    assert not np.isnan(X_filled).any()
    assert imputer.graph_["space"].shape == (len(X), 10)
    assert imputer.graph_["time"].shape == (len(X), 10)
    # Sliced columns must come from the precomputed graph
    assert torch.equal(imputer.graph_["space"], graph["space"][:, :10])


def test_neighbour_values_hidden():
    # Values NaN-ed in neighbour_values must be invisible as neighbour features
    # while the same rows remain fully usable as queries (targets/inputs)
    from mastnet._graph import NeighbourDataset

    coords = torch.rand(4, 5)
    values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    graph = {"space": torch.tensor([[1], [0], [1], [1]]),
             "time": torch.tensor([[2], [2], [0], [0]])}

    neighbour_values = values.clone()
    neighbour_values[1] = float("nan")  # hide row 1 from neighbourhoods
    ds = NeighbourDataset(coords, values, graph, neighbour_values=neighbour_values)

    sample = ds[0]  # spatial neighbour of query 0 is the hidden row 1
    assert torch.isnan(sample["space"]["features"]).all()
    assert not sample["space"]["mask"].any()
    # Row 1 as a *query* still exposes its own values
    assert torch.equal(ds[1]["query_features"], values[1])
    assert ds[1]["query_mask"].all()
    # Unhidden neighbours are unaffected (temporal neighbour of query 0 is row 2)
    assert torch.equal(sample["time"]["features"][0], values[2])


def test_transform_reuses_training_graph(monkeypatch):
    # transform on the training data must reuse self.graph_ instead of rebuilding
    from mastnet import imputer as imputer_module

    X, coords = _dummy_data()
    imputer = MaSTNetImputer(**_FAST)
    imputer.fit(X, coords)

    def _fail(*args, **kwargs):
        raise AssertionError("build_graph should not be called for training coords")

    monkeypatch.setattr(imputer_module, "build_graph", _fail)
    X_filled = imputer.transform(X, coords)  # same coords -> reuse, no rebuild
    assert not np.isnan(X_filled).any()

    with pytest.raises(AssertionError, match="should not be called"):
        imputer.transform(X[:100], coords[:100])  # new coords -> rebuild attempted


def test_precomputed_graph_validation():
    # Invalid precomputed graphs must raise informative errors
    X, coords = _dummy_data()
    imputer = MaSTNetImputer(**_FAST)

    with pytest.raises(ValueError, match="missing the 'time' key"):
        imputer.fit(X, coords, graph={"space": torch.zeros(len(X), 15, dtype=torch.long)})

    too_few = precompute_graph(coords, n_neighbours_space=5, n_neighbours_time=15)
    with pytest.raises(ValueError, match="Rebuild it with a larger k"):
        imputer.fit(X, coords, graph=too_few)

    wrong_n = precompute_graph(coords[:100], n_neighbours_space=15, n_neighbours_time=15)
    with pytest.raises(ValueError, match="expected"):
        imputer.fit(X, coords, graph=wrong_n)
