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


def test_predict_with_uncertainty_realisations():
    # return_realisations=True must expose the individual MC-dropout draws,
    # in physical units, whose mean matches the aggregate X_filled prediction
    X, coords = _dummy_data()
    imputer = MaSTNetImputer(**_FAST)
    imputer.fit(X, coords)
    X_filled, aleatoric_std, epistemic_std, realisations = imputer.predict_with_uncertainty(
        X, coords, return_realisations=True
    )
    assert realisations.shape == (imputer.n_inference, *X.shape)
    assert not np.isnan(realisations).any()

    missing = np.isnan(X)
    np.testing.assert_allclose(
        realisations.mean(0)[missing], X_filled[missing], rtol=1e-4, atol=1e-4
    )

    # Default call keeps the original 3-tuple return signature
    result = imputer.predict_with_uncertainty(X, coords)
    assert len(result) == 3


def test_predict_with_uncertainty_physical_units(monkeypatch):
    # aleatoric_std/epistemic_std must be in physical units (scaled by each
    # feature's data_range_), not MinMax-scaled [0, 1] units. As MC-dropout is
    # stochastic, the scaled output is intercepted via monkeypatch
    # rather than recomputed in a second, random call.
    from mastnet import imputer as imputer_module

    # Set up dummy data
    X, coords = _dummy_data()

    # Fit imputer
    imputer = MaSTNetImputer(**_FAST)
    imputer.fit(X, coords)

    # Intercept the single stochastic reconstruction call to capture its
    # scaled output, instead of triggering a second random draw
    raw = {}
    original = imputer_module.reconstruct_with_uncertainty

    def _capture(*args, **kwargs):
        result = original(*args, **kwargs)  # unchanged, still stochastic
        raw["aleatoric"], raw["epistemic"] = result[1].numpy(), result[2].numpy()  # side-channel copy
        return result

    monkeypatch.setattr(imputer_module, "reconstruct_with_uncertainty", _capture)
    _, aleatoric_std, epistemic_std = imputer.predict_with_uncertainty(X, coords)

    # Check if unscaling worked correctly
    ranges = np.array([imputer.scaler_dict_[name].data_range_[0] for name in imputer.feature_names_])
    np.testing.assert_allclose(aleatoric_std, raw["aleatoric"] * ranges, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(epistemic_std, raw["epistemic"] * ranges, rtol=1e-5, atol=1e-6)


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


def test_val_method_block_healpix():
    # Equal-area HEALPix block grid must work as alternative to the default degree grid
    X, coords = _dummy_data()
    X_filled = MaSTNetImputer(**_FAST, val_method="block", block_grid="healpix").fit_transform(X, coords)
    assert X_filled.shape == X.shape
    assert not np.isnan(X_filled).any()


def test_val_method_block_healpix_explicit_nside():
    # healpix_nside overrides the block_size_deg-derived default
    X, coords = _dummy_data()
    imputer = MaSTNetImputer(**_FAST, val_method="block", block_grid="healpix", healpix_nside=4)
    train_idx, val_idx = imputer._make_train_val_split(coords)
    assert len(train_idx) + len(val_idx) == len(X)
    assert len(np.intersect1d(train_idx, val_idx)) == 0


def test_val_method_block_unknown_grid_raises():
    # An error is raised when an unknown block_grid method is passed
    X, coords = _dummy_data()
    imputer = MaSTNetImputer(**_FAST, val_method="block", block_grid="hexbin")
    with pytest.raises(ValueError, match="Unknown block_grid"):
        imputer._make_train_val_split(coords)


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


def test_fit_restores_true_best_not_last_snapshot(monkeypatch):
    # Regression test: the best-weights snapshot must only be taken when
    # EarlyStopping itself counts the epoch as a new best. Previously, the
    # snapshot used a looser condition (no delta) than EarlyStopping, so a
    # run of epochs that each nudge below the stale best_loss without ever
    # clearing the delta could overwrite an earlier, better
    # snapshot with a later, worse one.
    from mastnet import _training

    # Example setup: Epoch 0 (val=1.0) is the only real best (clears the delta
    # against the initial np.inf). Epochs 1 and 2 both beat the stale best_loss=1.0
    # without clearing its delta margin, so patience=2 exhausts after epoch 2
    val_losses = iter([1.0, 0.99995, 0.99998])
    epoch_counter = {"i": -1}

    # Use cheap, dummy train/eval functions
    def fake_train_epoch(model, *a, **k):
        epoch_counter["i"] += 1
        with torch.no_grad():
            model.weight.fill_(float(epoch_counter["i"]))  # tag weights with epoch id
        return 0.1

    def fake_eval_epoch(*a, **k):
        return next(val_losses)

    monkeypatch.setattr(_training, "_train_epoch", fake_train_epoch)
    monkeypatch.setattr(_training, "_eval_epoch", fake_eval_epoch)

    # Fit a dummy model
    model = torch.nn.Linear(1, 1)  # dummy model
    _training.fit(
        model, train_loader=[], val_loader=[],
        max_epochs=10, lr=1e-3, mask_ratio=0.5, patience=2,
        device=torch.device("cpu"), verbose=False,
    )

    # Restored weights must be epoch 0's (the true best), not epoch 2's (the
    # last snapshot taken under the old, buggy condition)
    assert model.weight.item() == 0.0


def test_fit_reseeds_validation_generator_every_epoch(monkeypatch):
    # fit() must reset the validation mask generator to the same
    # state before every epoch's _eval_epoch call
    from mastnet import _training

    seen_states = []

    # Set up dummy train/eval computations
    def fake_train_epoch(model, *a, **k):
        torch.rand(1)  # vary global-RNG consumption, as a real epoch would
        return 0.1

    def fake_eval_epoch(model, loader, loss_fn, mask_ratio, device, **kwargs):
        gen = kwargs.get("generator")
        seen_states.append(gen.get_state().clone() if gen is not None else None)
        return 0.5

    monkeypatch.setattr(_training, "_train_epoch", fake_train_epoch)
    monkeypatch.setattr(_training, "_eval_epoch", fake_eval_epoch)

    # Fit a dummy model
    model = torch.nn.Linear(1, 1)
    _training.fit(
        model, train_loader=[], val_loader=[],
        max_epochs=4, lr=1e-3, mask_ratio=0.5, patience=10,
        device=torch.device("cpu"), verbose=False, seed=123,
    )

    # Ensure same generator states for all eval epochs
    assert len(seen_states) == 4
    assert all(s is not None for s in seen_states)
    assert all(torch.equal(seen_states[0], s) for s in seen_states[1:])


def test_eval_epoch_validation_masks_reproducible_across_epochs():
    # With a generator (reseeded to the same value), _eval_epoch must
    # draw identical masks and thus losses
    from mastnet import _training

    device = torch.device("cpu")

    # Dummy data
    rng = np.random.default_rng(0)
    batch = {
        "query_features": torch.as_tensor(rng.standard_normal((16, 4)), dtype=torch.float32),
        "query_mask": torch.ones(16, 4, dtype=torch.bool),
        "query_coords": torch.as_tensor(rng.standard_normal((16, 4)), dtype=torch.float32),
    }
    loader = [batch]

    # Set up a dummy model
    class DummyModel(torch.nn.Module):
        def forward(self, batch):
            B, F = batch["query_features"].shape
            return torch.zeros(B, F), torch.zeros(B, F)

    model = DummyModel()
    loss_fn = _training._build_loss("mse")

    losses = []
    for i in range(3):
        torch.rand(3 + i * 17)  # vary global RNG between "epochs"

        # Set up validation mask generator
        val_mask_generator = torch.Generator(device=device)
        val_mask_generator.manual_seed(7)

        # Compute loss during evaluation
        losses.append(_training._eval_epoch(
            model, loader, loss_fn, mask_ratio=0.5, device=device,
            cylindrical_mask_size=0.5, cylindrical_mask_prob=0.5,
            verbose=False, generator=val_mask_generator,
        ))

    # Ensure all losses are the same, i.e. eval masks were the same
    assert losses[0] == losses[1] == losses[2]


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


def test_knn_indices_nearest_neighbour_in_column_zero():
    # Column 0 of _knn_indices must be each point's true nearest neighbour,
    # not its second-nearest (regression test for the off-by-one self-drop)
    from mastnet._graph import _knn_indices

    rng = np.random.default_rng(0)
    coords = rng.uniform(-10, 10, size=(50, 3))
    k = 5
    idx = _knn_indices(coords, k).numpy()
    assert idx.shape == (50, k)

    for i in range(len(coords)):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        dists[i] = np.inf  # exclude self
        true_nearest = dists.argmin()
        assert idx[i, 0] == true_nearest


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


def _brute_force_1d(values: np.ndarray, k: int, period: float | None = None) -> np.ndarray:
    # Computes distances brute force
    n = len(values)
    idx = np.empty((n, k), dtype=np.int64)

    # Iterate over data points
    for i in range(n):
        diff = np.abs(values - values[i])  # Compute distance
        if period is not None:
            # Edge cases for cyclic time
            diff = np.minimum(diff, period - diff)
        diff[i] = np.inf  # exclude self
        idx[i] = np.argsort(diff, kind="stable")[:k]
    return idx


def test_sorted_knn_1d_matches_brute_force_linear():
    # Tests if 1d nearest neighbours matches brute force computations
    from mastnet._graph import _sorted_knn_1d

    # Generate some data
    rng = np.random.default_rng(0)
    values = rng.uniform(-5, 5, size=200)

    # Compute nearest neighbours with both methods
    k = 7
    idx = _sorted_knn_1d(values, k)
    expected = _brute_force_1d(values, k)

    # Distances (not raw indices) must match - ties can pick either
    # equidistant neighbour
    for i in range(len(values)):
        got_dists = np.sort(np.abs(values[idx[i]] - values[i]))
        exp_dists = np.sort(np.abs(values[expected[i]] - values[i]))
        assert np.allclose(got_dists, exp_dists)


def test_sorted_knn_1d_matches_brute_force_with_heavy_duplicates():
    # Tests if 1d nearest neighbours matches brute force computations
    # when many duplicates are present
    from mastnet._graph import _sorted_knn_1d

    # Set up data
    rng = np.random.default_rng(1)
    values = rng.integers(2000, 2020, size=500).astype(float)  # 20 distinct years

    # Compute nearest neighbours with both methods
    k = 10
    idx = _sorted_knn_1d(values, k)
    expected = _brute_force_1d(values, k)

    # Check if distances match
    for i in range(len(values)):
        got_dists = np.sort(np.abs(values[idx[i]] - values[i]))
        exp_dists = np.sort(np.abs(values[expected[i]] - values[i]))
        assert np.allclose(got_dists, exp_dists)


def test_sorted_knn_1d_wraps_around_period():
    # Test circular time (does wrap edge cases around work?)
    from mastnet._graph import _sorted_knn_1d

    # Set up data
    period = 2 * np.pi
    values = np.linspace(0, period, 100, endpoint=False)

    # Compute nearest neighbours with both methods
    k = 5
    idx = _sorted_knn_1d(values, k, period=period)
    expected = _brute_force_1d(values, k, period=period)

    # Check if distances match
    for i in range(len(values)):
        got_dists = np.sort(np.minimum(np.abs(values[idx[i]] - values[i]),
                                       period - np.abs(values[idx[i]] - values[i])))
        exp_dists = np.sort(np.minimum(np.abs(values[expected[i]] - values[i]),
                                       period - np.abs(values[expected[i]] - values[i])))
        assert np.allclose(got_dists, exp_dists)

    # Point 0 (angle 0) must see the points just below the wrap (angle ~2*pi)
    assert (len(values) - 1) in idx[0]


def test_time_knn_indices_cyclic_matches_linear_recovery():
    # Test circular time (is angle decoding working correctly?)
    from mastnet._graph import _time_knn_indices, _sorted_knn_1d

    # Set up data
    rng = np.random.default_rng(2)
    months = rng.integers(1, 13, size=300).astype(float)
    angle = 2 * np.pi * months / 12
    time_coords = np.column_stack([(np.cos(angle) + 1) / 2, (np.sin(angle) + 1) / 2])

    # Compute nearest neighbours with both methods
    k = 6
    idx = _time_knn_indices(time_coords, k).numpy()
    expected = _sorted_knn_1d(angle, k, period=2 * np.pi)

    # Check if distances match
    for i in range(len(months)):
        got_dists = np.sort(np.abs(angle[idx[i]] - angle[i]) % (2 * np.pi))
        exp_dists = np.sort(np.abs(angle[expected[i]] - angle[i]) % (2 * np.pi))
        assert np.allclose(got_dists, exp_dists, atol=1e-5)
