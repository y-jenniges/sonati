"""Functional tests for SONATIImputer."""

import logging
import numpy as np
import pytest
import torch

from sonati import CopyMask, CylindricalMask, SONATIImputer, PerFeatureMask, RandomMask, precompute_graph


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


# Validation indices (take every fifth of n=300 samples)
_VAL_IDX = np.arange(0, 300, 5)

# Minimal hyperparameters for fast tests, verbose=False keeps test output clean
_FAST = dict(
    max_epochs=2, n_neighbours_space=10, n_neighbours_time=10,
    patience=2, d_model=16, dim_feedforward=32, n_inference=2,
    verbose=False, val_indices=_VAL_IDX,
)


def test_import():
    # Check import
    import sonati
    assert hasattr(sonati, "__version__")
    assert sonati.SONATIImputer is SONATIImputer


def test_fit_transform():
    # Check fit_transform and output shapes and if nans were imputed
    X, coords = _dummy_data()
    X_filled = SONATIImputer(**_FAST).fit_transform(X, coords)
    assert X_filled.shape == X.shape
    assert not np.isnan(X_filled).any()


def test_fit_then_transform():
    # Check fit, then transform
    X, coords = _dummy_data()
    imputer = SONATIImputer(**_FAST)
    imputer.fit(X, coords)
    X_filled = imputer.transform(X, coords)
    assert X_filled.shape == X.shape
    assert not np.isnan(X_filled).any()


def test_observed_values_unchanged():
    # Check if observed values remain unchanged
    X, coords = _dummy_data()
    imputer = SONATIImputer(**_FAST)
    X_filled = imputer.fit_transform(X, coords)
    observed = ~np.isnan(X)
    np.testing.assert_array_equal(X_filled[observed], X[observed])


def test_predict_with_uncertainty():
    # Check if prediction with and without uncertainty work as intended
    X, coords = _dummy_data()
    imputer = SONATIImputer(**_FAST)
    imputer.fit(X, coords)
    X_filled, aleatoric_std, epistemic_std = imputer.predict_with_uncertainty(X, coords)
    assert X_filled.shape == X.shape
    assert aleatoric_std.shape == X.shape
    assert epistemic_std.shape == X.shape
    assert not np.isnan(X_filled).any()

    # With n_inference=1, epistemic std must be zero (single deterministic pass)
    imputer_det = SONATIImputer(**{**_FAST, "n_inference": 1})
    imputer_det.fit(X, coords)
    _, _, epistemic_det = imputer_det.predict_with_uncertainty(X, coords)
    assert (epistemic_det == 0).all()


def test_predict_with_uncertainty_mse_warns_and_returns_nan_aleatoric():
    # The variance head is untrained with loss='mse', so aleatoric std should be NaN
    X, coords = _dummy_data()
    imputer = SONATIImputer(**{**_FAST, "loss_name": "mse"})
    imputer.fit(X, coords)

    # Warning is generated
    with pytest.warns(UserWarning, match="variance head"):
        X_filled, aleatoric_std, epistemic_std = imputer.predict_with_uncertainty(X, coords)

    # Aleatoric uncertainty is NaN, but epistemic is still reported and imputation is done
    assert np.isnan(aleatoric_std).all()
    assert np.isfinite(epistemic_std).all()
    assert not np.isnan(X_filled).any()


def test_beta_nll_fit_and_uncertainty():
    # beta-NLL trains the variance head, so aleatoric std is finite
    X, coords = _dummy_data()
    imputer = SONATIImputer(**{**_FAST, "loss_name": "beta_nll", "loss_kwargs": {"beta": 0.5}})
    imputer.fit(X, coords)
    X_filled, aleatoric_std, _ = imputer.predict_with_uncertainty(X, coords)
    assert not np.isnan(X_filled).any()
    assert np.isfinite(aleatoric_std).all()


def test_beta_nll_beta_zero_matches_hetero():
    # beta=0 must reduce to the plain heteroscedastic NLL
    from sonati._training import _beta_nll, _heteroscedastic_nll
    g = torch.Generator().manual_seed(0)
    mean, logvar, target = (torch.randn(20, 3, generator=g) for _ in range(3))
    mask = torch.rand(20, 3, generator=g) > 0.3
    torch.testing.assert_close(_beta_nll(mean, logvar, target, mask, var_floor=1e-4, beta=0.0),
                               _heteroscedastic_nll(mean, logvar, target, mask, var_floor=1e-4))


def test_loss_kwargs_validation():
    # Unknown kwargs and out-of-range beta are rejected
    from sonati._training import _build_loss
    with pytest.raises(TypeError):
        _build_loss("hetero", {"beta": 0.5})
    with pytest.raises(TypeError):
        _build_loss("beta_nll", {"bta": 0.5})
    with pytest.raises(ValueError):
        _build_loss("beta_nll", {"beta": 1.5})


def test_transform_too_few_rows_raises():
    # Raise error if fewer than k+1 new rows are passed for inference - they cannot form a neighbourhood
    X, coords = _dummy_data()
    imputer = SONATIImputer(**_FAST)
    imputer.fit(X, coords)

    with pytest.raises(ValueError, match="more than"):
        # Transform raises
        imputer.transform(X[:10], coords[:10])

    with pytest.raises(ValueError, match="more than"):
        # Predict with uncertainty raises
        imputer.predict_with_uncertainty(X[:10], coords[:10])


def test_predict_with_uncertainty_realisations():
    # return_realisations=True must expose the individual MC-dropout draws,
    # in physical units, whose mean matches the aggregate X_filled prediction
    X, coords = _dummy_data()
    imputer = SONATIImputer(**_FAST)
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
    from sonati import imputer as imputer_module

    # Set up dummy data
    X, coords = _dummy_data()

    # Fit imputer
    imputer = SONATIImputer(**_FAST)
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
    X_filled = SONATIImputer(**_FAST, cyclic_time=True).fit_transform(X, coords)
    assert not np.isnan(X_filled).any()


def test_val_indices_required():
    # Raise error if no validation indices are passed
    with pytest.raises(TypeError):
        SONATIImputer(**{k: v for k, v in _FAST.items() if k != "val_indices"})


def test_val_indices_determine_the_split():
    X, coords = _dummy_data()
    imputer = SONATIImputer(**_FAST)

    train_idx, split_val = imputer._make_train_val_split(coords)
    assert np.array_equal(np.sort(split_val), np.sort(_VAL_IDX))
    assert len(np.intersect1d(train_idx, split_val)) == 0
    assert len(train_idx) + len(split_val) == len(X)

    X_filled = imputer.fit_transform(X, coords)
    assert X_filled.shape == X.shape
    assert not np.isnan(X_filled).any()
    assert np.array_equal(np.sort(imputer.val_idx_), np.sort(_VAL_IDX))


def test_test_idx_masks_values_before_fit():
    # test_idx rows must have every feature set to nan before fit()
    X, coords = _dummy_data()
    test_idx = np.arange(1, 300, 7)
    X[test_idx, 0] = 1.0  # ensure these rows start with a real, non-NaN value

    imputer = SONATIImputer(**{**_FAST, "test_idx": test_idx})
    imputer.fit(X, coords)

    assert torch.isnan(imputer.values_full_[test_idx]).all()
    # fit() must not mutate the caller's own X in place
    assert not np.isnan(X[test_idx, 0]).any()


def test_test_idx_noop_when_none():
    # test_idx=None (the default) must not change behaviour
    X, coords = _dummy_data()
    imputer = SONATIImputer(**_FAST)
    assert imputer.test_idx is None
    X_filled = imputer.fit_transform(X, coords)
    assert not np.isnan(X_filled).any()


def test_history_():
    # Check history_ returned by fit()
    X, coords = _dummy_data()
    imputer = SONATIImputer(**_FAST)
    imputer.fit(X, coords)

    assert set(imputer.history_.keys()) == {"train", "val"}
    n_epochs_train = len(imputer.history_["train"])
    n_epochs_val = len(imputer.history_["val"])
    assert n_epochs_train == n_epochs_val
    assert 0 < n_epochs_train <= _FAST["max_epochs"]
    # Keyed by epoch, values are finite losses
    assert set(imputer.history_["train"].keys()) == set(range(n_epochs_train))
    assert all(np.isfinite(v) for v in imputer.history_["train"].values())
    assert all(np.isfinite(v) for v in imputer.history_["val"].values())


def test_checkpoint_resume(tmp_path):
    # Interrupted run (2 of 4 epochs) resumes to 4 epochs
    X, coords = _dummy_data()
    checkpoint = str(tmp_path / "checkpoint.pt")
    params = {**_FAST, "patience": 10, "checkpoint_path": checkpoint}

    # Train for 2 epochs
    first = SONATIImputer(**{**params, "max_epochs": 2}).fit(X, coords)
    assert len(first.history_["train"]) == 2

    # Resume training (continue until epoch 4)
    resumed = SONATIImputer(**{**params, "max_epochs": 4}).fit(X, coords)
    assert len(resumed.history_["train"]) == 4
    assert resumed.history_["train"][0] == first.history_["train"][0]  # history carried over

    # No further training, if max_epochs (4) is reached
    again = SONATIImputer(**{**params, "max_epochs": 4}).fit(X, coords)
    assert again.history_ == resumed.history_


def test_fill_strategy_mean():
    # Test mean filling
    X, coords = _dummy_data()
    X_filled = SONATIImputer(**_FAST, fill_strategy="mean").fit_transform(X, coords)
    assert X_filled.shape == X.shape
    assert not np.isnan(X_filled).any()


def test_copy_mask_standalone():
    # Test copy masking
    X, coords = _dummy_data()
    X_filled = SONATIImputer(**_FAST, masking=[CopyMask(prob=1.0)]).fit_transform(X, coords)
    assert X_filled.shape == X.shape
    assert not np.isnan(X_filled).any()


def test_copy_mask_combined_with_cylindrical_mask():
    # Test combined copy and clindtical mask
    X, coords = _dummy_data()
    X_filled = SONATIImputer(
        **_FAST, masking=[CopyMask(prob=0.5), CylindricalMask(prob=0.5)]
    ).fit_transform(X, coords)
    assert X_filled.shape == X.shape
    assert not np.isnan(X_filled).any()


def test_per_feature_mask_standalone():
    # Test per-feature (column-wise) masking
    X, coords = _dummy_data()
    X_filled = SONATIImputer(**_FAST, masking=[PerFeatureMask(ratio=0.5)]).fit_transform(X, coords)
    assert X_filled.shape == X.shape
    assert not np.isnan(X_filled).any()


def test_masking_none_resolves_to_default_specs(monkeypatch):
    # masking=None (the default) must resolve to the package's fixed
    # defaults, and must be stored unmodified on the instance itself
    from sonati import imputer as imputer_module

    imputer = SONATIImputer(**_FAST)
    assert imputer.masking is None

    captured = {}

    def fake_fit(model, train_loader, val_loader, **kwargs):
        captured["masking"] = kwargs["masking"]

    monkeypatch.setattr(imputer_module, "fit", fake_fit)

    X, coords = _dummy_data()
    imputer.fit(X, coords)
    assert captured["masking"] == [RandomMask(ratio=0.5), CylindricalMask(radius=0.1, prob=1.0)]


def test_sklearn_api():
    # Check sklearn compatibility
    from sklearn.base import clone
    imputer = SONATIImputer(**_FAST)
    params = imputer.get_params()
    assert "max_epochs" in params
    assert "verbose" in params
    cloned = clone(imputer)

    # Key-by-key comparison
    cloned_params = cloned.get_params()
    assert cloned_params.keys() == params.keys()
    for key, value in params.items():
        np.testing.assert_array_equal(cloned_params[key], value)


def test_not_fitted_error():
    # Check compatibility with NotFittedError
    from sklearn.exceptions import NotFittedError
    imputer = SONATIImputer(**_FAST)
    X, coords = _dummy_data()
    with pytest.raises(NotFittedError):
        imputer.transform(X, coords)


def test_logvar_clamped():
    # An extreme logvar head must be clamped so exp(logvar) cannot overflow
    from sonati._model import LOGVAR_CLAMP, SONATI
    model = SONATI(n_features=2, coord_dim=5, d_model=16, dim_feedforward=32)
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


def _dummy_batch(B: int = 4, K: int = 5):
    # Dummy batch with space and time scopes
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
    return batch


@pytest.mark.parametrize("clamp", [2.0, 20.0])
def test_logvar_clamp_custom(clamp):
    # A non-default clamp bounds the logvar head, in both directions
    from sonati._model import SONATI
    model = SONATI(n_features=2, coord_dim=5, d_model=16, dim_feedforward=32, logvar_clamp=clamp)
    assert model.logvar_clamp == clamp
    for bias in (1e4, -1e4):
        with torch.no_grad():
            model.logvar_head.bias.fill_(bias)
        _, pred_logvar = model(_dummy_batch())
        assert pred_logvar.abs().max().item() == pytest.approx(clamp)


def test_logvar_clamp_invalid():
    # A logvar_clamp <= 0 raises
    from sonati._model import SONATI
    with pytest.raises(ValueError):
        SONATI(n_features=2, logvar_clamp=0.0)


@pytest.mark.parametrize("loss_name", ["hetero", "beta_nll"])
def test_loss_var_floor_follows_clamp(loss_name):
    # Variance floor is exp(-clamp): with the logvar at -20, a clamp of 10 floors the variance at
    # e^-10, whereas a clamp of 20 leaves it at e^-20
    from sonati._training import _build_loss
    pred_mean = torch.zeros(1, 1)
    target = torch.ones(1, 1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    logvar = torch.full((1, 1), -20.0)
    # Evaluate loss with different logvar clamps
    loss_10 = _build_loss(loss_name, logvar_clamp=10.0)(pred_mean, logvar, target, mask)
    loss_20 = _build_loss(loss_name, logvar_clamp=20.0)(pred_mean, logvar, target, mask)
    # Check if the clamp influenced the loss
    assert not torch.isclose(loss_10, loss_20)


def test_logvar_clamp_imputer():
    # Imputer passes the clamp to the model and supports sklearn clone/get_params
    from sklearn.base import clone
    X, coords = _dummy_data()
    imputer = SONATIImputer(**_FAST, logvar_clamp=0.01)
    assert imputer.get_params()["logvar_clamp"] == 0.01  # logvar_clamp is stored correctly
    assert clone(imputer).logvar_clamp == 0.01  # logvar_clamp is passed on clone
    assert SONATIImputer(**_FAST).logvar_clamp == 10.0  # default is 10

    imputer.fit(X, coords)
    assert imputer.model_.logvar_clamp == 0.01  # clamp remains after fit


@pytest.mark.parametrize("num_workers, inference_num_workers, expected", [
    (3, None, 3),  # None means "same as num_workers"
    (3, 0, 0),  # an explicit value (also 0) wins
    (3, 2, 2),
])
def test_inference_num_workers_defaults_to_num_workers(monkeypatch, num_workers, inference_num_workers, expected):
    imputer = SONATIImputer(**_FAST, num_workers=num_workers, inference_num_workers=inference_num_workers)
    assert imputer.get_params()["inference_num_workers"] == inference_num_workers  # param stays unchanged

    # Data that differs from the training data takes the non-reuse branch and builds a new loader
    imputer.coords_full_ = torch.zeros(1, 1)
    captured = {}
    monkeypatch.setattr(imputer, "_make_loader", lambda *a, **k: captured.update(k))
    imputer._inference_loader(torch.ones(2, 2), torch.ones(2, 2), graph={})
    assert captured["num_workers"] == expected
    assert captured["multiprocessing_context"] == "spawn"


def _fit_with_stub_epochs(monkeypatch, loss_name, **kwargs):
    # Helper:  It replaces _train_epoch and _eval_epoch with dummies, so fit() runs one "epoch" on a dummy Linear model
    from sonati import _training

    captured = {}

    # Fake _eval_epoch
    def fake_eval(model, loader, loss_fn, *args, **kw):
        captured["loss_fn"] = loss_fn
        return 0.0

    monkeypatch.setattr(_training, "_train_epoch", lambda *a, **k: 0.0)  # fake _train_epoch
    monkeypatch.setattr(_training, "_eval_epoch", fake_eval)
    model = torch.nn.Linear(1, 1)  # dummy linear model
    model.logvar_clamp = 10.0
    _training.fit(model, train_loader=[], val_loader=[], max_epochs=1, lr=1e-3, masking=[RandomMask()], patience=1,
                  device=torch.device("cpu"), verbose=False, loss_name=loss_name, **kwargs)
    return captured.get("loss_fn")  # None when fit() resumed a finished checkpoint and ran no epoch


def test_beta_nll_validates_on_plain_nll(monkeypatch):
    # beta-NLL loss uses plain NLL loss for validation/early stopping
    from sonati._training import _build_loss

    val_loss_fn = _fit_with_stub_epochs(monkeypatch, "beta_nll")

    # Dummy predictions with a clear error (0 vs 0.5), so that NLL and beta-NLL give different numbers
    mean, logvar, target = torch.zeros(4, 2), torch.full((4, 2), -2.0), torch.full((4, 2), 0.5)
    mask = torch.ones(4, 2, dtype=torch.bool)  # all entries observed
    nll = _build_loss("hetero")(mean, logvar, target, mask)  # reference: plain heteroscedastic NLL
    assert torch.isclose(val_loss_fn(mean, logvar, target, mask), nll)  # validation = plain NLL
    assert not torch.isclose(_build_loss("beta_nll")(mean, logvar, target, mask), nll)  # training loss differs


def test_hetero_validates_on_its_own_loss(monkeypatch):
    # Plain 'hetero' loss uses itself for validation/early stopping
    from sonati._training import _build_loss

    val_loss_fn = _fit_with_stub_epochs(monkeypatch, "hetero")
    mean, logvar, target = torch.zeros(4, 2), torch.full((4, 2), -2.0), torch.full((4, 2), 0.5)
    mask = torch.ones(4, 2, dtype=torch.bool)
    assert torch.isclose(val_loss_fn(mean, logvar, target, mask), _build_loss("hetero")(mean, logvar, target, mask))


def test_fit_aborts_on_nonfinite_loss(monkeypatch, caplog):
    # A non-finite epoch loss must abort training instead of burning patience
    from sonati import _training

    monkeypatch.setattr(_training, "_train_epoch", lambda *a, **k: float("nan"))
    monkeypatch.setattr(_training, "_eval_epoch", lambda *a, **k: float("nan"))

    model = torch.nn.Linear(1, 1)

    model.logvar_clamp = 10.0  # fit() reads the clamp from the model
    with caplog.at_level(logging.INFO, logger="sonati._training"):
        _training.fit(
            model, train_loader=[], val_loader=[],
            max_epochs=10, lr=1e-3, masking=[RandomMask()], patience=10,
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
    from sonati import _training

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
    model.logvar_clamp = 10.0  # fit() reads the clamp from the model
    _training.fit(
        model, train_loader=[], val_loader=[],
        max_epochs=10, lr=1e-3, masking=[RandomMask()], patience=2,
        device=torch.device("cpu"), verbose=False,
    )

    # Restored weights must be epoch 0's (the true best), not epoch 2's (the
    # last snapshot taken under the old, buggy condition)
    assert model.weight.item() == 0.0


def test_fit_reseeds_validation_generator_every_epoch(monkeypatch):
    # fit() must reset the validation mask generator to the same
    # state before every epoch's _eval_epoch call
    from sonati import _training

    seen_states = []

    # Set up dummy train/eval computations
    def fake_train_epoch(model, *a, **k):
        torch.rand(1)  # vary global-RNG consumption, as a real epoch would
        return 0.1

    def fake_eval_epoch(model, loader, loss_fn, masking, device, **kwargs):
        gen = kwargs.get("generator")
        seen_states.append(gen.get_state().clone() if gen is not None else None)
        return 0.5

    monkeypatch.setattr(_training, "_train_epoch", fake_train_epoch)
    monkeypatch.setattr(_training, "_eval_epoch", fake_eval_epoch)

    # Fit a dummy model
    model = torch.nn.Linear(1, 1)
    model.logvar_clamp = 10.0  # fit() reads the clamp from the model
    _training.fit(
        model, train_loader=[], val_loader=[],
        max_epochs=4, lr=1e-3, masking=[RandomMask()], patience=10,
        device=torch.device("cpu"), verbose=False, seed=123,
    )

    # Ensure same generator states for all eval epochs
    assert len(seen_states) == 4
    assert all(s is not None for s in seen_states)
    assert all(torch.equal(seen_states[0], s) for s in seen_states[1:])


def test_eval_epoch_validation_masks_reproducible_across_epochs():
    # With a generator (reseeded to the same value), _eval_epoch must
    # draw identical masks and thus losses
    from sonati import _training

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
            model, loader, loss_fn, masking=[RandomMask(), CylindricalMask(radius=0.5, prob=0.5)], device=device,
            verbose=False, generator=val_mask_generator,
        ))

    # Ensure all losses are the same, i.e. eval masks were the same
    assert losses[0] == losses[1] == losses[2]


def test_copy_mask_strategy():
    # CopyMask: prob=0 never masks, prob=1 always masks
    from sonati._masking import apply_masking_strategies

    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(0)

    n_points, n_features = 8, 3
    # One fully-missing donor row, one fully-observed donor row
    donor_pool_mask = torch.tensor([[False, False, False], [True, True, True]])

    mask_off = apply_masking_strategies(
        [CopyMask(prob=0.0)], n_points, n_features,
        donor_pool_mask=donor_pool_mask, device=device, generator=generator,
    )
    assert not mask_off.any()

    mask_on = apply_masking_strategies(
        [CopyMask(prob=1.0)], n_points, n_features,
        donor_pool_mask=donor_pool_mask, device=device, generator=generator,
    )
    # Every point's mask must come entirely from one donor, i.e. all-hidden or all-observed
    row_sums = mask_on.sum(dim=1)
    assert ((row_sums == 0) | (row_sums == n_features)).all()
    assert row_sums.sum() > 0  # at least one point drew the fully-missing donor


def test_copy_mask_requires_donor_pool():
    # CopyMask: Raises an error when donor pool not specified
    from sonati._masking import apply_masking_strategies

    device = torch.device("cpu")
    with pytest.raises(ValueError, match="donor_pool_mask"):
        apply_masking_strategies([CopyMask(prob=0.5)], 4, 3, device=device)


def test_unknown_masking_strategy_raises():
    # Unspecified masking strategies raise an error
    from sonati._masking import apply_masking_strategies

    with pytest.raises(ValueError, match="Unknown masking strategy"):
        apply_masking_strategies(["not_a_spec"], 4, 3, device=torch.device("cpu"))


def test_masking_strategies_combine(monkeypatch):
    # Multiple strategies must OR into the same hide-mask
    from sonati import _masking

    device = torch.device("cpu")
    B, n_features = 4, 3
    batch = {
        "query_mask": torch.ones(B, n_features, dtype=torch.bool),
        "query_coords": torch.zeros(B, 4),
    }

    # Dummy cylinder and copy masks
    cyl_fake = torch.tensor([[True, False, False]] * B)
    copy_fake = torch.tensor([[False, True, False]] * B)
    monkeypatch.setattr(CylindricalMask, "mask", lambda self, *a, **k: cyl_fake)
    monkeypatch.setattr(CopyMask, "mask", lambda self, *a, **k: copy_fake)

    # Dummy donor pool
    donor_pool_mask = torch.ones(2, n_features, dtype=torch.bool)

    # Apply masks
    _, loss_mask = _masking.mask_batch(
        batch, [CylindricalMask(prob=1.0), CopyMask(prob=1.0)], device,
        donor_pool_mask=donor_pool_mask,
    )

    # Both mechanisms' hidden positions must survive into the final loss mask
    assert (loss_mask & cyl_fake).equal(cyl_fake)
    assert (loss_mask & copy_fake).equal(copy_fake)
    # No RandomMask in the list, so nothing beyond these two mechanisms
    assert loss_mask.equal(cyl_fake | copy_fake)


def test_empty_masking_hides_nothing():
    # An empty masking list must hide nothing of query or neighbours
    from sonati._masking import mask_batch

    device = torch.device("cpu")
    B, K, n_features = 4, 5, 3
    batch = {
        "query_mask": torch.ones(B, n_features, dtype=torch.bool),
        "query_coords": torch.zeros(B, 4),
        "space": {
            "mask": torch.ones(B, K, n_features, dtype=torch.bool),
            "features": torch.zeros(B, K, n_features),
            "coords": torch.zeros(B, K, 4),
        },
    }

    new_batch, loss_mask = mask_batch(batch, [], device)

    assert not loss_mask.any()
    assert new_batch["query_mask"].all()  # nothing hidden from the model input
    assert new_batch["space"]["mask"].all()  # neighbour features untouched


def test_random_mask_ratio_zero_disables():
    # RandomMask: Disabled, when prob=0.0
    from sonati._masking import apply_masking_strategies

    device = torch.device("cpu")
    mask = apply_masking_strategies([RandomMask(ratio=0.0)], n_points=10, n_features=4, device=device)
    assert not mask.any()


def test_random_mask_is_row_wise():
    # RandomMask hides a fraction of features per point (row-wise)
    from sonati._masking import apply_masking_strategies

    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(0)
    n_points, n_features = 10, 6

    mask = apply_masking_strategies(
        [RandomMask(ratio=0.5)], n_points, n_features, device=device, generator=generator,
    )
    assert (mask.sum(dim=1) == n_features // 2).all()


def test_per_feature_mask_ratio_zero_disables():
    # PerFeatureMask: Disabled, when prob=0.0
    from sonati._masking import apply_masking_strategies

    device = torch.device("cpu")
    mask = apply_masking_strategies([PerFeatureMask(ratio=0.0)], n_points=10, n_features=4, device=device)
    assert not mask.any()


def test_per_feature_mask_is_column_wise():
    # PerFeatureMask hides a fraction of points per feature (column-wise)
    from sonati._masking import apply_masking_strategies

    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(0)
    n_points, n_features = 20, 4

    mask = apply_masking_strategies(
        [PerFeatureMask(ratio=0.5)], n_points, n_features, device=device, generator=generator,
    )
    assert (mask.sum(dim=0) == n_points // 2).all()


def test_precomputed_graph():
    # A graph built once with a larger k can be sliced and reused by fit
    X, coords = _dummy_data()
    graph = precompute_graph(coords, n_neighbours_space=15, n_neighbours_time=15)
    assert graph["space"].shape == (len(X), 15)

    imputer = SONATIImputer(**_FAST)  # n_neighbours=10 < 15
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
    from sonati._graph import _knn_indices

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
    from sonati._graph import NeighbourDataset

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
    from sonati import imputer as imputer_module

    X, coords = _dummy_data()
    imputer = SONATIImputer(**_FAST)
    imputer.fit(X, coords)

    def _fail(*args, **kwargs):
        raise AssertionError("build_graph should not be called for training coords")

    monkeypatch.setattr(imputer_module, "build_graph", _fail)
    X_filled = imputer.transform(X, coords)  # same coords -> reuse, no rebuild
    assert not np.isnan(X_filled).any()

    with pytest.raises(AssertionError, match="should not be called"):
        imputer.transform(X[:100], coords[:100])  # new coords -> rebuild attempted


def test_transform_accepts_precomputed_graph(monkeypatch):
    # transform(graph=...) must take priority over training-graph and fresh re-build
    from sonati import imputer as imputer_module

    X, coords = _dummy_data()
    imputer = SONATIImputer(**_FAST)
    imputer.fit(X, coords)

    graph = precompute_graph(coords[:100], n_neighbours_space=10, n_neighbours_time=10)

    def _fail(*args, **kwargs):
        raise AssertionError("build_graph should not be called when graph= is given")

    monkeypatch.setattr(imputer_module, "build_graph", _fail)
    X_filled = imputer.transform(X[:100], coords[:100], graph=graph)  # new coords, but graph given
    assert X_filled.shape == (100, X.shape[1])
    assert not np.isnan(X_filled).any()


def test_precomputed_graph_validation():
    # Invalid precomputed graphs must raise informative errors
    X, coords = _dummy_data()
    imputer = SONATIImputer(**_FAST)

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
    from sonati._graph import _sorted_knn_1d

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
    from sonati._graph import _sorted_knn_1d

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
    from sonati._graph import _sorted_knn_1d

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
    from sonati._graph import _time_knn_indices, _sorted_knn_1d

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
