"""SONATIImputer - Spatio-temporal Ocean Neighbourhood Asymmetric Transformer for Imputation,
a sklearn-compatible spatio-temporal imputer.
"""

import logging
import multiprocessing as mp
import warnings
from time import time
from typing import Dict
import numpy as np
import torch
from torch import Tensor
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# forkserver not available on Windows -> fall back to slower spawn method
_FIT_MP_CONTEXT = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"

from sonati._graph import NeighbourDataset, build_graph, precompute_graph
from sonati._masking import CylindricalMask, RandomMask, validate_masking
from sonati._model import SONATI
from sonati._preprocessing import inverse_transform, inverse_transform_std, preprocess
from sonati._training import fit, reconstruct, reconstruct_with_uncertainty


def _set_seed(seed: int) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _nan_aware_equal(a: Tensor, b: Tensor) -> bool:
    """Tensor equality treating NaN == NaN (torch.equal doesn't)."""
    return (
        a.shape == b.shape and
        torch.equal(a.isnan(), b.isnan()) and
        torch.equal(torch.nan_to_num(a), torch.nan_to_num(b)
    ))


class SONATIImputer(BaseEstimator, TransformerMixin):
    """
    SONATI (Spatio-temporal Ocean Neighbourhood Asymmetric Transformer for Imputation)
    reconstructs missing values by attending to k-nearest temporal and
    j-nearest spatial neighbours (in that order) in a factorised cross-attention
    architecture. Missingness enters the model only through a feature mixer
    applied to each token's values (mixing them with the observation mask)
    before attention. A heteroscedastic output head provides per-point
    uncertainty estimates.

    Coordinates are expected as four columns in this order:

        ``[latitude (°), longitude (°), depth (m), time]``

    Internally, latitude and longitude are mapped to unit sphere coordinates,
    depth and time are MinMax-scaled. Features are MinMax-scaled on
    observed values only.

    Parameters
    ----------
    val_indices : array-like of int
        Row indices to hold out as the internal validation set for early
        stopping.
    test_idx : array-like of int or None, default=None
        Row indices to withhold entirely from training. Coordinates for these
        rows are still used to build the k-NN graph.
    n_neighbours_space : int, default=30
        Number of nearest neighbours per query in the spatial graph.
    n_neighbours_time : int, default=30
        Number of nearest neighbours per query in the temporal graph.
    d_model : int, default=64
        Transformer embedding dimension.
    nhead : int, default=4
        Number of attention heads (must divide ``d_model``).
    n_spatial_layers : int, default=3
        Cross-attention layers over the spatial neighbourhood.
    n_temporal_layers : int, default=3
        Cross-attention layers over the temporal neighbourhood.
    dim_feedforward : int, default=128
        Feedforward sublayer width inside each attention block.
    dropout : float, default=0.1
        Dropout rate.
    fill_strategy : str, default='zero'
        How to fill unobserved positions before attention: ``'zero'`` or
        ``'mean'`` (global per-feature mean estimated from training data).
    loss_name : str, default='hetero'
        Training loss: ``'hetero'`` for heteroscedastic Gaussian NLL,
        ``'beta_nll'`` for the beta-NLL of Seitzer et al. (2022) or
        ``'mse'`` for masked MSE. With ``'beta_nll'`` as validation loss, early
        stopping and best-weights selection use the plain heteroscedastic NLL,
        (beta-NLL value does not reflect model quality).
    loss_kwargs : dict or None, default=None
        Extra options for the loss, e.g. ``'beta_nll'`` accepts
        ``{"beta": float}``.
    max_epochs : int, default=100
        Maximum number of training epochs.
    lr : float, default=1e-3
        Adam learning rate.
    batch_size : int, default=64
        Mini-batch size for training and inference.
    masking : list of RandomMask / CylindricalMask / CopyMask / PerFeatureMask or None
        Masking strategies applied during training and validation ot both query and
        neighbourhood token, e.g.::

            from sonati import SONATIImputer, RandomMask, CylindricalMask, CopyMask
            SONATIImputer(masking=[RandomMask(ratio=0.5), CopyMask(prob=0.3)])

        Each strategy is independent and additive. Defaults (``None``) to
        ``[RandomMask(ratio=0.5), CylindricalMask(radius=0.1, prob=1.0)]``.
    grad_clip_norm : float, default=1.0
        Maximum gradient norm; gradients are clipped with
        :func:`torch.nn.utils.clip_grad_norm_` before every optimiser step.
        Stabilises heteroscedastic training, which can otherwise diverge to
        NaN. Set to ``0`` to disable.
    patience : int, default=10
        Early-stopping patience (epochs without validation improvement).
    checkpoint_path : str or None, default=None
        If given, the training state is stored there so that training can be
        interrupted and resumed.
    logvar_clamp : float, default=10.0
        The predicted log-variance is clamped to ``[-logvar_clamp, logvar_clamp]``
        and the variance floor of the NLL losses is set to ``exp(-logvar_clamp)``.
    cyclic_time : bool, default=False
        Encode time cyclically with sine/cosine. Useful for monthly data
        (pass month 1–12 as the time column); use False for yearly data.
    device : str, default='auto'
        Compute device: ``'cpu'``, ``'cuda'``, or ``'auto'`` (CUDA when
        available).
    seed : int, default=42
        Random seed for reproducibility.
    feature_names : list of str or None, default=None
        Optional names for the feature columns supplied by the caller.
        Inferred automatically from a :class:`pandas.DataFrame` input.
        This is the *input* parameter; the resolved list is stored as the
        fitted attribute ``feature_names_`` after :meth:`fit`.
    n_inference : int, default=50
        Number of stochastic forward passes used by
        :meth:`predict_with_uncertainty` to estimate epistemic uncertainty
        via MC Dropout. Must be ``> 1`` to get a meaningful epistemic
        estimate; ``1`` produces a deterministic prediction.
    verbose : bool, default=True
        Show progress bars and per-epoch log lines during training and
        inference. Set to ``False`` to silence all console output (e.g.
        inside a grid search or cross-validation loop).
    n_jobs : int or None, default=None
        Parallel jobs for the spatial k-NN graph build (sklearn convention:
        ``None`` means 1, ``-1`` means all cores). No effect on a
        precomputed graph passed via ``fit(..., graph=...)``, nor on the
        single-threaded temporal k-NN search.
    num_workers : int, default=0
        Worker processes for the ``fit()`` train/val/full DataLoaders.
        ``0`` loads single-process (main bottleneck at large ``n``). Only
        applies inside :meth:`fit`, which warms these loaders up before
        moving the model to ``device`` to avoid forking after CUDA init.
        :meth:`transform` / :meth:`predict_with_uncertainty` reuse that
        warmed-up loader when called on the same data used to :meth:`fit`.
        For other data they use ``inference_num_workers`` (which defaults to
        ``num_workers``).
    pin_memory : bool, default=False
        Passed to every internal ``DataLoader``. Speeds up host-to-device
        transfer on GPU. No effect on CPU.
    inference_num_workers : int or None, default=None
        Worker processes for :meth:`transform` / :meth:`predict_with_uncertainty`
        when called on data that does not match the training data (so
        ``full_loader_`` can't be reused). ``None`` uses ``num_workers``.

    Attributes
    ----------
    model_ : SONATI
        Trained PyTorch model.
    scaler_dict_ : dict
        Fitted coordinate and feature scalers.
    feature_names_ : list of str
        Resolved feature names set during :meth:`fit`. Always a plain list
        of strings regardless of how ``feature_names`` was provided.
    n_features_in_ : int
    history_ : dict
        ``{"train": {epoch: loss, ...}, "val": {epoch: loss, ...}}`` from the
        most recent :meth:`fit` call with one entry per epoch.
    val_idx_ : ndarray of int
        Row indices held out as the internal validation set for early
        stopping. Callers can score their own metric with this.

    Examples
    --------
    >>> import numpy as np
    >>> from sonati import SONATIImputer
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> X = rng.standard_normal((n, 6))
    >>> X[rng.random((n, 6)) < 0.3] = np.nan           # 30 % missing
    >>> coords = np.column_stack([
    ...     rng.uniform(-90, 90, n),                   # latitude
    ...     rng.uniform(-180, 180, n),                 # longitude
    ...     rng.uniform(0, 5000, n),                   # depth (m)
    ...     rng.integers(1990, 2020, n).astype(float), # year
    ... ])
    >>> val_indices = rng.choice(n, size=n // 5, replace=False)  # rows for early stopping
    >>> imputer = SONATIImputer(val_indices, max_epochs=50, n_neighbours_space=15,
    ...                         n_neighbours_time=15, verbose=False)
    >>> X_filled = imputer.fit_transform(X, coords)
    """

    def __init__(
            self,
            val_indices,
            test_idx=None,
            n_neighbours_space: int = 30,
            n_neighbours_time: int = 30,
            d_model: int = 64,
            nhead: int = 4,
            n_spatial_layers: int = 3,
            n_temporal_layers: int = 3,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
            fill_strategy: str = "zero",
            loss_name: str = "hetero",
            loss_kwargs: dict | None = None,
            max_epochs: int = 100,
            lr: float = 1e-3,
            batch_size: int = 64,
            masking: list | None = None,
            grad_clip_norm: float = 1.0,
            patience: int = 10,
            cyclic_time: bool = False,
            device: str = "auto",
            seed: int = 42,
            feature_names=None,
            n_inference: int = 50,
            verbose: bool = True,
            n_jobs: int | None = None,
            num_workers: int = 0,
            pin_memory: bool = False,
            inference_num_workers: int | None = None,
            checkpoint_path: str | Path | None = None,
            logvar_clamp: float = 10.0,
    ):
        self.n_neighbours_space = n_neighbours_space
        self.n_neighbours_time = n_neighbours_time
        self.d_model = d_model
        self.nhead = nhead
        self.n_spatial_layers = n_spatial_layers
        self.n_temporal_layers = n_temporal_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.fill_strategy = fill_strategy
        self.loss_name = loss_name
        self.loss_kwargs = loss_kwargs
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.masking = masking
        self.grad_clip_norm = grad_clip_norm
        self.patience = patience
        self.cyclic_time = cyclic_time
        self.device = device
        self.seed = seed
        self.feature_names = feature_names
        self.n_inference = n_inference
        self.val_indices = val_indices
        self.test_idx = test_idx
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.inference_num_workers = inference_num_workers
        self.checkpoint_path = checkpoint_path
        self.logvar_clamp = logvar_clamp

    def _make_train_val_split(self, coords_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (train_idx, val_idx) from ``val_indices``."""
        idx = np.arange(len(coords_raw))  # all passed coordinates
        val_idx = np.asarray(self.val_indices, dtype=int)
        # Training data is all passed data without the validation indices
        return np.setdiff1d(idx, val_idx), val_idx

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _resolve_feature_names(self, X) -> list[str]:
        if hasattr(X, "columns") and self.feature_names is None:
            return list(X.columns)
        if self.feature_names is not None:
            return list(self.feature_names)
        return [f"feature_{i}" for i in range(np.asarray(X).shape[1])]

    def _slice_graph(self, graph: dict, n: int) -> dict:
        """Validate a precomputed graph and slice it to the configured neighbour counts."""
        sliced = {}
        for scope, k in (("space", self.n_neighbours_space),
                         ("time", self.n_neighbours_time)):
            if scope not in graph:
                raise ValueError(f"Precomputed graph is missing the '{scope}' key.")
            # Preserve the incoming integer dtype (e.g. int32 from
            # precompute_graph) rather than forcing int64, which would
            # double memory back up for large precomputed graphs.
            idx = torch.as_tensor(graph[scope])
            if idx.ndim != 2 or idx.shape[0] != n:
                raise ValueError(
                    f"Precomputed '{scope}' graph has shape {tuple(idx.shape)}; "
                    f"expected ({n}, >= {k}) for this dataset."
                )
            if idx.shape[1] < k:
                raise ValueError(
                    f"Precomputed '{scope}' graph provides {idx.shape[1]} neighbours "
                    f"but n_neighbours_{scope}={k}. Rebuild it with a larger k."
                )
            sliced[scope] = idx[:, :k]
        return sliced

    def _inference_graph(self, coords_scaled: torch.Tensor, graph: dict | None = None) -> dict:
        """
        Return the k-NN graph for inference coordinates.

        Uses ``graph`` if given, else if `coords_scaled`` is identical to the
        training coordinates, re-uses graph built during :meth:`fit`, else
        builds a fresh graph.
        """
        if graph is not None:
            return self._slice_graph(graph, len(coords_scaled))

        # Skip graph building
        if torch.equal(coords_scaled, self.coords_full_):
            logger.info("Coordinates match the training data; reusing the k-NN graph from fit().")
            return self.graph_

        # Raise error if not enough points were passed at inference
        k_max = max(self.n_neighbours_space, self.n_neighbours_time)
        if len(coords_scaled) <= k_max:
            raise ValueError(
                f"Need more than max(n_neighbours_space, n_neighbours_time)={k_max} rows "
                f"to build a k-NN graph for inference, got {len(coords_scaled)}. "
            )
        logger.info(
            "Building k-NN graph (n=%d, k_space=%d, k_time=%d)...",
            len(coords_scaled), self.n_neighbours_space, self.n_neighbours_time,
        )

        # Build graph
        t0 = time()
        graph = build_graph(coords_scaled, self.n_neighbours_space, self.n_neighbours_time,
                           n_jobs=self.n_jobs)
        logger.info("Graph built in %.1fs.", time() - t0)
        return graph

    def _make_loader(self, coords, values, graph, indices=None, shuffle=False,
                     generator=None, neighbour_values=None, num_workers=0,
                     multiprocessing_context=None):
        # Create dataset
        ds = NeighbourDataset(coords, values, graph, query_indices=indices,
                              neighbour_values=neighbour_values)

        # Return data loader
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          generator=generator if shuffle else None,
                          num_workers=num_workers, pin_memory=self.pin_memory,
                          persistent_workers=num_workers > 0,
                          multiprocessing_context=multiprocessing_context if num_workers > 0 else None)

    def _inference_loader(self, coords_scaled: torch.Tensor, values_scaled: torch.Tensor,
                          graph: Dict[str, Tensor]) -> DataLoader:
        """Re-use full_loader_ (parallel, forked pre-CUDA) if coords and values match
        the training data. Otherwise, use inference_num_workers via 'spawn' (the
        model is already on device by now, so forking would be unsafe)."""
        # Check if coordinates match training data
        if (torch.equal(coords_scaled, self.coords_full_)
                and _nan_aware_equal(values_scaled, self.values_full_)):
            logger.info("Coordinates and values match the training data -> reusing the DataLoader from fit().")
            return self.full_loader_

        # Otherwise use multiprocessing_context
        num_workers = self.num_workers if self.inference_num_workers is None else self.inference_num_workers
        if num_workers > 0:
            logger.info("Loading new inference data with %d spawned DataLoader worker(s)...", num_workers)
        return self._make_loader(coords_scaled, values_scaled, graph, num_workers=num_workers, multiprocessing_context="spawn")

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, coords, y=None, graph=None):
        """
        Fit SONATI on spatio-temporal data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix; ``NaN`` marks missing values.
        coords : array-like of shape (n_samples, 4)
            Spatio-temporal coordinates as ``[lat (°), lon (°), depth (m), time]``.
        y : ignored
        graph : dict or None, default=None
            Precomputed k-NN graph with keys ``'space'`` and ``'time'``, each an
            integer array of shape ``(n_samples, k)`` as returned by
            :func:`sonati.precompute_graph`. Each scope may provide *more*
            neighbours than configured; only the first ``n_neighbours_space`` /
            ``n_neighbours_time`` columns are used. Because the graph depends
            only on the coordinates and ``k``, building it once with the largest
            ``k`` under consideration and passing it here avoids repeating the
            (potentially long) k-NN build in every hyperparameter-tuning trial.

        Returns
        -------
        self
        """
        self.feature_names_ = self._resolve_feature_names(X)
        X = np.asarray(X, dtype=float)
        coords = np.asarray(coords, dtype=float)

        # Set test indices to nan
        if self.test_idx is not None and len(self.test_idx) > 0:
            X = X.copy()
            X[self.test_idx] = np.nan

        device = self._resolve_device()
        generator = _set_seed(self.seed)

        # Train / validation split
        train_idx, val_idx = self._make_train_val_split(coords)
        self.val_idx_ = val_idx  # expose val_idx_ so callers can score their own metric

        # Preprocess: fit scalers on training rows, transform full dataset
        coords_t = torch.as_tensor(coords, dtype=torch.float32)
        values_t = torch.as_tensor(X, dtype=torch.float32)

        _, _, scaler_dict = preprocess(
            coords_t[train_idx], values_t[train_idx],
            self.feature_names_, self.cyclic_time,
        )
        coords_full, values_full, scaler_dict = preprocess(
            coords_t, values_t,
            self.feature_names_, self.cyclic_time,
            scaler_dict=scaler_dict,
        )
        mask_full = ~torch.isnan(values_full)

        # Donor population for CopyMask are all training rows
        donor_pool_mask = mask_full[train_idx].to(device)

        # Resolve masking strategies and fail on typos
        masking = self.masking if self.masking is not None else [
            RandomMask(ratio=0.5),
            CylindricalMask(radius=0.1, prob=1.0),
        ]
        validate_masking(masking)

        self.scaler_dict_ = scaler_dict
        self.coords_full_ = coords_full
        self.values_full_ = values_full

        # Global means for fill_strategy='mean'
        obs = values_full.clone()
        obs[~mask_full] = float("nan")
        global_means = torch.nan_to_num(torch.nanmean(obs, dim=0), nan=0.0).tolist()

        # Validation rows are NaN-ed in the neighbour-feature tensor so their
        # observed values stay hidden from every neighbourhood during training
        neighbour_values = values_full.clone()
        neighbour_values[val_idx] = float("nan")

        if graph is not None:
            graph = self._slice_graph(graph, len(coords_full))
            logger.info(
                "Using precomputed k-NN graph (n=%d, k_space=%d, k_time=%d).",
                len(coords_full), self.n_neighbours_space, self.n_neighbours_time,
            )
        else:
            logger.info(
                "Building k-NN graph (n=%d, k_space=%d, k_time=%d)...",
                len(coords_full), self.n_neighbours_space, self.n_neighbours_time,
            )
            t0 = time()
            graph = build_graph(coords_full, self.n_neighbours_space, self.n_neighbours_time,
                               n_jobs=self.n_jobs)
            logger.info("Graph built in %.1fs.", time() - t0)
        self.graph_ = graph

        # Loaders (train/val hide values from neighbourhoods)
        train_loader = self._make_loader(coords_full, values_full, graph,
                                         indices=train_idx, shuffle=True,
                                         generator=generator,
                                         neighbour_values=neighbour_values,
                                         num_workers=self.num_workers,
                                         multiprocessing_context=_FIT_MP_CONTEXT)
        val_loader = self._make_loader(coords_full, values_full, graph,
                                       indices=val_idx,
                                       neighbour_values=neighbour_values,
                                       num_workers=self.num_workers,
                                       multiprocessing_context=_FIT_MP_CONTEXT)
        self.full_loader_ = self._make_loader(coords_full, values_full, graph,
                                              num_workers=self.num_workers,
                                              multiprocessing_context=_FIT_MP_CONTEXT)

        # Model
        coord_dim = coords_full.shape[1]
        self.model_ = SONATI(
            n_features=X.shape[1],
            coord_dim=coord_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            n_spatial_layers=self.n_spatial_layers,
            n_temporal_layers=self.n_temporal_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            fill_strategy=self.fill_strategy,
            global_means=global_means if self.fill_strategy == "mean" else None,
            logvar_clamp=self.logvar_clamp,
        ).to(device)

        self.history_ = fit(
            self.model_, train_loader, val_loader,
            max_epochs=self.max_epochs, lr=self.lr,
            masking=masking, patience=self.patience,
            device=device, loss_name=self.loss_name,
            loss_kwargs=self.loss_kwargs,
            donor_pool_mask=donor_pool_mask,
            grad_clip_norm=self.grad_clip_norm,
            verbose=self.verbose,
            seed=self.seed,
            checkpoint_path=self.checkpoint_path,
        )

        self.device_ = device
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, coords, graph=None):
        """
        Impute missing values using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        coords : array-like of shape (n_samples, 4)
        graph : dict or None, default=None
            Precomputed k-NN graph for these coordinates.

        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
            Copy of ``X`` with ``NaN`` positions filled by model predictions.
        """
        check_is_fitted(self, ["model_", "scaler_dict_"])

        X = np.asarray(X, dtype=float)
        coords = np.asarray(coords, dtype=float)

        # Pre-process input data
        coords_scaled, values_scaled, _ = preprocess(
            torch.as_tensor(coords, dtype=torch.float32),
            torch.as_tensor(X, dtype=torch.float32),
            self.feature_names_, self.cyclic_time,
            scaler_dict=self.scaler_dict_,
        )

        # Neighbourhood graphs (reused from fit() when coords are the training coords)
        graph = self._inference_graph(coords_scaled, graph=graph)
        loader = self._inference_loader(coords_scaled, values_scaled, graph)

        # Predict and unscale
        preds_np = inverse_transform(
            reconstruct(self.model_, loader, self.device_, verbose=self.verbose).numpy(),
            self.scaler_dict_, self.feature_names_,
        )

        X_out = X.copy()
        X_out[np.isnan(X_out)] = preds_np[np.isnan(X_out)]
        return X_out

    def fit_transform(self, X, coords=None, y=None, **fit_params):
        """
        Fit and impute in a single pass, reusing the graph built during fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        coords : array-like of shape (n_samples, 4)
        **fit_params
            Forwarded to :meth:`fit` (e.g. a precomputed ``graph``).

        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
        """
        # Fit model
        self.fit(X, coords, **fit_params)

        # Convert predictions to float array
        X = np.asarray(X, dtype=float)

        # Unscale predictions
        preds_np = inverse_transform(
            reconstruct(self.model_, self.full_loader_, self.device_, verbose=self.verbose).numpy(),
            self.scaler_dict_, self.feature_names_,
        )

        X_out = X.copy()
        X_out[np.isnan(X_out)] = preds_np[np.isnan(X_out)]
        return X_out

    def predict_with_uncertainty(self, X, coords, return_realisations: bool = False, graph=None):
        """
        Impute missing values and return per-point uncertainty estimates.

        Uses MC Dropout: the model is run ``n_inference`` times in train mode
        (dropout active). The spread of predictions across runs is the epistemic
        uncertainty; the model's predicted log-variance head gives aleatoric
        uncertainty.

        Requires ``n_inference > 1`` to produce a meaningful epistemic estimate;
        with ``n_inference=1`` ``epistemic_std`` will be all zeros.

        Aleatoric uncertainty requires ``loss_name='hetero'`` or ``'beta_nll'``.
        With ``loss_name='mse'`` the variance head is never trained.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        coords : array-like of shape (n_samples, 4)
        return_realisations : bool, default=False
            If True, also return the individual MC-dropout draws (physical
            units) instead of only their aggregate mean/std, e.g. for
            downstream methods that need the distribution across draws rather
            than its first two moments.
        graph : dict or None, default=None
            Precomputed k-NN graph for these coordinates.

        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
            Copy of ``X`` with ``NaN`` positions filled by the mean prediction.
        aleatoric_std : ndarray of shape (n_samples, n_features)
            Per-point aleatoric (data) uncertainty, in physical units
        epistemic_std : ndarray of shape (n_samples, n_features)
            Per-point epistemic (model) uncertainty, in physical units
        realisations : ndarray of shape (n_inference, n_samples, n_features), optional
            Only returned when ``return_realisations=True``: the individual
            per-pass mean predictions, inverse-transformed to physical units.
        """
        check_is_fitted(self, ["model_", "scaler_dict_"])

        # MSE loss does not train the variance head -> warn
        mse_loss = self.loss_name == "mse"
        if mse_loss:
            warnings.warn(
                "loss_name='mse' does not train the variance head and aleatoric_std is returned as NaN.",
                UserWarning, stacklevel=2,
            )

        X = np.asarray(X, dtype=float)
        coords = np.asarray(coords, dtype=float)

        # Pre-process input data
        coords_scaled, values_scaled, _ = preprocess(
            torch.as_tensor(coords, dtype=torch.float32),
            torch.as_tensor(X, dtype=torch.float32),
            self.feature_names_, self.cyclic_time,
            scaler_dict=self.scaler_dict_,
        )

        # Neighbourhood graphs (reused from fit() when coords are the training coords)
        graph = self._inference_graph(coords_scaled, graph=graph)
        loader = self._inference_loader(coords_scaled, values_scaled, graph)

        # Predict mean, epistemic and aleatoric uncertainties
        result = reconstruct_with_uncertainty(
            self.model_, loader, self.device_, n_inference=self.n_inference,
            verbose=self.verbose, return_realisations=return_realisations,
        )
        if return_realisations:
            mean_t, aleatoric_t, epistemic_t, realisations_t = result
        else:
            mean_t, aleatoric_t, epistemic_t = result

        # Unscale predictions and uncertainties to physical units
        mean_np = inverse_transform(mean_t.numpy(), self.scaler_dict_, self.feature_names_)
        aleatoric_np = inverse_transform_std(aleatoric_t.numpy(), self.scaler_dict_, self.feature_names_)
        if mse_loss:
            # Return NaN variance for MSE loss
            aleatoric_np = np.full_like(aleatoric_np, np.nan)
        epistemic_np = inverse_transform_std(epistemic_t.numpy(), self.scaler_dict_, self.feature_names_)

        X_out = X.copy()
        X_out[np.isnan(X_out)] = mean_np[np.isnan(X_out)]

        # Return all inference passes if requested
        if return_realisations:
            n_inference, n_samples, n_features = realisations_t.shape
            realisations_np = inverse_transform(
                realisations_t.reshape(-1, n_features).numpy(),
                self.scaler_dict_, self.feature_names_,
            ).reshape(n_inference, n_samples, n_features)
            return X_out, aleatoric_np, epistemic_np, realisations_np
        return X_out, aleatoric_np, epistemic_np
