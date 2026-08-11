"""MaSTNetImputer - sklearn-compatible spatio-temporal imputer."""

import logging
from time import time
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

from mastnet._graph import NeighbourDataset, build_graph, precompute_graph
from mastnet._model import MaSTNeT
from mastnet._preprocessing import inverse_transform, preprocess
from mastnet._training import fit, reconstruct, reconstruct_with_uncertainty


def _set_seed(seed: int) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class MaSTNetImputer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible imputer using MaSTNeT (Masked Spatio-temporal Neighbourhood Transformer).

    MaSTNeT reconstructs missing values by attending to k-nearest spatial and j-nearest
    temporal neighbours in a factorised cross-attention architecture. A
    heteroscedastic output head provides per-point uncertainty estimates.

    Coordinates are expected as four columns in this order:

        ``[latitude (°), longitude (°), depth (m), time]``

    Latitude and longitude are mapped internally to Cartesian sphere coordinates.
    Depth and time are MinMax-scaled. Features are scaled independently per
    variable on observed values only.

    Parameters
    ----------
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
    use_masks : bool, default=True
        Append the binary observation mask to each token.
    fill_strategy : str, default='zero'
        How to fill unobserved positions before attention: ``'zero'`` or
        ``'mean'`` (global per-feature mean estimated from training data).
    loss_name : str, default='hetero'
        Training loss: ``'hetero'`` for heteroscedastic Gaussian NLL
        (trains the variance head) or ``'mse'`` for plain masked MSE.
    max_epochs : int, default=100
        Maximum number of training epochs.
    lr : float, default=1e-3
        Adam learning rate.
    batch_size : int, default=64
        Mini-batch size for training and inference.
    mask_ratio : float, default=0.5
        Fraction of observed values randomly hidden during each training step.
    cylindrical_mask_size : float, default=0.1
        Radius of the cylindrical block mask in Cartesian (x, y, z) coordinate
        space. A cylinder is defined as a horizontal disk at this radius -
        depth is excluded from the distance, so all depths at a geographic
        location fall inside together.
    cylindrical_mask_prob : float, default=0.3
        Per-batch probability of applying the cylindrical block mask during
        training. This forces the model to reconstruct entire spatial patches
        rather than individual points and is important for generalisation.
        Set to ``0`` to disable.
    grad_clip_norm : float, default=1.0
        Maximum gradient norm; gradients are clipped with
        :func:`torch.nn.utils.clip_grad_norm_` before every optimiser step.
        Stabilises heteroscedastic training, which can otherwise diverge to
        NaN. Set to ``0`` to disable.
    patience : int, default=10
        Early-stopping patience (epochs without validation improvement).
    val_frac : float, default=0.2
        Fraction of rows held out for internal validation.
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
    val_method : str, default='random'
        How to create the internal validation split: ``'random'`` draws a
        uniform random subset; ``'block'`` assigns points to spatial lat/lon
        blocks and holds out a contiguous set of blocks, preventing spatial
        autocorrelation from leaking into the validation signal (recommended
        for geographic data).
    block_size_deg : float, default=15.0
        Width/height of each spatial block in degrees. Only used when
        ``val_method='block'``.
    val_indices : array-like of int or None, default=None
        Explicit row indices to hold out as the internal validation set for
        early stopping. Takes precedence over ``val_method`` and ``val_frac``
        when provided. Useful to reproduce an external experiment's fixed
        train/validation split exactly.
    verbose : bool, default=True
        Show progress bars and per-epoch log lines during training and
        inference. Set to ``False`` to silence all console output (e.g.
        inside a grid search or cross-validation loop).

    Attributes
    ----------
    model_ : MaSTNeT
        Trained PyTorch model.
    scaler_dict_ : dict
        Fitted coordinate and feature scalers.
    feature_names_ : list of str
        Resolved feature names set during :meth:`fit`. Always a plain list
        of strings regardless of how ``feature_names`` was provided.
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np
    >>> from mastnet import MaSTNetImputer
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> X = rng.standard_normal((n, 6))
    >>> X[rng.random((n, 6)) < 0.3] = np.nan          # 30 % missing
    >>> coords = np.column_stack([
    ...     rng.uniform(-90, 90, n),                   # latitude
    ...     rng.uniform(-180, 180, n),                 # longitude
    ...     rng.uniform(0, 5000, n),                   # depth (m)
    ...     rng.integers(1990, 2020, n).astype(float), # year
    ... ])
    >>> imputer = MaSTNetImputer(max_epochs=50, n_neighbours_space=15, n_neighbours_time=15, verbose=False)
    >>> X_filled = imputer.fit_transform(X, coords)
    """

    def __init__(
            self,
            n_neighbours_space: int = 30,
            n_neighbours_time: int = 30,
            d_model: int = 64,
            nhead: int = 4,
            n_spatial_layers: int = 3,
            n_temporal_layers: int = 3,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
            use_masks: bool = True,
            fill_strategy: str = "zero",
            loss_name: str = "hetero",
            max_epochs: int = 100,
            lr: float = 1e-3,
            batch_size: int = 64,
            mask_ratio: float = 0.5,
            cylindrical_mask_size: float = 0.1,
            cylindrical_mask_prob: float = 0.3,
            grad_clip_norm: float = 1.0,
            patience: int = 10,
            val_frac: float = 0.2,
            cyclic_time: bool = False,
            device: str = "auto",
            seed: int = 42,
            feature_names=None,
            n_inference: int = 50,
            val_method: str = "random",
            block_size_deg: float = 15.0,
            val_indices=None,
            verbose: bool = True,
    ):
        self.n_neighbours_space = n_neighbours_space
        self.n_neighbours_time = n_neighbours_time
        self.d_model = d_model
        self.nhead = nhead
        self.n_spatial_layers = n_spatial_layers
        self.n_temporal_layers = n_temporal_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.use_masks = use_masks
        self.fill_strategy = fill_strategy
        self.loss_name = loss_name
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.cylindrical_mask_size = cylindrical_mask_size
        self.cylindrical_mask_prob = cylindrical_mask_prob
        self.grad_clip_norm = grad_clip_norm
        self.patience = patience
        self.val_frac = val_frac
        self.cyclic_time = cyclic_time
        self.device = device
        self.seed = seed
        self.feature_names = feature_names
        self.n_inference = n_inference
        self.val_method = val_method
        self.block_size_deg = block_size_deg
        self.val_indices = val_indices
        self.verbose = verbose

    def _make_train_val_split(self, coords_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (train_idx, val_idx) from ``val_indices`` if given, else according to ``val_method``."""
        idx = np.arange(len(coords_raw))
        if self.val_indices is not None:
            val_idx = np.asarray(self.val_indices, dtype=int)
            return np.setdiff1d(idx, val_idx), val_idx
        if self.val_method == "block":
            lat = coords_raw[:, 0]
            lon = coords_raw[:, 1]
            lat_block = np.floor(lat / self.block_size_deg).astype(int)
            lon_block = np.floor(lon / self.block_size_deg).astype(int)
            block_ids = lat_block * 10000 + lon_block
            unique_blocks = np.unique(block_ids)
            rng = np.random.default_rng(self.seed)
            rng.shuffle(unique_blocks)
            n_val = max(1, int(len(unique_blocks) * self.val_frac))
            val_set = set(unique_blocks[:n_val].tolist())
            in_val = np.array([b in val_set for b in block_ids])
            return idx[~in_val], idx[in_val]
        else:
            return train_test_split(idx, test_size=self.val_frac, random_state=self.seed)

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

    def _inference_graph(self, coords_scaled: torch.Tensor) -> dict:
        """
        Return the k-NN graph for inference coordinates.

        Reuses the graph built during :meth:`fit` when ``coords_scaled`` is
        identical to the training coordinates (the common transform-after-fit
        case), skipping a rebuild that can take many minutes on large datasets.
        """
        # Skip graph building
        if torch.equal(coords_scaled, self.coords_full_):
            logger.info("Coordinates match the training data; reusing the k-NN graph from fit().")
            return self.graph_
        logger.info(
            "Building k-NN graph (n=%d, k_space=%d, k_time=%d)...",
            len(coords_scaled), self.n_neighbours_space, self.n_neighbours_time,
        )

        # Build graph
        t0 = time()
        graph = build_graph(coords_scaled, self.n_neighbours_space, self.n_neighbours_time)
        logger.info("Graph built in %.1fs.", time() - t0)
        return graph

    def _make_loader(self, coords, values, graph, indices=None, shuffle=False,
                     generator=None, neighbour_values=None):
        ds = NeighbourDataset(coords, values, graph, query_indices=indices,
                              neighbour_values=neighbour_values)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          generator=generator if shuffle else None)

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, coords, y=None, graph=None):
        """
        Fit MaSTNeT on spatio-temporal data.

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
            :func:`mastnet.precompute_graph`. Each scope may provide *more*
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

        device = self._resolve_device()
        generator = _set_seed(self.seed)

        # Train / validation split
        train_idx, val_idx = self._make_train_val_split(coords)

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
            graph = build_graph(coords_full, self.n_neighbours_space, self.n_neighbours_time)
            logger.info("Graph built in %.1fs.", time() - t0)
        self.graph_ = graph

        # Loaders (train/val hide validation values from neighbourhoods)
        train_loader = self._make_loader(coords_full, values_full, graph,
                                         indices=train_idx, shuffle=True,
                                         generator=generator,
                                         neighbour_values=neighbour_values)
        val_loader = self._make_loader(coords_full, values_full, graph,
                                       indices=val_idx,
                                       neighbour_values=neighbour_values)
        self.full_loader_ = self._make_loader(coords_full, values_full, graph)

        # Model
        coord_dim = coords_full.shape[1]
        self.model_ = MaSTNeT(
            n_features=X.shape[1],
            coord_dim=coord_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            n_spatial_layers=self.n_spatial_layers,
            n_temporal_layers=self.n_temporal_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            use_masks=self.use_masks,
            fill_strategy=self.fill_strategy,
            global_means=global_means if self.fill_strategy == "mean" else None,
        ).to(device)

        fit(
            self.model_, train_loader, val_loader,
            max_epochs=self.max_epochs, lr=self.lr,
            mask_ratio=self.mask_ratio, patience=self.patience,
            device=device, loss_name=self.loss_name,
            cylindrical_mask_size=self.cylindrical_mask_size,
            cylindrical_mask_prob=self.cylindrical_mask_prob,
            grad_clip_norm=self.grad_clip_norm,
            verbose=self.verbose,
        )

        self.device_ = device
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, coords):
        """
        Impute missing values using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        coords : array-like of shape (n_samples, 4)

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
        graph = self._inference_graph(coords_scaled)
        loader = self._make_loader(coords_scaled, values_scaled, graph)

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

    def predict_with_uncertainty(self, X, coords, return_realizations: bool = False):
        """
        Impute missing values and return per-point uncertainty estimates.

        Uses MC Dropout: the model is run ``n_inference`` times in train mode
        (dropout active). The spread of predictions across runs is the epistemic
        uncertainty; the model's predicted log-variance head gives aleatoric
        uncertainty.

        Requires ``n_inference > 1`` to produce a meaningful epistemic estimate;
        with ``n_inference=1`` ``epistemic_std`` will be all zeros.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        coords : array-like of shape (n_samples, 4)
        return_realizations : bool, default=False
            If True, also return the individual MC-dropout draws (physical
            units) instead of only their aggregate mean/std, e.g. for
            downstream methods that need the distribution across draws rather
            than its first two moments.

        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
            Copy of ``X`` with ``NaN`` positions filled by the mean prediction.
        aleatoric_std : ndarray of shape (n_samples, n_features)
            Per-point aleatoric (data) uncertainty: ``sqrt(mean(exp(logvar)))``.
        epistemic_std : ndarray of shape (n_samples, n_features)
            Per-point epistemic (model) uncertainty: std of means across MC passes.
        realizations : ndarray of shape (n_inference, n_samples, n_features), optional
            Only returned when ``return_realizations=True``: the individual
            per-pass mean predictions, inverse-transformed to physical units.
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
        graph = self._inference_graph(coords_scaled)
        loader = self._make_loader(coords_scaled, values_scaled, graph)

        # Predict mean, epistemic and aleatoric uncertainties
        result = reconstruct_with_uncertainty(
            self.model_, loader, self.device_, n_inference=self.n_inference,
            verbose=self.verbose, return_realizations=return_realizations,
        )
        if return_realizations:
            mean_t, aleatoric_t, epistemic_t, realizations_t = result
        else:
            mean_t, aleatoric_t, epistemic_t = result

        # Unscale predictions
        mean_np = inverse_transform(mean_t.numpy(), self.scaler_dict_, self.feature_names_)

        # Convert to numpy
        aleatoric_np = aleatoric_t.numpy()
        epistemic_np = epistemic_t.numpy()

        X_out = X.copy()
        X_out[np.isnan(X_out)] = mean_np[np.isnan(X_out)]

        if return_realizations:
            n_inference, n_samples, n_features = realizations_t.shape
            realizations_np = inverse_transform(
                realizations_t.reshape(-1, n_features).numpy(),
                self.scaler_dict_, self.feature_names_,
            ).reshape(n_inference, n_samples, n_features)
            return X_out, aleatoric_np, epistemic_np, realizations_np
        return X_out, aleatoric_np, epistemic_np
