"""MaSTNetImputer — sklearn-compatible spatio-temporal imputer."""

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader

from mastnet._graph import NeighbourDataset, build_graph
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
        space. A cylinder is defined as a horizontal disk at this radius —
        depth is excluded from the distance, so all depths at a geographic
        location fall inside together.
    cylindrical_mask_prob : float, default=0.3
        Per-batch probability of applying the cylindrical block mask during
        training. This forces the model to reconstruct entire spatial patches
        rather than individual points and is important for generalisation.
        Set to ``0`` to disable.
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
    >>> imputer = MaSTNetImputer(max_epochs=50, n_neighbours_space=15, n_neighbours_time=15)
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
        patience: int = 10,
        val_frac: float = 0.2,
        cyclic_time: bool = False,
        device: str = "auto",
        seed: int = 42,
        feature_names=None,
        n_inference: int = 50,
    ):
        self.n_neighbours_space    = n_neighbours_space
        self.n_neighbours_time     = n_neighbours_time
        self.d_model               = d_model
        self.nhead                 = nhead
        self.n_spatial_layers      = n_spatial_layers
        self.n_temporal_layers     = n_temporal_layers
        self.dim_feedforward       = dim_feedforward
        self.dropout               = dropout
        self.use_masks             = use_masks
        self.fill_strategy         = fill_strategy
        self.loss_name             = loss_name
        self.max_epochs            = max_epochs
        self.lr                    = lr
        self.batch_size            = batch_size
        self.mask_ratio            = mask_ratio
        self.cylindrical_mask_size = cylindrical_mask_size
        self.cylindrical_mask_prob = cylindrical_mask_prob
        self.patience              = patience
        self.val_frac              = val_frac
        self.cyclic_time           = cyclic_time
        self.device                = device
        self.seed                  = seed
        self.feature_names         = feature_names
        self.n_inference           = n_inference

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

    def _make_loader(self, coords, values, graph, indices=None, shuffle=False, generator=None):
        ds = NeighbourDataset(coords, values, graph, query_indices=indices)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          generator=generator if shuffle else None)

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, coords, y=None):
        """
        Fit MaSTNeT on spatio-temporal data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix; ``NaN`` marks missing values.
        coords : array-like of shape (n_samples, 4)
            Spatio-temporal coordinates as ``[lat (°), lon (°), depth (m), time]``.
        y : ignored

        Returns
        -------
        self
        """
        self.feature_names_ = self._resolve_feature_names(X)
        X      = np.asarray(X,      dtype=float)
        coords = np.asarray(coords, dtype=float)

        device    = self._resolve_device()
        generator = _set_seed(self.seed)

        # Train / validation split
        idx = np.arange(len(X))
        train_idx, val_idx = train_test_split(idx, test_size=self.val_frac,
                                              random_state=self.seed)

        # Preprocess: fit scalers on training rows, transform full dataset
        coords_t = torch.as_tensor(coords, dtype=torch.float32)
        values_t = torch.as_tensor(X,      dtype=torch.float32)

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

        self.scaler_dict_  = scaler_dict
        self.coords_full_  = coords_full
        self.values_full_  = values_full

        # Global means for fill_strategy='mean'
        obs = values_full.clone()
        obs[~mask_full] = float("nan")
        global_means = torch.nan_to_num(torch.nanmean(obs, dim=0), nan=0.0).tolist()

        # Build graph (validation feature values masked to prevent leakage)
        values_for_graph = values_full.clone()
        values_for_graph[val_idx] = float("nan")
        graph = build_graph(coords_full, self.n_neighbours_space, self.n_neighbours_time)
        self.graph_ = graph

        # Loaders
        train_loader = self._make_loader(coords_full, values_full, graph,
                                         indices=train_idx, shuffle=True,
                                         generator=generator)
        val_loader   = self._make_loader(coords_full, values_full, graph,
                                         indices=val_idx)
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
        )

        self.device_       = device
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

        X      = np.asarray(X,      dtype=float)
        coords = np.asarray(coords, dtype=float)

        # Pre-process input data
        coords_scaled, values_scaled, _ = preprocess(
            torch.as_tensor(coords, dtype=torch.float32),
            torch.as_tensor(X,      dtype=torch.float32),
            self.feature_names_, self.cyclic_time,
            scaler_dict=self.scaler_dict_,
        )

        # Build neighbourhood graphs
        graph  = build_graph(coords_scaled, self.n_neighbours_space, self.n_neighbours_time)
        loader = self._make_loader(coords_scaled, values_scaled, graph)

        # Predict and unscale
        preds_np = inverse_transform(
            reconstruct(self.model_, loader, self.device_).numpy(),
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

        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
        """
        # Fit model
        self.fit(X, coords)

        # Convert predictions to float array
        X = np.asarray(X, dtype=float)

        # Unscale predictions
        preds_np = inverse_transform(
            reconstruct(self.model_, self.full_loader_, self.device_).numpy(),
            self.scaler_dict_, self.feature_names_,
        )

        X_out = X.copy()
        X_out[np.isnan(X_out)] = preds_np[np.isnan(X_out)]
        return X_out

    def predict_with_uncertainty(self, X, coords):
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

        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
            Copy of ``X`` with ``NaN`` positions filled by the mean prediction.
        aleatoric_std : ndarray of shape (n_samples, n_features)
            Per-point aleatoric (data) uncertainty: ``sqrt(mean(exp(logvar)))``.
        epistemic_std : ndarray of shape (n_samples, n_features)
            Per-point epistemic (model) uncertainty: std of means across MC passes.
        """
        check_is_fitted(self, ["model_", "scaler_dict_"])

        X      = np.asarray(X,      dtype=float)
        coords = np.asarray(coords, dtype=float)

        # Pre-process input data
        coords_scaled, values_scaled, _ = preprocess(
            torch.as_tensor(coords, dtype=torch.float32),
            torch.as_tensor(X,      dtype=torch.float32),
            self.feature_names_, self.cyclic_time,
            scaler_dict=self.scaler_dict_,
        )

        # Build neighbourhood graphs
        graph  = build_graph(coords_scaled, self.n_neighbours_space, self.n_neighbours_time)
        loader = self._make_loader(coords_scaled, values_scaled, graph)

        # Predict mean, epistemic and aleatoric uncertainties
        mean_t, aleatoric_t, epistemic_t = reconstruct_with_uncertainty(
            self.model_, loader, self.device_, n_inference=self.n_inference
        )

        # Unscale predictions
        mean_np      = inverse_transform(mean_t.numpy(), self.scaler_dict_, self.feature_names_)

        # Convert to numpy
        aleatoric_np = aleatoric_t.numpy()
        epistemic_np = epistemic_t.numpy()

        X_out = X.copy()
        X_out[np.isnan(X_out)] = mean_np[np.isnan(X_out)]
        return X_out, aleatoric_np, epistemic_np
