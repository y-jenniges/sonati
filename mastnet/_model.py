"""MaSTNeT model - factorised time-space cross-attention imputer."""

import torch
import torch.nn as nn
from torch import Tensor

# Predicted log-variance is clamped to this symmetric range. Without it the
# heteroscedastic NLL can drive the logvar head to extreme values whose exp()
# overflows and collapses training to NaN.
LOGVAR_CLAMP = 20.0


class _CrossAttentionBlock(nn.Module):
    """Single cross-attention layer: query attends to a neighbourhood."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        # Multi-head attention (only query attends to neighbours)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        """Apply cross-attention from ``q`` to ``kv``, then feedforward."""
        attn_out, _ = self.attn(q, kv, kv)
        q = self.norm1(q + attn_out)
        return self.norm2(q + self.mlp(q))


class MaSTNeT(nn.Module):
    """
    Masked Spatio-temporal Neighbourhood Transformer.

    Reconstructs missing values in spatio-temporal point-cloud data by attending
    to k-nearest temporal neighbours first, then k-nearest spatial neighbours
    (factorised time-space cross-attention).

    The model outputs a per-point mean **and** log-variance for every feature,
    supporting heteroscedastic uncertainty quantification.

    Parameters
    ----------
    n_features : int
        Number of variables to impute.
    coord_dim : int
        Dimensionality of the (preprocessed) coordinate vector: 5 for linear
        time encoding, 6 for cyclic time encoding.
    d_model : int
        Transformer embedding dimension.
    nhead : int
        Number of attention heads (must divide ``d_model``).
    n_spatial_layers : int
        Cross-attention layers applied over spatial neighbours.
    n_temporal_layers : int
        Cross-attention layers applied over temporal neighbours.
    dim_feedforward : int
        Hidden width of the feedforward sublayer in each attention block.
    dropout : float
        Dropout probability (active only during training).
    fill_strategy : str
        How to fill unobserved feature positions before computing attention:
        ``'zero'`` fills with 0; ``'mean'`` fills with the per-feature global
        mean estimated from training data.
    global_means : list[float] or None
        Per-feature means used when ``fill_strategy='mean'``. Registered as a
        buffer and automatically moved to the correct device.
    """

    def __init__(
            self,
            n_features: int,
            coord_dim: int = 5,
            d_model: int = 64,
            nhead: int = 4,
            n_spatial_layers: int = 3,
            n_temporal_layers: int = 3,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
            fill_strategy: str = "zero",
            global_means=None,
    ):
        super().__init__()
        self.n_features = n_features
        self.coord_dim = coord_dim
        self.fill_strategy = fill_strategy

        if global_means is not None:
            self.register_buffer(
                "global_means", torch.tensor(global_means, dtype=torch.float32)
            )
        else:
            self.global_means = None

        # Feature mixer: Small MLP mixing filled features and missingness information
        self.feature_mixer = nn.Sequential(
            nn.Linear(n_features * 2, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
        )

        # Input projection
        token_dim = coord_dim + n_features
        self.input_proj = nn.Linear(token_dim, d_model)

        # Spatial layers
        self.spatial_layers = nn.ModuleList([
            _CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(n_spatial_layers)
        ])

        # Temporal layers
        self.temporal_layers = nn.ModuleList([
            _CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(n_temporal_layers)
        ])

        # Output heads
        self.mean_head = nn.Linear(d_model, n_features)
        self.logvar_head = nn.Linear(d_model, n_features)

    def _fill(self, features: Tensor, mask: Tensor) -> Tensor:
        """Replace unobserved positions with zeros or global feature means."""
        if self.fill_strategy == "mean" and self.global_means is not None:
            fill = self.global_means.expand_as(features)
        else:
            fill = torch.zeros_like(features)
        return torch.where(mask, features, fill)

    def _token(self, coords: Tensor, features: Tensor, mask: Tensor) -> Tensor:
        """Fill, mix in the missingness mask, then concatenate the mixed features with coordinates."""
        filled = self._fill(features, mask)
        mixed = self.feature_mixer(torch.cat([filled, mask.float()], dim=-1))
        return torch.cat([coords, mixed], dim=-1)

    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        batch : dict with keys
            - ``query_features`` (B, F)
            - ``query_mask`` (B, F) bool - True = observed
            - ``query_coords`` (B, C)
            - ``time`` dict with ``features`` (B, K, F), ``mask`` (B, K, F),
                                 ``coords`` (B, K, C)
            - ``space`` same structure as ``time``

        Returns
        -------
        pred_mean : Tensor (B, F)
        pred_logvar : Tensor (B, F)
            Clamped to ``[-LOGVAR_CLAMP, LOGVAR_CLAMP]`` to keep the
            heteroscedastic NLL numerically stable.
        """
        # Project query token
        q = self.input_proj(
            self._token(batch["query_coords"], batch["query_features"], batch["query_mask"])
        ).unsqueeze(1)  # (B, 1, D)

        # Iterate over scopes (time first, then space)
        for scope, layers in (("time", self.temporal_layers), ("space", self.spatial_layers)):
            nb = batch[scope]

            # Project neighbour tokens
            n = self.input_proj(self._token(nb["coords"], nb["features"], nb["mask"]))

            # Apply layers
            for layer in layers:
                q = layer(q, n)

        # Only return query token
        h = q[:, 0]  # (B, D)
        pred_logvar = self.logvar_head(h).clamp(min=-LOGVAR_CLAMP, max=LOGVAR_CLAMP)
        return self.mean_head(h), pred_logvar
