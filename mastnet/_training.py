"""Training loop, losses, and early stopping for MaSTNeT."""

from copy import deepcopy
from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


# ------------------------------------------------------------------
# Losses
# ------------------------------------------------------------------

def _masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """MSE loss computed only on observed (masked-in) positions."""
    return F.mse_loss(pred[mask], target[mask])


def _heteroscedastic_nll(pred_mean: Tensor, pred_logvar: Tensor,
                         target: Tensor, mask: Tensor) -> Tensor:
    """Gaussian NLL with predicted log-variance (Kendall & Gal 2017)."""
    var = pred_logvar[mask].exp().clamp(min=1e-6)
    return (0.5 * (pred_logvar[mask] + (target[mask] - pred_mean[mask]) ** 2 / var)).mean()


def _build_loss(loss_name: str) -> Callable:
    """Return the loss callable corresponding to ``loss_name``."""
    if loss_name == "mse":
        return lambda mean, logvar, target, mask: _masked_mse(mean, target, mask)
    elif loss_name == "hetero":
        return _heteroscedastic_nll
    else:
        raise ValueError(f"Unknown loss_name '{loss_name}'. Choose 'mse' or 'hetero'.")


# ------------------------------------------------------------------
# Masking
# ------------------------------------------------------------------

def cylindrical_feature_mask(
    batch: dict,
    feature_dim: int,
    size: float = 0.1,
    p: float = 0.3,
    device: torch.device = "cpu",
) -> Tensor:
    """
    Block-mask all features of every query point inside a spatial cylinder.

    The "cylinder" is defined by a disk of radius ``size`` in the Cartesian
    (x, y, z) subspace of ``query_coords`` (columns 0–2 after preprocessing).
    Because depth is excluded from the distance calculation, all depths at a
    given horizontal position fall inside the cylinder together.

    Applied with probability ``p``; returns an all-False mask otherwise.

    Parameters
    ----------
    batch : dict
        Batch dict from :class:`mastnet._graph.NeighbourDataset`.
    feature_dim : int
        Number of features (``F``).
    size : float, default=0.1
        Cylinder radius in Cartesian coordinate space.
    p : float, default=0.3
        Probability of applying the mask on any given call.
    device : torch.device

    Returns
    -------
    Tensor of shape (B, F), dtype bool
    """
    coords = batch["query_coords"][:, :3]   # Cartesian (x, y, z); no depth, no time
    B = coords.shape[0]

    if torch.rand(1).item() > p:
        return torch.zeros(B, feature_dim, dtype=torch.bool, device=device)

    idx = torch.randint(0, B, (1,), device=device)
    center = coords[idx]                            # (1, 3)
    dist = torch.norm(coords - center, dim=1)       # (B,)
    mask_block = dist < size                        # (B,)

    return mask_block.unsqueeze(1).expand(B, feature_dim)


def _apply_random_mask(
    batch: dict,
    mask_ratio: float,
    device: torch.device,
    cylindrical_mask_size: float = 0.0,
    cylindrical_mask_prob: float = 0.0,
) -> tuple[dict, Tensor]:
    """
    Hide a fraction of the observed query features (random + optional cylindrical block).

    Parameters
    ----------
    batch : dict
        Batch dict from :class:`mastnet._graph.NeighbourDataset`.
    mask_ratio : float
        Fraction of observed query values to hide for reconstruction.
    device : torch.device
    cylindrical_mask_size : float, default=0.0
        Cylinder radius for the block mask; ignored when ``cylindrical_mask_prob=0``.
    cylindrical_mask_prob : float, default=0.0
        Probability of applying the cylindrical block mask per batch step.
        Set to ``0`` to disable.

    Returns
    -------
    batch : dict
        Shallow copy with ``query_mask`` and neighbour masks reduced.
    loss_mask : Tensor of shape (B, F)
        Boolean mask selecting positions to compute reconstruction loss on.
    """
    q_mask = batch["query_mask"]            # (B, F) bool — truly observed
    B, F = q_mask.shape
    rand = torch.rand(B, F, device=device) < mask_ratio

    if cylindrical_mask_prob > 0.0:
        cyl = cylindrical_feature_mask(
            batch, F,
            size=cylindrical_mask_size,
            p=cylindrical_mask_prob,
            device=device,
        )
        rand = rand | cyl

    input_mask = q_mask & ~rand             # model sees these
    loss_mask  = q_mask & rand              # reconstruct these

    # Also independently mask some neighbour features to prevent shortcut copying
    batch = {**batch}                       # shallow copy
    batch["query_mask"] = input_mask
    for scope in ("space", "time"):
        if scope in batch:
            nb = batch[scope]
            nb_mask = nb["mask"]
            nb_rand = torch.rand_like(nb_mask.float()) < mask_ratio
            batch[scope] = {**nb, "mask": nb_mask & ~nb_rand.bool()}

    return batch, loss_mask


def _to_device(obj, device: torch.device):
    """Recursively move tensors in a nested dict/Tensor structure to ``device``."""
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, Tensor):
        return obj.to(device)
    return obj


# ------------------------------------------------------------------
# Early stopping
# ------------------------------------------------------------------

class EarlyStopping:
    """
    Early stopping based on validation loss improvement.

    Parameters
    ----------
    patience : int, optional
        Number of epochs to wait without improvement before stopping,
        by default 10.
    min_delta_ratio : float, optional
        Minimum fractional improvement required to count as a new best,
        by default 1e-4.
    """

    def __init__(self, patience: int = 10, min_delta_ratio: float = 1e-4):
        self.patience        = patience
        self.min_delta_ratio = min_delta_ratio
        self.best_loss  = np.inf
        self.counter    = 0
        self.best_epoch = 0
        self.epoch      = 0

    def __call__(self, val_loss: float) -> bool:
        delta = max(1e-8, abs(val_loss) * self.min_delta_ratio)
        if val_loss < self.best_loss - delta:
            self.best_loss  = val_loss
            self.best_epoch = self.epoch
            self.counter    = 0
        else:
            self.counter += 1
        self.epoch += 1
        return self.counter >= self.patience


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def _train_epoch(
    model, loader: DataLoader, optimizer, loss_fn: Callable,
    mask_ratio: float, device: torch.device,
    cylindrical_mask_size: float = 0.0,
    cylindrical_mask_prob: float = 0.0,
) -> float:
    """Run one training epoch; return mean loss per observed value."""
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        batch, loss_mask = _apply_random_mask(
            batch, mask_ratio, device,
            cylindrical_mask_size=cylindrical_mask_size,
            cylindrical_mask_prob=cylindrical_mask_prob,
        )
        if not loss_mask.any():
            continue
        pred_mean, pred_logvar = model(batch)
        target = batch["query_features"]
        loss = loss_fn(pred_mean, pred_logvar, target, loss_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * loss_mask.sum().item()
        n     += loss_mask.sum().item()
    return total / max(n, 1)


@torch.no_grad()
def _eval_epoch(
    model, loader: DataLoader, loss_fn: Callable,
    mask_ratio: float, device: torch.device,
    cylindrical_mask_size: float = 0.0,
    cylindrical_mask_prob: float = 0.0,
) -> float:
    """Evaluate on ``loader`` without gradient updates; return mean loss."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        batch, loss_mask = _apply_random_mask(
            batch, mask_ratio, device,
            cylindrical_mask_size=cylindrical_mask_size,
            cylindrical_mask_prob=cylindrical_mask_prob,
        )
        if not loss_mask.any():
            continue
        pred_mean, pred_logvar = model(batch)
        target = batch["query_features"]
        loss = loss_fn(pred_mean, pred_logvar, target, loss_mask)
        total += loss.item() * loss_mask.sum().item()
        n     += loss_mask.sum().item()
    return total / max(n, 1)


@torch.no_grad()
def reconstruct(model, loader: DataLoader, device: torch.device) -> Tensor:
    """
    Run full reconstruction without masking; return predicted means.

    Parameters
    ----------
    model : MaSTNeT
    loader : DataLoader
    device : torch.device

    Returns
    -------
    Tensor of shape (n, n_features)
        Predicted means for every point in the loader.
    """
    model.eval()
    preds = []
    for batch in loader:
        batch = _to_device(batch, device)
        pred_mean, _ = model(batch)
        preds.append(pred_mean.cpu())
    return torch.cat(preds)


def reconstruct_with_uncertainty(
    model, loader: DataLoader, device: torch.device, n_inference: int = 10
) -> tuple[Tensor, Tensor, Tensor]:
    """
    MC Dropout inference to estimate both aleatoric and epistemic uncertainty.

    The model is kept in *train* mode so dropout layers remain active. Each of
    the ``n_inference`` forward passes produces a slightly different prediction;
    their spread is the epistemic (model) uncertainty.

    Parameters
    ----------
    model : MaSTNeT
    loader : DataLoader
    device : torch.device
    n_inference : int, default=10
        Number of stochastic forward passes.

    Returns
    -------
    mean : Tensor of shape (n, n_features)
        Mean prediction across all passes.
    aleatoric_std : Tensor of shape (n, n_features)
        Per-point aleatoric (data) uncertainty: ``sqrt(mean(exp(logvar)))``.
    epistemic_std : Tensor of shape (n, n_features)
        Per-point epistemic (model) uncertainty: std of means across passes.
    """
    model.train()   # activates dropout
    all_means: list[Tensor] = []
    all_logvars: list[Tensor] = []

    with torch.no_grad():
        for _ in range(n_inference):
            batch_means: list[Tensor] = []
            batch_logvars: list[Tensor] = []
            for batch in loader:
                batch = _to_device(batch, device)
                pred_mean, pred_logvar = model(batch)
                batch_means.append(pred_mean.cpu())
                batch_logvars.append(pred_logvar.cpu())
            all_means.append(torch.cat(batch_means))
            all_logvars.append(torch.cat(batch_logvars))

    model.eval()

    stacked_means   = torch.stack(all_means)    # (n_inference, n, F)
    stacked_logvars = torch.stack(all_logvars)  # (n_inference, n, F)

    mean           = stacked_means.mean(0)
    epistemic_std  = stacked_means.std(0)
    aleatoric_std  = (stacked_logvars.mean(0) * 0.5).exp()   # sqrt(exp(logvar))

    return mean, aleatoric_std, epistemic_std


def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
    lr: float,
    mask_ratio: float,
    patience: int,
    device: torch.device,
    loss_name: str = "hetero",
    cylindrical_mask_size: float = 0.0,
    cylindrical_mask_prob: float = 0.0,
) -> None:
    """
    Train ``model`` in-place with early stopping; restores best weights.

    Parameters
    ----------
    model : MaSTNeT
    train_loader : DataLoader
    val_loader : DataLoader
    max_epochs : int
    lr : float
        Adam learning rate.
    mask_ratio : float
        Fraction of observed values randomly hidden per training step.
    patience : int
        Early-stopping patience in epochs.
    device : torch.device
    loss_name : str, optional
        ``'hetero'`` for heteroscedastic Gaussian NLL or ``'mse'`` for
        plain masked MSE, by default ``'hetero'``.
    cylindrical_mask_size : float, default=0.0
        Radius of the cylindrical block mask in Cartesian coordinate space.
        Ignored when ``cylindrical_mask_prob=0``.
    cylindrical_mask_prob : float, default=0.0
        Per-batch probability of applying the cylindrical block mask during
        training **and** validation. Set to ``0`` to disable.
    """
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn    = _build_loss(loss_name)
    stopper    = EarlyStopping(patience=patience)
    best_state = deepcopy(model.state_dict())

    for epoch in tqdm(range(max_epochs), desc="Training", unit="epoch"):
        train_loss = _train_epoch(
            model, train_loader, optimizer, loss_fn, mask_ratio, device,
            cylindrical_mask_size=cylindrical_mask_size,
            cylindrical_mask_prob=cylindrical_mask_prob,
        )
        val_loss = _eval_epoch(
            model, val_loader, loss_fn, mask_ratio, device,
            cylindrical_mask_size=cylindrical_mask_size,
            cylindrical_mask_prob=cylindrical_mask_prob,
        )
        tqdm.write(f"Epoch {epoch:03d}: train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < stopper.best_loss:
            best_state = deepcopy(model.state_dict())
        if stopper(val_loss):
            tqdm.write(f"Early stopping at epoch {epoch} (best={stopper.best_epoch})")
            break

    model.load_state_dict(best_state)
