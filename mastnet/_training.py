"""Training loop, losses, and early stopping for MaSTNeT."""

from copy import deepcopy
from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import sys
from time import strftime, time

logger = logging.getLogger(__name__)


def _tqdm_ok(verbose: bool) -> bool:
    """True only when stderr is an interactive terminal."""
    return verbose and sys.stderr.isatty()


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
        device: torch.device = torch.device("cpu"),
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
    coords = batch["query_coords"][:, :3]  # Cartesian (x, y, z); no depth, no time
    B = coords.shape[0]

    if torch.rand(1).item() > p:
        return torch.zeros(B, feature_dim, dtype=torch.bool, device=device)

    idx = torch.randint(0, B, (1,), device=device)
    center = coords[idx]  # (1, 3)
    dist = torch.norm(coords - center, dim=1)  # (B,)
    mask_block = dist < size  # (B,)

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
    q_mask = batch["query_mask"]  # (B, F) bool - truly observed
    B, F = q_mask.shape

    # Random per-sample masking
    n_mask = max(1, int(F * mask_ratio))  # number of features to hide
    rand_idx = torch.rand(B, F, device=device).argsort(dim=1)  # idxs to hide
    rand = torch.zeros(B, F, dtype=torch.bool, device=device)  # init bool array
    rand.scatter_(1, rand_idx[:, :n_mask], True)  # hide features

    # Apply cylindrical masking
    if cylindrical_mask_prob > 0.0:
        cyl = cylindrical_feature_mask(
            batch, F,
            size=cylindrical_mask_size,
            p=cylindrical_mask_prob,
            device=device,
        )
        rand = rand | cyl

    input_mask = q_mask & ~rand  # model sees these
    loss_mask = q_mask & rand  # reconstruct these

    # Also independently mask some neighbour features to prevent shortcut copying
    batch = {**batch, "query_mask": input_mask}  # shallow copy
    for scope in ("space", "time"):
        if scope in batch:
            nb = batch[scope]
            nb_mask = nb["mask"]
            B_nb, K, F_nb = nb_mask.shape

            # Random masking
            n_mask_count = max(1, int(F_nb * mask_ratio))  # number of features to hide
            rand_idx = torch.rand(B_nb, K, F_nb, device=device).argsort(dim=2)  # idxs to hide
            nb_rand = torch.zeros(B_nb, K, F_nb, dtype=torch.bool, device=device)  # init bool array
            nb_rand.scatter_(2, rand_idx[:, :, :n_mask_count], True)  # hide features
            batch[scope] = {**nb, "mask": nb_mask & ~nb_rand}  # store

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
        self.patience = patience
        self.min_delta_ratio = min_delta_ratio
        self.best_loss = np.inf
        self.counter = 0
        self.best_epoch = 0
        self.epoch = 0

    def __call__(self, val_loss: float) -> bool:
        delta = max(1e-8, abs(val_loss) * self.min_delta_ratio)
        if val_loss < self.best_loss - delta:
            self.best_loss = val_loss
            self.best_epoch = self.epoch
            self.counter = 0
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
        grad_clip_norm: float = 1.0,
        verbose: bool = True,
) -> float:
    """ Run one training epoch; return mean loss per observed value.

    Batches with a non-finite loss are skipped (no optimiser step) so a single
    diverging batch cannot poison the model weights. Returns ``nan`` when every
    batch was skipped, which aborts training in :func:`fit`.
    """
    model.train()
    total, n, n_skipped = 0.0, 0, 0
    for batch in tqdm(loader, desc="  train", leave=False, disable=not _tqdm_ok(verbose)):
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
        if not torch.isfinite(loss):
            n_skipped += 1
            continue
        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total += loss.item() * loss_mask.sum().item()
        n += loss_mask.sum().item()
    if n_skipped:
        logger.warning("Skipped %d batch(es) with non-finite loss this epoch.", n_skipped)
    return total / n if n else float("nan")


@torch.no_grad()
def _eval_epoch(
        model, loader: DataLoader, loss_fn: Callable,
        mask_ratio: float, device: torch.device,
        cylindrical_mask_size: float = 0.0,
        cylindrical_mask_prob: float = 0.0,
        verbose: bool = True,
) -> float:
    """Evaluate on ``loader`` without gradient updates; return mean loss."""
    model.eval()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="    val", leave=False, disable=not _tqdm_ok(verbose)):
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
        n += loss_mask.sum().item()
    return total / max(n, 1)


@torch.no_grad()
def reconstruct(model, loader: DataLoader, device: torch.device, verbose: bool = True) -> Tensor:
    """
    Run full reconstruction without masking; return predicted means.

    Parameters
    ----------
    model : MaSTNeT
    loader : DataLoader
    device : torch.device
    verbose : bool, default=True
        Show a progress bar over batches.

    Returns
    -------
    Tensor of shape (n, n_features)
        Predicted means for every point in the loader.
    """
    model.eval()
    preds = []
    logger.info("Reconstructing %d batches...", len(loader))
    t0 = time()
    for batch in tqdm(loader, desc="Reconstructing", leave=False, disable=not _tqdm_ok(verbose)):
        batch = _to_device(batch, device)
        pred_mean, _ = model(batch)
        preds.append(pred_mean.cpu())
    logger.info("Reconstruction finished in %.1fs.", time() - t0)
    return torch.cat(preds)


def reconstruct_with_uncertainty(
        model, loader: DataLoader, device: torch.device, n_inference: int = 10,
        verbose: bool = True, return_realizations: bool = False,
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
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
    verbose : bool, default=True
        Show a progress bar over inference passes.
    return_realizations : bool, default=False
        If True, also return the individual per-pass mean predictions (before
        aggregation), e.g. for downstream methods that need the distribution
        across draws rather than its first two moments.

    Returns
    -------
    mean : Tensor of shape (n, n_features)
        Mean prediction across all passes.
    aleatoric_std : Tensor of shape (n, n_features)
        Per-point aleatoric (data) uncertainty: ``sqrt(mean(exp(logvar)))``.
    epistemic_std : Tensor of shape (n, n_features)
        Per-point epistemic (model) uncertainty: std of means across passes.
    realizations : Tensor of shape (n_inference, n, n_features), optional
        Only returned when ``return_realizations=True``: the individual
        per-pass mean predictions.
    """
    model.train()  # activates dropout
    all_means: list[Tensor] = []
    all_logvars: list[Tensor] = []

    with torch.no_grad():
        for i in tqdm(range(n_inference), desc="MC inference", disable=not _tqdm_ok(verbose)):
            t0 = time()
            batch_means: list[Tensor] = []
            batch_logvars: list[Tensor] = []
            for batch in loader:
                batch = _to_device(batch, device)
                pred_mean, pred_logvar = model(batch)
                batch_means.append(pred_mean.cpu())
                batch_logvars.append(pred_logvar.cpu())
            all_means.append(torch.cat(batch_means))
            all_logvars.append(torch.cat(batch_logvars))
            logger.info("MC inference pass %d/%d finished in %.1fs.",
                        i + 1, n_inference, time() - t0)

    model.eval()

    stacked_means = torch.stack(all_means)  # (n_inference, n, F)
    stacked_logvars = torch.stack(all_logvars)  # (n_inference, n, F)

    mean = stacked_means.mean(0)
    epistemic_std = stacked_means.std(0, correction=0)
    aleatoric_std = (stacked_logvars.mean(0) * 0.5).exp()  # sqrt(exp(logvar))

    if return_realizations:
        return mean, aleatoric_std, epistemic_std, stacked_means
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
        grad_clip_norm: float = 1.0,
        verbose: bool = True,
) -> None:
    """Train ``model`` in-place with early stopping; restores best weights.

    Training aborts as soon as the train or validation loss becomes non-finite
    (instead of burning the remaining patience on NaN epochs); the best weights
    seen so far are restored either way.

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
    grad_clip_norm : float, default=1.0
        Maximum gradient norm for :func:`torch.nn.utils.clip_grad_norm_`
        applied before each optimiser step. Set to ``0`` to disable.
    verbose : bool, default=True
        Show tqdm progress bars and per-epoch log lines. Set to ``False``
        to suppress all console output (useful inside grid searches).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = _build_loss(loss_name)
    stopper = EarlyStopping(patience=patience)
    best_state = deepcopy(model.state_dict())
    val_history: list[float] = []

    for epoch in tqdm(range(max_epochs), desc="Training", unit="epoch", disable=not _tqdm_ok(verbose)):
        train_loss = _train_epoch(
            model, train_loader, optimizer, loss_fn, mask_ratio, device,
            cylindrical_mask_size=cylindrical_mask_size,
            cylindrical_mask_prob=cylindrical_mask_prob,
            grad_clip_norm=grad_clip_norm,
            verbose=verbose,
        )
        val_loss = _eval_epoch(
            model, val_loader, loss_fn, mask_ratio, device,
            cylindrical_mask_size=cylindrical_mask_size,
            cylindrical_mask_prob=cylindrical_mask_prob,
            verbose=verbose,
        )

        if val_loss < stopper.best_loss:
            # A new best that beats the previous one by far more than the
            # typical epoch-to-epoch change is often a pre-divergence artifact
            # rather than a genuinely better model (see TODO item on hetero
            # instability); warn so such runs can be inspected.
            improvement = stopper.best_loss - val_loss
            if len(val_history) >= 5:
                median_step = float(np.median(np.abs(np.diff(val_history))))
                if median_step > 0 and improvement > 5 * median_step:
                    logger.warning(
                        "Val loss improved by %.4f at epoch %d, over 5x the median "
                        "epoch-to-epoch change (%.4f). This new best may be an "
                        "instability artifact rather than a better model.",
                        improvement, epoch, median_step,
                    )
            best_state = deepcopy(model.state_dict())
        should_stop = stopper(val_loss)
        val_history.append(val_loss)

        msg = (
            f"[{strftime('%d-%m-%Y %H:%M:%S')}] Epoch {epoch:03d}/{max_epochs}"
            f" | train={train_loss:.4f} | val={val_loss:.4f}"
            f" | patience {stopper.counter}/{stopper.patience}"
        )
        logger.info(msg)
        if _tqdm_ok(verbose):
            tqdm.write(msg)

        if not (np.isfinite(train_loss) and np.isfinite(val_loss)):
            abort_msg = (
                f"[{strftime('%d-%m-%Y %H:%M:%S')}] Non-finite loss at epoch {epoch} "
                f"(train={train_loss}, val={val_loss}); aborting training and "
                f"restoring best weights (epoch {stopper.best_epoch})."
            )
            logger.warning(abort_msg)
            if _tqdm_ok(verbose):
                tqdm.write(abort_msg)
            break

        if should_stop:
            stop_msg = f"[{strftime('%d-%m-%Y %H:%M:%S')}] Early stopping at epoch {epoch} (best={stopper.best_epoch})"
            logger.info(stop_msg)
            if _tqdm_ok(verbose):
                tqdm.write(stop_msg)
            break

    model.load_state_dict(best_state)
