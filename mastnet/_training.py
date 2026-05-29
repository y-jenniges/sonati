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
    return F.mse_loss(pred[mask], target[mask])


def _heteroscedastic_nll(pred_mean: Tensor, pred_logvar: Tensor,
                         target: Tensor, mask: Tensor) -> Tensor:
    """Gaussian NLL with predicted log-variance (Kendall & Gal 2017)."""
    var = pred_logvar[mask].exp().clamp(min=1e-6)
    return (0.5 * (pred_logvar[mask] + (target[mask] - pred_mean[mask]) ** 2 / var)).mean()


def _build_loss(loss_name: str) -> Callable:
    if loss_name == "mse":
        return lambda mean, logvar, target, mask: _masked_mse(mean, target, mask)
    elif loss_name == "hetero":
        return _heteroscedastic_nll
    else:
        raise ValueError(f"Unknown loss_name '{loss_name}'. Choose 'mse' or 'hetero'.")


# ------------------------------------------------------------------
# Masking
# ------------------------------------------------------------------

def _apply_random_mask(batch: dict, mask_ratio: float, device: torch.device) -> tuple[dict, Tensor]:
    """
    Randomly hide a fraction of the observed query features.

    Returns a modified batch (query_mask reflects what the model sees) and
    a loss_mask (positions to compute reconstruction loss on).
    """
    q_mask = batch["query_mask"]            # (B, F) bool — truly observed
    B, F = q_mask.shape
    rand = torch.rand(B, F, device=device) < mask_ratio
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
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, Tensor):
        return obj.to(device)
    return obj


# ------------------------------------------------------------------
# Early stopping
# ------------------------------------------------------------------

class EarlyStopping:
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

def _train_epoch(model, loader: DataLoader, optimizer, loss_fn: Callable,
                 mask_ratio: float, device: torch.device) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        batch, loss_mask = _apply_random_mask(batch, mask_ratio, device)
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
def _eval_epoch(model, loader: DataLoader, loss_fn: Callable,
                mask_ratio: float, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        batch, loss_mask = _apply_random_mask(batch, mask_ratio, device)
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
    """Run full reconstruction (no masking). Returns predictions (n, F)."""
    model.eval()
    preds = []
    for batch in loader:
        batch = _to_device(batch, device)
        pred_mean, _ = model(batch)
        preds.append(pred_mean.cpu())
    return torch.cat(preds)


def fit(model, train_loader: DataLoader, val_loader: DataLoader,
        max_epochs: int, lr: float, mask_ratio: float,
        patience: int, device: torch.device, loss_name: str = "hetero") -> None:
    """Train ``model`` in-place; restores best weights at the end."""
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn    = _build_loss(loss_name)
    stopper    = EarlyStopping(patience=patience)
    best_state = deepcopy(model.state_dict())

    for epoch in tqdm(range(max_epochs), desc="Training", unit="epoch"):
        train_loss = _train_epoch(model, train_loader, optimizer, loss_fn, mask_ratio, device)
        val_loss   = _eval_epoch(model,   val_loader,            loss_fn, mask_ratio, device)
        tqdm.write(f"Epoch {epoch:03d}: train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < stopper.best_loss:
            best_state = deepcopy(model.state_dict())
        if stopper(val_loss):
            tqdm.write(f"Early stopping at epoch {epoch} (best={stopper.best_epoch})")
            break

    model.load_state_dict(best_state)
