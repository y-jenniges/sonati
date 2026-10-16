"""Training loop, losses and early stopping for SONATI."""

from copy import deepcopy
from typing import Callable
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import sys
from time import strftime, time

from sonati._masking import mask_batch
from sonati._model import LOGVAR_CLAMP

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
                         target: Tensor, mask: Tensor, var_floor: float) -> Tensor:
    """Gaussian NLL with log-variance (Kendall & Gal 2017)."""
    # Variance floored to match the model head's logvar clamp
    var = pred_logvar[mask].exp().clamp(min=var_floor)
    return (0.5 * (pred_logvar[mask] + (target[mask] - pred_mean[mask]) ** 2 / var)).mean()


def _beta_nll(pred_mean: Tensor, pred_logvar: Tensor,
              target: Tensor, mask: Tensor, var_floor: float, beta: float = 0.5) -> Tensor:
    """Beta-NLL loss. It multiplies each point's Gaussian heteroscedastic NLL by its predicted variance^β.

    Beta is the weighing of the exponent and in [0, 1].
    ``beta=0`` gives the plain heteroscedastic NLL,
    ``beta=1`` results in an MSE loss, while variance is still learned.

    Reference:
    Seitzer, M., Tavakoli, A., Antic, D., & Martius, G. (2022). On the
    pitfalls of heteroscedastic uncertainty estimation with probabilistic
    neural networks. International Conference on Learning Representations
    (ICLR). https://openreview.net/forum?id=aPOpXlnV1T
    """
    logvar = pred_logvar[mask]
    # Variance floored to match the model head's logvar clamp
    var = logvar.exp().clamp(min=var_floor)
    nll = 0.5 * (logvar + (target[mask] - pred_mean[mask]) ** 2 / var)
    return (var.detach() ** beta * nll).mean()


def _build_loss(loss_name: str, loss_kwargs: dict | None = None,
                logvar_clamp: float = LOGVAR_CLAMP) -> Callable:
    """
    Return the loss callable corresponding to ``loss_name``.

    ``loss_kwargs`` are loss-specific options (e.g. ``{"beta": 0.5}`` for
    ``'beta_nll'``); unexpected keys raise a ``TypeError``.
    """
    # Loss-specific function calls
    loss_kwargs = dict(loss_kwargs or {})
    var_floor = math.exp(-logvar_clamp)
    if loss_name == "mse":
        loss = lambda mean, logvar, target, mask: _masked_mse(mean, target, mask)
        allowed: set[str] = set()
    elif loss_name == "hetero":
        loss = lambda mean, logvar, target, mask: _heteroscedastic_nll(mean, logvar, target, mask, var_floor)
        allowed = set()
    elif loss_name == "beta_nll":
        beta = loss_kwargs.get("beta", 0.5)
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must lie in [0, 1], got {beta}.")
        loss = lambda mean, logvar, target, mask: _beta_nll(mean, logvar, target, mask, var_floor, beta=beta)
        allowed = {"beta"}
    else:
        raise ValueError(f"Unknown loss_name '{loss_name}'. Choose 'mse', 'hetero' or 'beta_nll'.")

    # Check if unknown loss kwargs were passed
    unexpected = set(loss_kwargs) - allowed
    if unexpected:
        raise TypeError(f"loss_name '{loss_name}' got unexpected loss_kwargs {sorted(unexpected)}; "
                        f"allowed: {sorted(allowed) or 'none'}.")
    return loss


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

    Attributes
    ----------
    is_best : bool
        Whether the most recent ``__call__`` counted as a new best (cleared
        ``best_loss`` by more than the delta margin). Callers that snapshot
        model weights should gate on this flag rather than re-deriving their
        own "is this better" comparison, so the snapshot always agrees with
        ``best_loss``/``best_epoch``.
    """

    def __init__(self, patience: int = 10, min_delta_ratio: float = 1e-4):
        self.patience = patience
        self.min_delta_ratio = min_delta_ratio
        self.best_loss = np.inf
        self.counter = 0
        self.best_epoch = 0
        self.epoch = 0
        self.is_best = False

    def __call__(self, val_loss: float) -> bool:
        # Delta relative to the current best loss
        first_call = math.isinf(self.best_loss)
        delta = 0.0 if first_call else max(1e-8, abs(self.best_loss) * self.min_delta_ratio)

        # Check if current epoch is the best (by delta)
        self.is_best = val_loss < self.best_loss - delta

        # Update state if current epoch is best
        if self.is_best:
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
        masking: list, device: torch.device,
        donor_pool_mask: Tensor | None = None,
        grad_clip_norm: float = 1.0,
        verbose: bool = True,
) -> float:
    """
    Run one training epoch and return mean loss per observed value.

    Batches with a non-finite loss are skipped (no optimiser step) so a single
    diverging batch cannot poison the model weights. Returns ``nan`` when every
    batch was skipped, which aborts training in :func:`fit`.
    """
    model.train()
    total, n, n_skipped = 0.0, 0, 0
    for batch in tqdm(loader, desc="  train", leave=False, disable=not _tqdm_ok(verbose)):
        batch = _to_device(batch, device)

        # Apply masking
        batch, loss_mask = mask_batch(batch, masking, device, donor_pool_mask=donor_pool_mask)
        if not loss_mask.any():
            continue

        # Forward pass and loss on the masked positions only
        pred_mean, pred_logvar = model(batch)
        target = batch["query_features"]
        loss = loss_fn(pred_mean, pred_logvar, target, loss_mask)
        if not torch.isfinite(loss):
            n_skipped += 1
            continue

        # Backward pass with optional gradient clipping
        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        # Aggregate loss
        total += loss.item() * loss_mask.sum().item()
        n += loss_mask.sum().item()
    if n_skipped:
        logger.warning("Skipped %d batch(es) with non-finite loss this epoch.", n_skipped)
    return total / n if n else float("nan")


@torch.no_grad()
def _eval_epoch(
        model, loader: DataLoader, loss_fn: Callable,
        masking: list, device: torch.device,
        donor_pool_mask: Tensor | None = None,
        verbose: bool = True,
        generator: torch.Generator | None = None,
) -> float:
    """Evaluate on ``loader`` without gradient updates and return mean loss."""
    model.eval()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="    val", leave=False, disable=not _tqdm_ok(verbose)):
        batch = _to_device(batch, device)
        batch, loss_mask = mask_batch(batch, masking, device, donor_pool_mask=donor_pool_mask, generator=generator)
        if not loss_mask.any():
            continue

        # Forward pass only, no optimiser step
        pred_mean, pred_logvar = model(batch)
        target = batch["query_features"]
        loss = loss_fn(pred_mean, pred_logvar, target, loss_mask)
        total += loss.item() * loss_mask.sum().item()
        n += loss_mask.sum().item()
    return total / max(n, 1)


@torch.no_grad()
def reconstruct(model, loader: DataLoader, device: torch.device, verbose: bool = True) -> Tensor:
    """
    Run full reconstruction without masking and return predicted means.

    Parameters
    ----------
    model : SONATI
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
        verbose: bool = True, return_realisations: bool = False,
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    MC Dropout inference to estimate both aleatoric and epistemic uncertainty.

    The model is kept in *train* mode so dropout layers remain active. Each of
    the ``n_inference`` forward passes produces a slightly different prediction;
    their spread is the epistemic (model) uncertainty.

    Parameters
    ----------
    model : SONATI
    loader : DataLoader
    device : torch.device
    n_inference : int, default=10
        Number of stochastic forward passes.
    verbose : bool, default=True
        Show a progress bar over inference passes.
    return_realisations : bool, default=False
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
    realisations : Tensor of shape (n_inference, n, n_features), optional
        Only returned when ``return_realisations=True``: the individual
        per-pass mean predictions.
    """
    model.train()  # activate dropout

    # Online mean/variance across passes (Welford 1962, less memory-intense)
    final_mean: Tensor | None = None
    m2: Tensor | None = None
    sum_exp_logvar: Tensor | None = None
    realisations: Tensor | None = None

    with torch.no_grad():
        # Iterate over inferences
        for i in tqdm(range(n_inference), desc="MC inference", disable=not _tqdm_ok(verbose)):
            t0 = time()
            batch_means: list[Tensor] = []
            batch_logvars: list[Tensor] = []

            # Iterate over batches and get mean and variances
            for batch in loader:
                batch = _to_device(batch, device)
                pred_mean, pred_logvar = model(batch)
                batch_means.append(pred_mean.cpu())
                batch_logvars.append(pred_logvar.cpu())
            pass_mean = torch.cat(batch_means)
            pass_logvar = torch.cat(batch_logvars)

            if final_mean is None:
                # First iteration
                final_mean = pass_mean.clone()
                m2 = torch.zeros_like(pass_mean)  # sum of squared deviations
                sum_exp_logvar = pass_logvar.exp()
            else:
                # Aggregate later iterations
                delta = pass_mean - final_mean
                final_mean += delta / (i + 1)  # running mean
                m2 += delta * (pass_mean - final_mean)  # Welford update
                sum_exp_logvar += pass_logvar.exp()

            # Collect realisations if requested
            if return_realisations:
                if realisations is None:
                    realisations = pass_mean.new_empty((n_inference, *pass_mean.shape))
                realisations[i] = pass_mean

            logger.info("MC inference pass %d/%d finished in %.1fs.",
                        i + 1, n_inference, time() - t0)

    model.eval()  # deactivate dropout

    # Compute std from variances
    epistemic_std = (m2 / n_inference).sqrt()
    aleatoric_std = (sum_exp_logvar / n_inference).sqrt()  # sqrt(mean(exp(logvar)))

    if return_realisations:
        return final_mean, aleatoric_std, epistemic_std, realisations
    return final_mean, aleatoric_std, epistemic_std


def fit(
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int,
        lr: float,
        masking: list,
        patience: int,
        device: torch.device,
        loss_name: str = "hetero",
        donor_pool_mask: Tensor | None = None,
        grad_clip_norm: float = 1.0,
        verbose: bool = True,
        seed: int = 0,
        checkpoint_path: str | Path | None = None,
        loss_kwargs: dict | None = None,
) -> dict:
    """Train ``model`` in-place with early stopping; restores best weights.

    Interrupted training can be resumed if ``checkpoint_path`` is given. In that
    case, the full training state (model, optimiser, early-stopper, best weights, loss history)
    is written there after every epoch.
    Note: The train-loader shuffle order after a resume differs from an uninterrupted run.

    Training aborts as soon as the train or validation loss becomes non-finite.

    Parameters
    ----------
    model : SONATI
    train_loader : DataLoader
    val_loader : DataLoader
    max_epochs : int
    lr : float
        Adam learning rate.
    masking : list of RandomMask / CylindricalMask / CopyMask / PerFeatureMask
        Masking strategies applied during training and validation to both the
        query token and every neighbour scope.
    patience : int
        Early-stopping patience in epochs.
    device : torch.device
    loss_name : str, optional
        ``'hetero'`` for heteroscedastic Gaussian NLL, ``'beta_nll'`` for the
        beta-NLL of Seitzer et al. (2022), or ``'mse'`` for masked MSE.
        Default is ``'hetero'``.
    donor_pool_mask : Tensor, shape (N_pool, F), dtype bool, optional
        Population to draw copy-mask donors from (the full training set).
        Required when ``masking`` includes a :class:`sonati.CopyMask`.
    grad_clip_norm : float, default=1.0
        Maximum gradient norm for :func:`torch.nn.utils.clip_grad_norm_`
        applied before each optimiser step. Set to ``0`` to disable.
    verbose : bool, default=True
        Show tqdm progress bars and per-epoch log lines. Set to ``False``
        to suppress all console output (useful inside grid searches).
    seed : int, default=0
        Seeds a dedicated generator for validation masking, reset before
        every epoch so the *same* positions are hidden each time.
    checkpoint_path : str or None, default None
        If given, the training state is stored there so that training can be
        interrupted and resumed.
    loss_kwargs : dict or None, default None
        Extra args for the loss, e.g. ``'beta_nll'`` accepts
        ``{"beta": float}``.

    Notes
    -----
    With ``loss_name='beta_nll'`` the training loss is the beta-NLL, but validation, early stopping and the
    best-weights selection use the plain heteroscedastic NLL (Seitzer et al., 2022, Sec. 4.1).

    Returns
    -------
    history : dict
        ``{"train": {epoch: loss, ...}, "val": {epoch: loss, ...}}`` with one
        entry per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = _build_loss(loss_name, loss_kwargs, logvar_clamp=model.logvar_clamp)
    # beta-NLL loss evaluates model quality plain using NLL loss (Seitzer et al., 2022)
    monitor_name = "hetero" if loss_name == "beta_nll" else loss_name
    val_loss_fn = _build_loss("hetero", logvar_clamp=model.logvar_clamp) if loss_name == "beta_nll" else loss_fn
    val_label = "val_nll" if loss_name == "beta_nll" else "val"
    stopper = EarlyStopping(patience=patience)
    best_state = deepcopy(model.state_dict())
    train_history: list[float] = []
    val_history: list[float] = []
    val_mask_generator = torch.Generator(device=device)

    # Epoch status monitoring
    start_epoch = 0
    finished = False

    # Create the checkpoint directory
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # If a checkpoint path is given, resume from last saved model state
    if checkpoint_path is not None and checkpoint_path.exists():
        # Load checkpoint data
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])  # model
        optimizer.load_state_dict(checkpoint["optimizer"])  # optimiser
        stopper.__dict__.update(checkpoint["stopper"])  # early-stopper
        stopper.patience = patience  # update patience to current setting
        best_state = checkpoint["best_state"]  # best state
        train_history, val_history = checkpoint["train_history"], checkpoint["val_history"]  # history
        start_epoch, finished = len(train_history), checkpoint["finished"]  # epoch status

        # Saved validation metric must match resume validation metric
        saved_monitor = checkpoint.get("val_metric", loss_name)
        if saved_monitor != monitor_name:
            raise ValueError(f"Checkpoint {checkpoint_path} early-stopped on '{saved_monitor}' validation loss, but "
                             f"loss_name='{loss_name}' monitors '{monitor_name}'. Retrain from scratch.")

        logger.info("Resumed from checkpoint %s at epoch %d (best=%d, finished=%s)",
                    checkpoint_path, start_epoch, stopper.best_epoch, finished)

    def _save_checkpoint(is_finished: bool) -> None:
        """Saves model, early-stopper and history states to ``checkpoint_path``."""
        if checkpoint_path is None:
            return
        tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "stopper": dict(stopper.__dict__), "best_state": best_state,
            "train_history": train_history, "val_history": val_history,
            "finished": is_finished, "val_metric": monitor_name,
        }, tmp_path)
        tmp_path.replace(checkpoint_path)  # a kill mid-write keeps the old file

    for epoch in tqdm(range(start_epoch, start_epoch if finished else max_epochs), desc="Training",
                      unit="epoch", disable=not _tqdm_ok(verbose), initial=start_epoch,
                      total=max_epochs):
        # One training epoch, one validation epoch
        train_loss = _train_epoch(
            model, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, masking=masking, device=device,
            donor_pool_mask=donor_pool_mask,
            grad_clip_norm=grad_clip_norm,
            verbose=verbose,
        )
        val_mask_generator.manual_seed(seed)  # same val masks every epoch
        val_loss = _eval_epoch(
            model, loader=val_loader, loss_fn=val_loss_fn, masking=masking, device=device,
            donor_pool_mask=donor_pool_mask,
            verbose=verbose,
            generator=val_mask_generator,
        )

        # Snapshot weights only when stopper agrees this is a genuine best
        prev_best_loss = stopper.best_loss
        should_stop = stopper(val_loss)
        if stopper.is_best:
            # A new best beating the previous one by too much, may need inspection
            improvement = prev_best_loss - val_loss
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
        train_history.append(train_loss)
        val_history.append(val_loss)

        # Log epoch summary
        msg = (
            f"[{strftime('%d-%m-%Y %H:%M:%S')}] Epoch {epoch:03d}/{max_epochs}"
            f" | train={train_loss:.4f} | {val_label}={val_loss:.4f}"
            f" | patience {stopper.counter}/{stopper.patience}"
        )
        logger.info(msg)
        if _tqdm_ok(verbose):
            tqdm.write(msg)

        is_non_finite = not (np.isfinite(train_loss) and np.isfinite(val_loss))  # check if loss is non-finite
        _save_checkpoint(is_finished=is_non_finite or should_stop)  # reaching max_epochs stays resumable

        # Abort on a non-finite loss
        if is_non_finite:
            abort_msg = (
                f"[{strftime('%d-%m-%Y %H:%M:%S')}] Non-finite loss at epoch {epoch} "
                f"(train={train_loss}, val={val_loss}); aborting training and "
                f"restoring best weights (epoch {stopper.best_epoch})."
            )
            logger.warning(abort_msg)
            if _tqdm_ok(verbose):
                tqdm.write(abort_msg)
            break

        # Stop once patience is exhausted
        if should_stop:
            stop_msg = f"[{strftime('%d-%m-%Y %H:%M:%S')}] Early stopping at epoch {epoch} (best={stopper.best_epoch})"
            logger.info(stop_msg)
            if _tqdm_ok(verbose):
                tqdm.write(stop_msg)
            break

    # Restore the best weights
    model.load_state_dict(best_state)
    return {"train": dict(enumerate(train_history)), "val": dict(enumerate(val_history))}
