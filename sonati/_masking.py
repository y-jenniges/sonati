"""Masking strategies for SONATI training.

Each strategy is a small, composable, frozen dataclass implementing
``MaskStrategy``'s ``mask()`` method.

Custom strategies can be added by subclassing :class:`MaskStrategy`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
from torch import Tensor


def _ratio_to_count(ratio: float, n: int) -> int:
    """Convert a hide ratio to a count: ``0`` disables entirely, any positive ratio hides >=1."""
    return 0 if ratio <= 0.0 else max(1, int(n * ratio))


class MaskStrategy(ABC):
    """Base class for masking strategies."""

    @abstractmethod
    def mask(self, n_points: int, n_features: int, coords: Tensor | None,
             donor_pool_mask: Tensor | None, device: torch.device,
             generator: torch.Generator | None) -> Tensor:
        """
        Return a hide-mask for this strategy.

        Parameters
        ----------
        n_points : int
        n_features : int
        coords : Tensor, shape (n_points, >=3), optional
            Cartesian (x, y, z) coordinates. Only required by strategies
            that need geometry (e.g. :class:`CylindricalMask`).
        donor_pool_mask : Tensor, shape (N_pool, n_features), dtype bool, optional
            Observation mask of the population to draw donors from. Only
            required by strategies that need one (e.g. :class:`CopyMask`).
        device : torch.device
        generator : torch.Generator, optional
            Draw from this generator (must live on ``device``) instead of
            the global RNG.

        Returns
        -------
        Tensor of shape (n_points, n_features), dtype bool
        """
        ...


@dataclass(frozen=True)
class RandomMask(MaskStrategy):
    """
    Hide a fixed fraction of features per point/sample, independently per point.

    Parameters
    ----------
    ratio : float, default=0.5
        Fraction of features to hide per point. Any positive ratio
        still hides at least one feature.
    """
    ratio: float = 0.5

    def mask(self, n_points, n_features, coords, donor_pool_mask, device,
             generator) -> Tensor:
        n_mask = _ratio_to_count(self.ratio, n_features)
        idx = torch.rand(n_points, n_features, device=device, generator=generator).argsort(dim=1)
        mask = torch.zeros(n_points, n_features, dtype=torch.bool, device=device)
        mask.scatter_(1, idx[:, :n_mask], True)
        return mask


@dataclass(frozen=True)
class CylindricalMask(MaskStrategy):
    """
    Block-mask all features of every point inside a spatial cylinder (across all times).

    The "cylinder" is defined by a circle of ``radius`` in the Cartesian
    (x, y, z) space of the point coordinates.

    Applied with probability ``prob`` per call.

    Parameters
    ----------
    radius : float, default=0.1
        Cylinder radius in Cartesian coordinate space.
    prob : float, default=1.0
        Probability of applying the mask on any given call.
    """
    radius: float = 0.1
    prob: float = 1.0

    def mask(self, n_points, n_features, coords, donor_pool_mask, device,
             generator) -> Tensor:
        if coords is None:
            raise ValueError("CylindricalMask requires coordinates.")

        # Skip the block mask with probability 1 - prob
        if torch.rand(1, device=device, generator=generator).item() > self.prob:
            return torch.zeros(n_points, n_features, dtype=torch.bool, device=device)

        # Pick a random centre point and mask every point within `radius` of it
        idx = torch.randint(0, n_points, (1,), device=device, generator=generator)
        center = coords[idx]  # (1, 3)
        dist = torch.norm(coords - center, dim=1)  # (n_points,)
        mask_block = dist < self.radius  # (n_points,)
        return mask_block.unsqueeze(1).expand(n_points, n_features)


@dataclass(frozen=True)
class CopyMask(MaskStrategy):
    """
    Per point, hide features that are missing in a randomly drawn donor
    sample from the training data, following Gorla A, Wang R, Liu Z, An U,
    Sankararaman S. CACTI: Leveraging Copy Masking and Contextual Information
    to Improve Tabular Data Imputation. Proc Mach Learn Res.
    2025 Jul;267:20187-20225. PMID: 41959945; PMCID: PMC13060770., code on
    github.com/sriramlab/CACTI, though no minimum number of features must stay
    observed.

    Applied with probability ``prob`` per point and
    requires a ``donor_pool_mask`` (see :func:`mask_batch`).

    Parameters
    ----------
    prob : float, default=1.0
        Probability of applying the copy mask to each point.
    """
    prob: float = 1.0

    def mask(self, n_points, n_features, coords, donor_pool_mask, device,
             generator) -> Tensor:
        if donor_pool_mask is None:
            raise ValueError("CopyMask requires a donor_pool_mask (see mask_batch()).")

        apply_mask = torch.rand(n_points, device=device, generator=generator) < self.prob

        # Draw a donor sample
        donor_idx = torch.randint(0, donor_pool_mask.shape[0], (n_points,), device=device, generator=generator)
        donor_missing = ~donor_pool_mask[donor_idx]  # features missing in the donor sample

        # Mask replicates the missingness pattern of the donor sample
        return donor_missing & apply_mask.unsqueeze(1)


@dataclass(frozen=True)
class PerFeatureMask(MaskStrategy):
    """
    Hide a fixed fraction of points for every feature independently (column-wise).

    Parameters
    ----------
    ratio : float, default=0.5
        Fraction of points to hide, independently per feature column. Any positive
        ratio still hides at least one point per feature.
    """
    ratio: float = 0.5

    def mask(self, n_points, n_features, coords, donor_pool_mask, device,
             generator) -> Tensor:
        n_mask = _ratio_to_count(self.ratio, n_points)
        mask = torch.zeros(n_points, n_features, dtype=torch.bool, device=device)
        if n_mask == 0:
            return mask

        # For each feature column independently, pick n_mask points to hide
        rand_idx = torch.rand(n_features, n_points, device=device, generator=generator).argsort(dim=1)
        for f in range(n_features):
            mask[rand_idx[f, :n_mask], f] = True
        return mask


def validate_masking(masking: list) -> None:
    """Raise ``ValueError`` if any entry of ``masking`` is not a recognised strategy."""
    for spec in masking:
        if not isinstance(spec, MaskStrategy):
            raise ValueError(
                f"Unknown masking strategy {spec!r}. Expected a MaskStrategy instance "
                f"(e.g. RandomMask, CylindricalMask, CopyMask, PerFeatureMask)."
            )


def apply_masking_strategies(masking: list, n_points: int, n_features: int,
                             coords: Tensor | None = None,
                             donor_pool_mask: Tensor | None = None,
                             device: torch.device = torch.device("cpu"),
                             generator: torch.Generator | None = None) -> Tensor:
    """
    Combine every strategy in ``masking`` into one hide-mask.

    Parameters
    ----------
    masking : list of MaskStrategy
        Strategies to apply, in order (e.g. RandomMask / CylindricalMask /
        CopyMask / PerFeatureMask, or a custom subclass). An empty list
        hides nothing.
    n_points : int
    n_features : int
    coords : Tensor, shape (n_points, >=3), optional
        Cartesian (x, y, z) coordinates. Required by :class:`CylindricalMask`.
    donor_pool_mask : Tensor, shape (N_pool, n_features), dtype bool, optional
        Observation mask of the population to draw copy-mask donors from
        (the full training set). Required by :class:`CopyMask`.
    device : torch.device
    generator : torch.Generator, optional
        Draw from this generator (must live on ``device``) instead of the
        global RNG.

    Returns
    -------
    Tensor of shape (n_points, n_features), dtype bool
    """
    validate_masking(masking)
    hide = torch.zeros(n_points, n_features, dtype=torch.bool, device=device)
    for spec in masking:
        hide = hide | spec.mask(n_points, n_features, coords, donor_pool_mask, device, generator)
    return hide


def mask_batch(batch: dict, masking: list, device: torch.device,
               donor_pool_mask: Tensor | None = None,
               generator: torch.Generator | None = None) -> tuple[dict, Tensor]:
    """
    Apply ``masking`` to both the query token and every neighbour scope in ``batch``.

    Parameters
    ----------
    batch : dict
        Batch dict from :class:`sonati._graph.NeighbourDataset`.
    masking : list of MaskStrategy
        Strategies to apply, see :func:`apply_masking_strategies`.
    device : torch.device
    donor_pool_mask : Tensor, shape (N_pool, F), dtype bool, optional
        Required when ``masking`` includes a :class:`CopyMask`.
    generator : torch.Generator, optional
        Draw from this generator (must live on ``device``) instead of the
        global RNG. Passed by :func:`sonati._training.fit` for validation
        batches so the same mask is redrawn every epoch (instead of a fresh
        random one). Training keeps using the global RNG.

    Returns
    -------
    batch : dict
        Shallow copy with ``query_mask`` and neighbour masks reduced.
    loss_mask : Tensor of shape (B, F)
        Boolean mask selecting query positions to compute reconstruction loss on.
    """
    q_mask = batch["query_mask"]  # (B, F) bool - truly observed
    B, F = q_mask.shape
    q_coords = batch["query_coords"][:, :3]  # Cartesian (x, y, z); no depth, no time

    # Apply combined masking strategies
    hide = apply_masking_strategies(
        masking, B, F, coords=q_coords, donor_pool_mask=donor_pool_mask,
        device=device, generator=generator,
    )
    input_mask = q_mask & ~hide  # model sees these
    loss_mask = q_mask & hide  # reconstruct these

    # Same strategies applied to neighbour tokens
    batch = {**batch, "query_mask": input_mask}  # shallow copy
    for scope in ("space", "time"):
        if scope in batch:
            nb = batch[scope]
            nb_mask = nb["mask"]  # (B, K, F)
            B_nb, K, F_nb = nb_mask.shape
            nb_coords = nb["coords"][:, :, :3].reshape(B_nb * K, -1)

            # Apply combined masking strategies
            nb_hide = apply_masking_strategies(
                masking, B_nb * K, F_nb, coords=nb_coords, donor_pool_mask=donor_pool_mask,
                device=device, generator=generator,
            ).reshape(B_nb, K, F_nb)
            batch[scope] = {**nb, "mask": nb_mask & ~nb_hide}

    return batch, loss_mask
