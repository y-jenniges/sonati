"""k-NN graph construction and neighbour dataset for MaSTNeT."""

from typing import Any, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch import Tensor
from torch.utils.data import Dataset

from mastnet._preprocessing import preprocess


def _knn_indices(coords: np.ndarray, k: int) -> torch.Tensor:
    """Return (n, k) neighbour indices excluding self.

    Stored as int32 rather than the default int64 to reduce memory.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    idx = nbrs.kneighbors(return_distance=False)
    return torch.tensor(idx[:, 1:], dtype=torch.int32)  # drop self


def build_graph(
        coords_scaled: Tensor,
        n_neighbours_space: int,
        n_neighbours_time: int,
) -> Dict[str, Tensor]:
    """
    Build separate spatial and temporal k-NN graphs.

    Spatial neighbours are found in the (x, y, z, depth) subspace;
    temporal neighbours in the time subspace (last one or two columns,
    depending on whether cyclic encoding was used).

    Parameters
    ----------
    coords_scaled : Tensor of shape (n, coord_dim)
        Preprocessed coordinates from :func:`mastnet._preprocessing.preprocess`.
        Columns 0–3 are (x, y, z, depth); columns 4+ are scaled time.
    n_neighbours_space : int
        Number of nearest neighbours per query in the spatial graph.
    n_neighbours_time : int
        Number of nearest neighbours per query in the temporal graph.

    Returns
    -------
    dict with keys ``'space'`` and ``'time'``, each a LongTensor of shape (n, k).
    """
    c = coords_scaled.cpu().numpy()
    space_idx = _knn_indices(c[:, :4], n_neighbours_space)  # x, y, z, depth
    time_idx = _knn_indices(c[:, 4:], n_neighbours_time)  # scaled time (1 or 2 cols)
    return {"space": space_idx, "time": time_idx}


def precompute_graph(
        coords,
        n_neighbours_space: int,
        n_neighbours_time: int,
        cyclic_time: bool = False,
) -> Dict[str, Tensor]:
    """
    Build a k-NN graph from raw coordinates, e.g. for reuse across tuning trials.

    The graph depends only on the coordinates and the neighbour counts -> during
    hyperparameter tuning it can be built once (with the largest ``k`` under
    consideration) and passed to :meth:`mastnet.MaSTNetImputer.fit` via its
    ``graph`` argument, which slices the first
    ``n_neighbours_space`` / ``n_neighbours_time`` columns per trial.

    Note: coordinate scalers are fitted on *all* rows here, whereas
    :meth:`~mastnet.MaSTNetImputer.fit` fits them on the training rows only.
    MinMax scaling is monotonic per dimension, so this can at most swap
    marginal neighbours and is negligible in practice.

    Parameters
    ----------
    coords : array-like of shape (n, 4)
        Raw coordinates as ``[lat (°), lon (°), depth (m), time]``.
    n_neighbours_space : int
        Number of spatial neighbours to store per query (choose the maximum
        ``n_neighbours_space`` you intend to tune over).
    n_neighbours_time : int
        Number of temporal neighbours to store per query.
    cyclic_time : bool, default=False
        Must match the ``cyclic_time`` setting of the imputer the graph is
        used with.

    Returns
    -------
    dict with keys ``'space'`` and ``'time'``, each a LongTensor of shape (n, k).
    """
    coords_t = torch.as_tensor(np.asarray(coords, dtype=float), dtype=torch.float32)
    coords_scaled, _, _ = preprocess(
        coords_t, torch.zeros((len(coords_t), 0)), [], cyclic_time,
    )
    return build_graph(coords_scaled, n_neighbours_space, n_neighbours_time)


class NeighbourDataset(Dataset):
    """
    PyTorch Dataset returning a query point and its spatial / temporal neighbours.

    Parameters
    ----------
    coords : Tensor (n, coord_dim) - preprocessed (scaled) coordinates
    values : Tensor (n, n_features) - preprocessed features, NaN = missing
    graph  : dict {"space": (n, k), "time": (n, k)} - precomputed neighbour indices
    query_indices : array-like or None - subset of rows to use as queries;
                    None uses all rows
    neighbour_values : Tensor (n, n_features) or None - values to expose as
                    *neighbour* features (query features/targets always come
                    from ``values``). Pass a copy with e.g. validation rows set
                    to NaN so their observed values stay hidden from other
                    queries' neighbourhoods; None exposes ``values``.
    """

    def __init__(self, coords: Tensor, values: Tensor,
                 graph: Dict[str, Tensor],
                 query_indices=None,
                 neighbour_values: Tensor | None = None):
        self.coords = coords
        self.values = values
        self.mask = ~torch.isnan(values)
        self.neighbour_values = values if neighbour_values is None else neighbour_values
        self.neighbour_mask = ~torch.isnan(self.neighbour_values)
        self.graph = graph
        self.query_indices = (
            torch.arange(len(values)) if query_indices is None
            else torch.as_tensor(query_indices, dtype=torch.long)
        )  # ensure tensor type

    def __len__(self) -> int:
        return len(self.query_indices)

    def __getitem__(self, idx) -> Dict[str, Any]:
        # Define query features/mask/coords
        q = self.query_indices[idx]
        out: Dict[str, Any] = {
            "query_features": self.values[q],
            "query_mask": self.mask[q],
            "query_coords": self.coords[q],
        }

        # Iterate over scopes
        for scope, n_idx in self.graph.items():
            # Define neighbour features/mask/coords/rel_positions
            # (.long(): advanced indexing requires int64; n_idx itself is
            # stored as int32 to keep the full (n, k) graph small)
            nb = n_idx[q].long()
            out[scope] = {
                "features": self.neighbour_values[nb],
                "mask": self.neighbour_mask[nb],
                "coords": self.coords[nb],
                "rel_positions": self.coords[nb] - self.coords[q],
            }

        return out
