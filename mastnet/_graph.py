"""k-NN graph construction and neighbour dataset for MaSTNeT."""

from typing import Any, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _knn_indices(coords: np.ndarray, k: int) -> torch.Tensor:
    """Return (n, k) neighbour indices excluding self."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    idx = nbrs.kneighbors(return_distance=False)
    return torch.tensor(idx[:, 1:], dtype=torch.long)  # drop self


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


class NeighbourDataset(Dataset):
    """
    PyTorch Dataset returning a query point and its spatial / temporal neighbours.

    Parameters
    ----------
    coords : Tensor (n, coord_dim)  — preprocessed (scaled) coordinates
    values : Tensor (n, n_features) — preprocessed features, NaN = missing
    graph  : dict {"space": (n, k), "time": (n, k)}  — precomputed neighbour indices
    query_indices : array-like or None  — subset of rows to use as queries;
                    None uses all rows
    """

    def __init__(self, coords: Tensor, values: Tensor,
                 graph: Dict[str, Tensor],
                 query_indices=None):
        self.coords = coords
        self.values = values
        self.mask = ~torch.isnan(values)
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
            nb = n_idx[q]
            out[scope] = {
                "features": self.values[nb],
                "mask": self.mask[nb],
                "coords": self.coords[nb],
                "rel_positions": self.coords[nb] - self.coords[q],
            }

        return out
