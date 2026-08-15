"""k-NN graph construction and neighbour dataset for MaSTNeT."""

from typing import Any, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch import Tensor
from torch.utils.data import Dataset

from mastnet._preprocessing import preprocess


def _knn_indices(coords: np.ndarray, k: int, n_jobs: int | None = None,
                  query_chunk_size: int = 5_000_000) -> torch.Tensor:
    """
    Return (n, k) neighbour indices excluding self.
    Stored as int32 (not default int64) and queried in chunks
    to reduce memory.

    Parameters
    ----------
    coords : coordinates of the data points
    k : number of neighbours per point, excluding self
    n_jobs : number of parallel jobs for the tree-based spatial search (sklearn
        convention: None means 1, -1 means all available cores). The temporal search
        (:func:`_time_knn_indices`) is sort-based and single-threaded.
    query_chunk_size : rows queried per batch (to reduce peak memory for large datasets).
    """
    # Fit nearest neighbours
    n = len(coords)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=n_jobs).fit(coords)

    # Pre-allocate results array
    result = np.empty((n, k), dtype=np.int32)

    # Iterate over chunks
    for start in range(0, n, query_chunk_size):
        stop = min(start + query_chunk_size, n)
        # Query neighbours
        chunk_idx = nbrs.kneighbors(coords[start:stop], n_neighbors=k + 1, return_distance=False)
        # Fill results array
        result[start:stop] = chunk_idx[:, 1:]

    # Shut down joblib's executor used to compute the neighbourhood graph in parallel
    # (otherwise it could block DataLoader fork later)
    if n_jobs not in (None, 1):
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)

    return torch.from_numpy(result)


def _sorted_knn_1d(values: np.ndarray, k: int, period: float | None = None,
                    chunk_size: int = 500_000) -> np.ndarray:
    """
    k-NN indices along a single scalar axis, via sorting instead of a tree.

    Parameters
    ----------
    values : (n,) array to compute neighbours on
    k : neighbours per point, excluding self
    period : if given, `values` is treated as circular with the given period
        (e.g. 2*pi for an angle), so windows can cross the wrap point.
        None for a plain linear axis.
    chunk_size : rows processed per batch to bound peak memory.

    Returns
    -------
    (n, k) int32 array of neighbour indices
    """
    n = len(values)
    if k >= n:
        raise ValueError(f"n_neighbours ({k}) must be smaller than n_samples ({n})")

    # Sort values
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]

    # Check if cyclic time is used
    if period is not None:
        # Pad windows near both edges (so they see the other edge)
        pad_vals = np.concatenate([sorted_vals[-k:] - period, sorted_vals, sorted_vals[:k] + period])
        pad_order = np.concatenate([order[-k:], order, order[:k]])
    else:
        # Pad with +/-inf (to be usable with fixed-size sliding window)
        pad_vals = np.concatenate([np.full(k, -np.inf), sorted_vals, np.full(k, np.inf)])
        pad_order = np.concatenate([np.full(k, -1, dtype=order.dtype), order, np.full(k, -1, dtype=order.dtype)])

    # Define sliding windows (centred around point k;
    # potential neighbours can be left and/or right of the centre)
    window_len = 2 * k + 1
    val_windows = np.lib.stride_tricks.sliding_window_view(pad_vals, window_len)
    idx_windows = np.lib.stride_tricks.sliding_window_view(pad_order, window_len)
    center = k

    # Iterate over chunks (for memory efficiency)
    result = np.empty((n, k), dtype=np.int32)
    for start in range(0, n, chunk_size):
        # Get last data point for the chunk
        stop = min(start + chunk_size, n)

        # Define windows of potential nearest neighbours
        vw = val_windows[start:stop]
        iw = idx_windows[start:stop]

        # Compute distances
        dist = np.abs(vw - vw[:, center:center + 1])
        dist[:, center] = np.inf  # exclude self

        # Get nearest k neighbours
        nearest = np.argpartition(dist, k, axis=1)[:, :k]  # idx of k nearest distances
        nearest_dist = np.take_along_axis(dist, nearest, axis=1)  # actual distances
        nearest = np.take_along_axis(nearest, np.argsort(nearest_dist, axis=1), axis=1)  # sort indexes according to distance

        # Project window indexes to original dataset row indexes
        neighbour_idx = np.take_along_axis(iw, nearest, axis=1)

        # Add neighbours to results
        result[order[start:stop]] = neighbour_idx

    return result


def _time_knn_indices(time_coords: np.ndarray, k: int) -> torch.Tensor:
    """k-NN indices in the time subspace via :func:`_sorted_knn_1d`."""
    if time_coords.shape[1] == 1:
        # Linear time
        idx = _sorted_knn_1d(time_coords[:, 0], k, period=None)
    else:
        # Cyclic time (compute angles for circular distance search)
        cos_a, sin_a = 2 * time_coords[:, 0] - 1, 2 * time_coords[:, 1] - 1
        angle = np.arctan2(sin_a, cos_a)
        idx = _sorted_knn_1d(angle, k, period=2 * np.pi)
    return torch.from_numpy(idx)


def build_graph(
        coords_scaled: Tensor,
        n_neighbours_space: int,
        n_neighbours_time: int,
        n_jobs: int | None = None,
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
    n_jobs : int or None, default=None
        Parallel jobs for the spatial (tree-based) k-NN search, forwarded to
        :class:`sklearn.neighbors.NearestNeighbors` (``None`` means 1, ``-1``
        means all available cores). The temporal search is sort-based and
        single-threaded, so this has no effect on it.

    Returns
    -------
    dict with keys ``'space'`` and ``'time'``, each a LongTensor of shape (n, k)
    """
    c = coords_scaled.cpu().numpy()
    space_idx = _knn_indices(c[:, :4], n_neighbours_space, n_jobs=n_jobs)  # x, y, z, depth
    time_idx = _time_knn_indices(c[:, 4:], n_neighbours_time)  # scaled time (1 or 2 cols)
    return {"space": space_idx, "time": time_idx}


def precompute_graph(
        coords,
        n_neighbours_space: int,
        n_neighbours_time: int,
        cyclic_time: bool = False,
        n_jobs: int | None = None,
) -> Dict[str, Tensor]:
    """
    Build a k-NN graph from raw coordinates, e.g. for re-use across tuning trials.

    The graph depends only on the coordinates and the neighbour counts -> during
    hyperparameter tuning it can be built once (with the largest ``k`` under
    consideration) and passed to :meth:`mastnet.MaSTNetImputer.fit` via its
    ``graph`` argument, which slices the first
    ``n_neighbours_space`` / ``n_neighbours_time`` columns per trial.

    Note: coordinate scalers are fitted on *all* rows here, whereas
    :meth:`~mastnet.MaSTNetImputer.fit` fits them on the training rows only.

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
    n_jobs : int or None, default=None
        Parallel jobs for the spatial k-NN search; see :func:`build_graph`.

    Returns
    -------
    dict with keys ``'space'`` and ``'time'``, each a LongTensor of shape (n, k)
    """
    coords_t = torch.as_tensor(np.asarray(coords, dtype=float), dtype=torch.float32)
    coords_scaled, _, _ = preprocess(
        coords_t, torch.zeros((len(coords_t), 0)), [], cyclic_time,
    )
    return build_graph(coords_scaled, n_neighbours_space, n_neighbours_time, n_jobs=n_jobs)


class NeighbourDataset(Dataset):
    """
    PyTorch Dataset returning a query point and its spatial/temporal neighbours.

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
