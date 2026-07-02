"""Coordinate and feature preprocessing for MaSTNeT."""

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


def preprocess(coords: torch.Tensor, values: torch.Tensor,
               feature_names: list[str], cyclic_time: bool = False,
               scaler_dict: dict | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Scale spatio-temporal coordinates and feature values.

    Coordinates are expected as columns [latitude (°), longitude (°), depth (m), time].
    Latitude and longitude are mapped to Cartesian coordinates on the unit sphere,
    then shifted to [0, 1]. Depth and time are MinMax-scaled using scalers fitted
    on the data passed to this call (or provided via ``scaler_dict``).

    Features are scaled independently per variable using MinMaxScaler fitted on
    observed (non-NaN) values only.

    Parameters
    ----------
    coords : torch.Tensor of shape (n, 4)
    values : torch.Tensor of shape (n, n_features) - NaN marks missing values
    feature_names : list of str - one name per feature column (used as scaler keys)
    cyclic_time : bool - encode time with sine/cosine instead of linear MinMax
    scaler_dict : dict or None - if provided, reuse these fitted scalers

    Returns
    -------
    coords_scaled : torch.Tensor of shape (n, 5) or (n, 6) for cyclic_time
    values_scaled : torch.Tensor of shape (n, n_features)
    scaler_dict : dict - scalers keyed by "depth", "time", and each feature name
    """
    if scaler_dict is None:
        scaler_dict = {}

    coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
    values_np = values.cpu().numpy() if torch.is_tensor(values) else np.asarray(values)

    lat, lon, depth, time = (coords_np[:, i] for i in range(4))

    # Lat/lon → unit sphere Cartesian, shifted to [0, 1]
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    x = (np.cos(lat_r) * np.cos(lon_r) + 1) / 2
    y = (np.cos(lat_r) * np.sin(lon_r) + 1) / 2
    z = (np.sin(lat_r) + 1) / 2

    # Depth
    if "depth" in scaler_dict:
        depth_s = scaler_dict["depth"].transform(depth.reshape(-1, 1))
    else:
        scaler_dict["depth"] = MinMaxScaler().fit(depth.reshape(-1, 1))
        depth_s = scaler_dict["depth"].transform(depth.reshape(-1, 1))

    # Time - linear MinMax or cyclic sine/cosine
    if cyclic_time:
        # Assumes time is a month index (1–12); wraps cyclically
        t = 2 * np.pi * time / 12
        time_cols = np.column_stack([(np.cos(t) + 1) / 2, (np.sin(t) + 1) / 2])
    else:
        if "time" in scaler_dict:
            time_cols = scaler_dict["time"].transform(time.reshape(-1, 1))
        else:
            scaler_dict["time"] = MinMaxScaler().fit(time.reshape(-1, 1))
            time_cols = scaler_dict["time"].transform(time.reshape(-1, 1))

    coords_scaled = np.column_stack([x, y, z, depth_s, time_cols])

    # Features - per-variable MinMaxScaler on observed values only
    values_scaled = np.full_like(values_np, np.nan)
    for i, name in enumerate(feature_names):
        col = values_np[:, i]
        observed = ~np.isnan(col)

        # No scaling if no observations
        if observed.sum() == 0:
            continue

        # Use input scalers if given or create new ones
        if name in scaler_dict:
            values_scaled[observed, i] = scaler_dict[name].transform(col[observed].reshape(-1, 1)).flatten()
        else:
            scaler_dict[name] = MinMaxScaler().fit(col[observed].reshape(-1, 1))
            values_scaled[observed, i] = scaler_dict[name].transform(col[observed].reshape(-1, 1)).flatten()

    return (
        torch.tensor(coords_scaled, dtype=torch.float32),
        torch.tensor(values_scaled, dtype=torch.float32),
        scaler_dict,
    )


def inverse_transform(values_scaled: np.ndarray, scaler_dict: dict,
                      feature_names: list[str]) -> np.ndarray:
    """
    Invert per-feature MinMax scaling.

    Parameters
    ----------
    values_scaled : ndarray of shape (n, n_features)
    scaler_dict : dict
        Fitted scalers keyed by feature name, as returned by :func:`preprocess`.
    feature_names : list of str

    Returns
    -------
    ndarray of shape (n, n_features)
        Values restored to the original (unscaled) range.
    """
    out = values_scaled.copy()

    # Iterate over features
    for i, name in enumerate(feature_names):
        # Unscale
        if name in scaler_dict:
            out[:, i] = scaler_dict[name].inverse_transform(
                out[:, i].reshape(-1, 1)
            ).flatten()
    return out
