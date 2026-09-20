"""Equal-area spatial block assignment via HEALPix."""

import astropy_healpix as ah
import astropy.units as u
import numpy as np


def nside_for_block_size_deg(block_size_deg: float) -> int:
    """Return the power-of-2 HEALPix ``nside`` whose pixel resolution is
    closest to ``block_size_deg``, for comparability with a degree-grid block size.
    """
    return int(ah.pixel_resolution_to_nside(block_size_deg * u.deg, round="nearest"))


def healpix_pixel_area_deg2(nside: int) -> float:
    """Area of a single HEALPix pixel at this ``nside``, in deg^2."""
    return float(ah.nside_to_pixel_area(nside).to_value(u.deg ** 2))


def healpix_block_ids(lat: np.ndarray, lon: np.ndarray, nside: int) -> np.ndarray:
    """Assign each ``(lat, lon)`` point to an equal-area HEALPix pixel.

    Parameters
    ----------
    lat, lon : ndarray of float
        Coordinates in degrees.
    nside : int
        HEALPix resolution parameter (see :func:`nside_for_block_size_deg`).
        Should be a power of 2 for full compatibility with HEALPix's
        hierarchical pixel-nesting property.

    Returns
    -------
    ndarray of int
        Pixel index per point, usable directly as a block ID.
    """
    pix = ah.lonlat_to_healpix(np.asarray(lon) * u.deg, np.asarray(lat) * u.deg, nside, order="ring")
    return np.asarray(pix)
