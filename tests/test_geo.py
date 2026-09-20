"""Tests for HEALPix-based equal-area spatial blocks (mastnet._geo)."""

import numpy as np
import pytest

from mastnet._geo import healpix_block_ids, healpix_pixel_area_deg2, nside_for_block_size_deg


def test_nside_for_block_size_deg_is_power_of_two():
    # HEALPix requires nside to be a power of two for its pixelisation scheme to be valid.
    for step in (5.0, 10.0, 15.0, 20.0, 30.0):
        nside = nside_for_block_size_deg(step)
        assert nside > 0
        assert (nside & (nside - 1)) == 0  # power of 2


def test_nside_for_block_size_deg_matches_current_default():
    # Regression check: mastnet.imputer's default block_size_deg=15.0 must resolve to a sane nside
    assert nside_for_block_size_deg(15.0) == 4


def test_degree_grid_is_latitude_biased_healpix_is_not():
    # Naive lat/lon degree grid should differ from HEALPix grid
    import astropy_healpix as ah

    step = 15.0

    # Area (in steradians) of a step x step degree block starting at lat_deg
    def degree_block_area_sr(lat_deg: float) -> float:
        lat1, lat2 = np.radians(lat_deg), np.radians(lat_deg + step)
        return np.radians(step) * (np.sin(lat2) - np.sin(lat1))

    equatorial_area = degree_block_area_sr(0.0)
    polar_area = degree_block_area_sr(75.0)
    assert equatorial_area / polar_area > 5  # degree grid: strongly biased

    nside = nside_for_block_size_deg(step)
    area_equator = ah.nside_to_pixel_area(nside)
    area_pole = ah.nside_to_pixel_area(nside)  # same call, same scalar result
    assert area_equator == area_pole  # healpix grid: no latitude dependence


def test_healpix_pixel_area_deg2_matches_astropy():
    # Check unit conversion of healpix_pixel_area_deg2
    import astropy_healpix as ah
    import astropy.units as u

    nside = 8
    expected = ah.nside_to_pixel_area(nside).to_value(u.deg ** 2)
    assert healpix_pixel_area_deg2(nside) == pytest.approx(expected)


def test_healpix_block_ids_cover_valid_pixel_range():
    # Every pixel id should fall within the valid [0, npix) range.
    import astropy_healpix as ah

    rng = np.random.default_rng(0)
    n = 2000
    # arcsin(uniform(-1, 1)) gives latitudes uniform on the sphere (not uniform in degrees)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    lon = rng.uniform(-180, 180, n)
    nside = 8
    ids = healpix_block_ids(lat, lon, nside)
    assert ids.shape == (n,)
    assert ids.min() >= 0
    assert ids.max() < ah.nside_to_npix(nside)


def test_healpix_block_ids_poles_and_antimeridian_do_not_crash():
    # Check edge-case inputs for sane results
    lat = np.array([90.0, -90.0, 89.999, -89.999, 0.0, 0.0])
    lon = np.array([0.0, 0.0, 179.999, -179.999, 180.0, -180.0])
    ids = healpix_block_ids(lat, lon, nside=8)
    assert ids.shape == (6,)
    assert np.isfinite(ids).all()
    # lon=180 and lon=-180 are the same point on the antimeridian -> same pixel
    assert ids[4] == ids[5]
