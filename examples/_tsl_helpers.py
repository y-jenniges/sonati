"""
Dataset download/loading helpers ported from tsl (torch-spatiotemporal), MIT licensed.
Copyright (c) 2022 Andrea Cini, Ivan Marisca. https://github.com/TorchSpatiotemporal/tsl
    tsl/datasets/metr_la.py::MetrLA -> load_metr_la()
    tsl/datasets/air_quality.py::AirQuality -> load_aqi36()

Helpers are ported to avoid dependence on the full tsl package and adapted to what
examples/real_world_test.py needs.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
import pandas as pd
import requests


# Dataset URLs
METR_LA_URL = ("https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/traffic/metr_la.zip")
AQI_URL = ("https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/air/air_quality.zip")


def _download_and_extract(url: str, cache_dir: Path, expected_file: str) -> None:
    """Download a zip into cache_dir and extract it, unless expected_file (one of
    the files the zip contains) is already there from a previous run."""
    if (cache_dir / expected_file).exists():
        return

    # Download
    print(f"  Downloading {url.rsplit('/', 1)[-1]} ...", end="", flush=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    # Extract zip
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(cache_dir)
    print(" done  (cached at examples/.cache/ for future runs)")


# --- METR-LA ------------------------------------------------------------------ #
# Ported from tsl.datasets.metr_la.MetrLA.load_raw().

def load_metr_la(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download, cache, and return (speed_df, locations_df).

    speed_df : (time x sensor) matrix, 5-min steps, 0.0 marks a sensor failure.
    locations_df : One row per sensor with sensor_id/latitude/longitude columns.
    """
    _download_and_extract(METR_LA_URL, cache_dir, "metr_la.h5")

    # Load traffic data
    speed_df = pd.read_hdf(cache_dir / "metr_la.h5")

    # Add missing values
    full_index = pd.date_range(speed_df.index[0], speed_df.index[-1], freq="5min")
    speed_df = speed_df.reindex(index=full_index)

    # Load location data
    locations_df = pd.read_csv(cache_dir / "sensor_locations_la.csv")
    return speed_df, locations_df


# --- Air quality (AQI-36) ------------------------------------------------------ #
# Ported from tsl.datasets.air_quality.AirQuality, small=True path

def load_aqi36(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download, cache, and return (pm25_df, eval_mask_df, stations_df).

    pm25_df : (time x station) PM2.5 readings for 36 Beijing stations, hourly,
        one year (2014-05 to 2015-04). ~13 % naturally missing.
    eval_mask_df : (time x station), bool - True marks the cells to evaluate on.
    stations_df : One row per station with latitude/longitude columns.
    """
    _download_and_extract(AQI_URL, cache_dir, "small36.h5")

    # Drop the full dataset (only keep the small subset)
    (cache_dir / "full437.h5").unlink(missing_ok=True)

    # Load air quality data
    pm25_df = pd.read_hdf(cache_dir / "small36.h5", "pm25")
    eval_mask_df = pd.read_hdf(cache_dir / "small36.h5", "eval_mask")
    stations_df = pd.read_hdf(cache_dir / "small36.h5", "stations")

    return pm25_df, eval_mask_df, stations_df
