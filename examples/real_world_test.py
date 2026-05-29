"""
Real-world tests of MaSTNetImputer on two spatiotemporal domains.

Usage
-----
    python examples/real_world_test.py            # both datasets
    python examples/real_world_test.py airquality  # air quality only
    python examples/real_world_test.py traffic     # traffic only

Requirements (beyond mastnet-imputer deps)
------------------------------------------
    pip install requests

Datasets
--------
1. Air quality — Open-Meteo free API (no key required)
   20 European city-centre locations, last 7 days hourly.
   Variables: PM2.5, PM10, European AQI.
   https://open-meteo.com/en/docs/air-quality-api

2. Traffic — METR-LA (Los Angeles freeway sensors)
   207 real inductive-loop detector positions from the DCRNN paper
   (Yaguang Li et al., 2018).  Sensor lat/lon fetched from GitHub;
   speed and volume values are SYNTHETIC with a realistic rush-hour
   profile.  To benchmark on real traffic, download metr-la.h5 from
   https://github.com/liyaguang/DCRNN and adapt the loading section.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from mastnet import MaSTNetImputer


# ── helpers ──────────────────────────────────────────────────────────────────

def _mask_fraction(X: np.ndarray, frac: float, seed: int = 0):
    """Randomly blank out `frac` of the observed (non-NaN) values."""
    rng  = np.random.default_rng(seed)
    held = ~np.isnan(X) & (rng.random(X.shape) < frac)
    Xm   = X.copy()
    Xm[held] = np.nan
    return Xm, held


def _report(tag: str, X_in: np.ndarray, X_out: np.ndarray, X_true: np.ndarray,
            held_out: np.ndarray) -> None:
    nan_in  = np.isnan(X_in).mean()
    nan_out = np.isnan(X_out).mean()
    obs     = ~np.isnan(X_in)
    delta_obs  = np.abs(X_out[obs]      - X_in[obs]).mean()
    delta_held = np.abs(X_out[held_out] - X_true[held_out]).mean()
    print(f"\n  {tag}")
    print(f"    shape            : {X_in.shape}")
    print(f"    missing → missing: {nan_in:.1%} → {nan_out:.1%}")
    print(f"    MAE (observed)   : {delta_obs:.4f}  [should be ~0]")
    print(f"    MAE (held-out)   : {delta_held:.4f}  [imputation quality]")


# ── 1. Air quality ────────────────────────────────────────────────────────────

CITIES = [
    ("London",      51.51,  -0.13), ("Paris",       48.85,   2.35),
    ("Berlin",      52.52,  13.41), ("Madrid",      40.42,  -3.70),
    ("Rome",        41.90,  12.49), ("Warsaw",      52.23,  21.01),
    ("Amsterdam",   52.37,   4.90), ("Vienna",      48.21,  16.37),
    ("Brussels",    50.85,   4.35), ("Prague",      50.08,  14.44),
    ("Budapest",    47.50,  19.04), ("Stockholm",   59.33,  18.07),
    ("Copenhagen",  55.68,  12.57), ("Oslo",        59.91,  10.75),
    ("Helsinki",    60.17,  24.94), ("Athens",      37.98,  23.73),
    ("Lisbon",      38.72,  -9.14), ("Bucharest",   44.44,  26.10),
    ("Sofia",       42.70,  23.32), ("Zagreb",      45.81,  15.98),
]
AQ_VARS = ["pm2_5", "pm10", "european_aqi"]


def run_air_quality() -> None:
    print("\n" + "=" * 64)
    print("AIR QUALITY — Open-Meteo, 20 European cities, 7-day hourly")
    print("=" * 64)

    end_date   = date.today()
    start_date = end_date - timedelta(days=7)

    rows: list[pd.DataFrame] = []
    print(f"  Fetching {len(CITIES)} cities from Open-Meteo ...", end="", flush=True)
    for _, lat, lon in CITIES:
        r = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude":   lat,
                "longitude":  lon,
                "hourly":     ",".join(AQ_VARS),
                "start_date": start_date.isoformat(),
                "end_date":   end_date.isoformat(),
            },
            timeout=15,
        )
        r.raise_for_status()
        df = pd.DataFrame(r.json()["hourly"])
        df["lat"] = lat
        df["lon"] = lon
        rows.append(df)
    print(" done")

    aq = pd.concat(rows, ignore_index=True)
    t0          = pd.to_datetime(aq["time"]).min()
    aq["time_h"] = (pd.to_datetime(aq["time"]) - t0).dt.total_seconds() / 3600

    X      = aq[AQ_VARS].to_numpy(dtype=float)
    coords = np.column_stack([aq["lat"], aq["lon"],
                               np.zeros(len(aq)), aq["time_h"]])

    natural_nan = np.isnan(X).mean()
    X_masked, held_out = _mask_fraction(X, frac=0.20, seed=0)
    print(f"  {len(X)} rows | natural NaN {natural_nan:.1%} "
          f"→ after extra 20 % masking: {np.isnan(X_masked).mean():.1%}")

    imputer = MaSTNetImputer(
        n_neighbours=20, max_epochs=40, patience=5,
        d_model=32, dim_feedforward=64,
        n_spatial_layers=2, n_temporal_layers=2,
        device="auto", seed=42,
    )
    X_filled = imputer.fit_transform(X_masked, coords)
    _report("Air quality", X_masked, X_filled, X, held_out)


# ── 2. Traffic (METR-LA positions + synthetic values) ─────────────────────────

METR_LA_SENSOR_URL = (
    "https://raw.githubusercontent.com/liyaguang/DCRNN"
    "/master/data/sensor_graph/graph_sensor_locations.csv"
)
TRAFFIC_HOURS = 72          # 3 days — increase for a fuller benchmark
TRAFFIC_VARS  = ["speed_kmh", "volume"]


def _synthetic_traffic(sensors: pd.DataFrame, n_hours: int,
                        seed: int = 0) -> pd.DataFrame:
    """Rush-hour speed + volume profile. Replace with real metr-la.h5 data."""
    rng     = np.random.default_rng(seed)
    records = []
    for _, row in sensors.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        for h in range(n_hours):
            hod   = h % 24
            rush  = (np.exp(-((hod - 8) ** 2) / 4) +
                     np.exp(-((hod - 17) ** 2) / 4))
            speed = max(5.0,  105 - 55 * rush + rng.normal(0, 4))
            vol   = max(0.0,  250 * rush + 40  + rng.normal(0, 20))
            records.append([lat, lon, float(h), speed, vol])
    return pd.DataFrame(records, columns=["lat", "lon", "time_h"] + TRAFFIC_VARS)


def run_traffic() -> None:
    print("\n" + "=" * 64)
    print("TRAFFIC — METR-LA sensor positions (synthetic speed/volume)")
    print("=" * 64)

    print("  Fetching METR-LA sensor locations ...", end="", flush=True)
    sensors = pd.read_csv(METR_LA_SENSOR_URL,
                          usecols=["sensor_id", "latitude", "longitude"])
    print(f" {len(sensors)} sensors")

    df     = _synthetic_traffic(sensors, n_hours=TRAFFIC_HOURS)
    X      = df[TRAFFIC_VARS].to_numpy(dtype=float)
    coords = np.column_stack([df["lat"], df["lon"],
                               np.zeros(len(df)), df["time_h"]])

    X_masked, held_out = _mask_fraction(X, frac=0.25, seed=1)
    print(f"  {len(X)} rows ({len(sensors)} sensors × {TRAFFIC_HOURS} h) | "
          f"masked: {np.isnan(X_masked).mean():.1%}")

    imputer = MaSTNetImputer(
        n_neighbours=15, max_epochs=20, patience=5,
        d_model=32, dim_feedforward=64,
        n_spatial_layers=2, n_temporal_layers=2,
        device="auto", seed=0,
    )
    X_filled = imputer.fit_transform(X_masked, coords)
    _report("Traffic", X_masked, X_filled, X, held_out)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    if target in ("airquality", "air", "both"):
        run_air_quality()
    if target in ("traffic", "both"):
        run_traffic()

    print("\nDone.")
