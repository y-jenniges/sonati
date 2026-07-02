"""
Real-world tests of MaSTNeT imputer on two spatio-temporal domains.

Usage
-----
    python examples/real_world_test.py            # both datasets
    python examples/real_world_test.py air quality  # air quality only
    python examples/real_world_test.py traffic     # traffic only

Requirements (beyond mastnet-imputer deps)
------------------------------------------
    pip install requests
    scikit-learn is already a core dependency

Datasets
--------
1. Air quality - Open-Meteo free API (no key required)
   20 European city-centre locations, last 7 days hourly.
   Variables: PM2.5 (μg/m³), PM10 (μg/m³), European AQI.
   https://open-meteo.com/en/docs/air-quality-api

2. Traffic - METR-LA (real data, Los Angeles freeway sensors)
   207 inductive-loop detector positions from the DCRNN paper
   (Yaguang Li et al., 2018). Speed in mph, 5-min intervals.
   Downloaded from SwitchDrive (Swiss research infrastructure),
   cached locally after first run.
   Source used by torch-spatiotemporal (Cini et al., 2022).

Baselines
---------
  Mean fill - per-feature column mean. Ignores all structure.
  KNN (feat) - sklearn KNNImputer(k=5) in feature space.
               Finds nearest neighbours by feature similarity only;
               ignores lat/lon/time entirely.
  MaSTNet - spatiotemporal cross-attention (this package).

Metrics
-------
  MAE - mean absolute error in original units.
  NMAE - normalised MAE = MAE / mean(|true values|) x 100 %.
         Allows comparison across features with different scales
         and with published benchmarks.
"""

from __future__ import annotations

import io
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.impute import KNNImputer, SimpleImputer

from mastnet import MaSTNetImputer

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


# -- evaluation helpers -------------------------------------------------------

def _mask_fraction(X: np.ndarray, frac: float, seed: int = 0):
    """Randomly blank out `frac` of the observed (non-NaN) values."""
    rng = np.random.default_rng(seed)
    held = ~np.isnan(X) & (rng.random(X.shape) < frac)
    Xm = X.copy()
    Xm[held] = np.nan
    return Xm, held


def _compute_metrics(X_true: np.ndarray, X_pred: np.ndarray,
                     held_out: np.ndarray,
                     feature_names: list[str]) -> dict[str, dict[str, float]]:
    """MAE and NMAE per feature and overall, evaluated on held-out positions."""
    mae: dict[str, float] = {}
    nmae: dict[str, float] = {}

    # Iterate over features
    for i, name in enumerate(feature_names):
        # Filter values
        col_mask = held_out[:, i]
        true_vals = X_true[col_mask, i]
        pred_vals = X_pred[col_mask, i]

        # Compute metrics
        mae[name] = float(np.abs(pred_vals - true_vals).mean())
        nmae[name] = float(mae[name] / np.abs(true_vals).mean() * 100)

    # Aggregate metrics
    mae["overall"] = float(np.abs(X_pred[held_out] - X_true[held_out]).mean())
    nmae["overall"] = float(mae["overall"] / np.abs(X_true[held_out]).mean() * 100)
    return {"mae": mae, "nmae": nmae}


def _print_tables(results: dict[str, dict],
                  feature_names: list[str],
                  mae_label: str = "MAE  (lower is better)",
                  nmae_label: str = "NMAE %  = MAE / mean(|true|) x 100  (lower is better)",
                  ref_note: str = "") -> None:
    """Print results."""
    col_names = feature_names + ["overall"]
    col_w = max(10, max(len(n) for n in col_names) + 2)
    meth_w = max(len(m) for m in results) + 2

    def _table(metric: str, fmt: str, suffix: str = "") -> None:
        header = f"  {'method':<{meth_w}}" + "".join(f"{n:>{col_w}}" for n in col_names)
        sep = "  " + "-" * (meth_w + col_w * len(col_names))
        print(header)
        print(sep)
        for method, metrics in results.items():
            vals = metrics[metric]
            row = f"  {method:<{meth_w}}"
            for name in col_names:
                row += f"{vals[name]:>{col_w}{fmt}}{suffix}"
            print(row)

    print(f"\n  {mae_label}")
    _table("mae", ".3f")
    print(f"\n  {nmae_label}")
    _table("nmae", ".1f", "%")
    if ref_note:
        print(f"\n  {ref_note}")


# -- 1. Air quality -----------------------------------------------------------

CITIES = [
    ("London", 51.51, -0.13), ("Paris", 48.85, 2.35),
    ("Berlin", 52.52, 13.41), ("Madrid", 40.42, -3.70),
    ("Rome", 41.90, 12.49), ("Warsaw", 52.23, 21.01),
    ("Amsterdam", 52.37, 4.90), ("Vienna", 48.21, 16.37),
    ("Brussels", 50.85, 4.35), ("Prague", 50.08, 14.44),
    ("Budapest", 47.50, 19.04), ("Stockholm", 59.33, 18.07),
    ("Copenhagen", 55.68, 12.57), ("Oslo", 59.91, 10.75),
    ("Helsinki", 60.17, 24.94), ("Athens", 37.98, 23.73),
    ("Lisbon", 38.72, -9.14), ("Bucharest", 44.44, 26.10),
    ("Sofia", 42.70, 23.32), ("Zagreb", 45.81, 15.98),
]
AQ_VARS = ["pm2_5", "pm10", "european_aqi"]


def run_air_quality() -> None:
    print("\n" + "=" * 64)
    print("AIR QUALITY -- Open-Meteo, 20 European cities, 7-day hourly")
    print("=" * 64)

    end_date = date.today()
    start_date = end_date - timedelta(days=7)

    rows: list[pd.DataFrame] = []
    print(f"  Fetching {len(CITIES)} cities from Open-Meteo ...", end="", flush=True)
    for _, lat, lon in CITIES:
        r = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(AQ_VARS),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
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
    t0 = pd.to_datetime(aq["time"]).min()
    aq["time_h"] = (pd.to_datetime(aq["time"]) - t0).dt.total_seconds() / 3600

    X = aq[AQ_VARS].to_numpy(dtype=float)
    coords = np.column_stack([aq["lat"], aq["lon"],
                              np.zeros(len(aq)), aq["time_h"]])

    natural_nan = np.isnan(X).mean()
    X_masked, held_out = _mask_fraction(X, frac=0.20, seed=0)
    print(f"  {len(X)} rows | natural NaN {natural_nan:.1%} "
          f"-> after extra 20 % masking: {np.isnan(X_masked).mean():.1%}")
    print(f"  Units: pm2_5 ug/m3 | pm10 ug/m3 | european_aqi EAQI")

    results: dict[str, dict] = {}

    print("\n  Running baselines ...", end="", flush=True)
    X_mean = SimpleImputer(strategy="mean").fit_transform(X_masked)
    results["Mean fill"] = _compute_metrics(X, X_mean, held_out, AQ_VARS)

    X_knn = KNNImputer(n_neighbors=5).fit_transform(X_masked)
    results["KNN feat-space (k=5)"] = _compute_metrics(X, X_knn, held_out, AQ_VARS)
    print(" done")

    print("  Running MaSTNet ...")
    imputer = MaSTNetImputer(
        n_neighbours_space=20, n_neighbours_time=20, max_epochs=40, patience=5,
        d_model=32, dim_feedforward=64,
        n_spatial_layers=2, n_temporal_layers=2,
        device="auto", seed=42,
    )
    X_mastnet = imputer.fit_transform(X_masked, coords)
    obs_mae = np.abs(X_mastnet[~np.isnan(X_masked)] - X_masked[~np.isnan(X_masked)]).mean()
    print(f"  MAE on observed values: {obs_mae:.4f}  [should be ~0]")
    results["MaSTNet"] = _compute_metrics(X, X_mastnet, held_out, AQ_VARS)

    _print_tables(results, AQ_VARS)


# -- 2. Traffic -- real METR-LA data ------------------------------------------

METR_LA_URL = "https://drive.switch.ch/index.php/s/Z8cKHAVyiDqkzaG/download"
METR_LA_HOURS = 72  # 3 days of hourly data (subsampled from 5-min) max is 4 months?
TRAFFIC_VARS = ["speed_mph"]


def _load_metr_la() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download once, cache, and return (speed_df, locations)."""
    h5_path = CACHE_DIR / "metr_la.h5"
    loc_path = CACHE_DIR / "sensor_locations_la.csv"

    if not h5_path.exists():
        print("  Downloading METR-LA (~13 MB) ...", end="", flush=True)
        r = requests.get(METR_LA_URL, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(CACHE_DIR)
        print(" done  (cached at examples/.cache/ for future runs)")
    else:
        print("  Loading cached METR-LA data")

    speed_df = pd.read_hdf(h5_path)
    locations = pd.read_csv(loc_path)
    return speed_df, locations


def run_traffic() -> None:
    print("\n" + "=" * 64)
    print("TRAFFIC -- METR-LA, 207 LA freeway sensors (real data)")
    print("=" * 64)

    speed_df, locations = _load_metr_la()

    # Subset: first 3 days at hourly resolution (every 12th 5-min sample)
    speed_sub = speed_df.iloc[:METR_LA_HOURS * 12: 12].copy()  # (72, 207)

    # Zeros encode sensor failures -- treat as missing
    speed_sub = speed_sub.replace(0.0, np.nan)
    natural_nan_pct = speed_sub.isna().mean().mean()

    # Build (n_samples, n_features) arrays in long format
    locations = locations.set_index("sensor_id")
    sensor_ids = speed_sub.columns.astype(int)
    lats = locations.loc[sensor_ids, "latitude"].values  # (207,)
    lons = locations.loc[sensor_ids, "longitude"].values  # (207,)

    t0 = speed_sub.index[0]
    t_h = np.array([(t - t0).total_seconds() / 3600 for t in speed_sub.index])

    n_t, n_s = speed_sub.shape  # 72, 207
    lats_flat = np.repeat(lats, n_t)  # sensor lat for each row
    lons_flat = np.repeat(lons, n_t)
    t_flat = np.tile(t_h, n_s)  # hour for each row
    X = speed_sub.values.T.flatten().reshape(-1, 1)  # (14904, 1)

    coords = np.column_stack([lats_flat, lons_flat,
                              np.zeros(len(X)), t_flat])

    X_masked, held_out = _mask_fraction(X, frac=0.25, seed=1)
    print(f"  {len(X)} rows ({n_s} sensors x {METR_LA_HOURS} h hourly) | "
          f"natural NaN {natural_nan_pct:.1%} | total masked: {np.isnan(X_masked).mean():.1%}")
    print(f"  Unit: speed in mph")

    results: dict[str, dict] = {}

    print("\n  Running baselines ...", end="", flush=True)
    X_mean = SimpleImputer(strategy="mean").fit_transform(X_masked)
    results["Mean fill"] = _compute_metrics(X, X_mean, held_out, TRAFFIC_VARS)

    X_knn = KNNImputer(n_neighbors=5).fit_transform(X_masked)
    results["KNN feat-space (k=5)"] = _compute_metrics(X, X_knn, held_out, TRAFFIC_VARS)
    print(" done")

    print("  Running MaSTNet ...")
    imputer = MaSTNetImputer(
        n_neighbours_space=15, n_neighbours_time=15, max_epochs=20, patience=5,
        d_model=32, dim_feedforward=64,
        n_spatial_layers=2, n_temporal_layers=2,
        device="auto", seed=0,
    )
    X_mastnet = imputer.fit_transform(X_masked, coords)
    obs_mae = np.abs(X_mastnet[~np.isnan(X_masked)] - X_masked[~np.isnan(X_masked)]).mean()
    print(f"  MAE on observed values: {obs_mae:.4f}  [should be ~0]")
    results["MaSTNet"] = _compute_metrics(X, X_mastnet, held_out, TRAFFIC_VARS)

    _print_tables(
        results, TRAFFIC_VARS,
        ref_note=(
            "Reference -- GRIN (Cini et al., 2022) on METR-LA, 25 % random missing:\n"
            "  MAE ~1.40 mph  |  NMAE ~3.1 %\n"
            "  (larger model, full 4-month dataset, trained to convergence)"
        ),
    )


# Entry point
if __name__ == "__main__":
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    if target in ("airquality", "air", "both"):
        run_air_quality()
    if target in ("traffic", "both"):
        run_traffic()

    print("\nDone.")
