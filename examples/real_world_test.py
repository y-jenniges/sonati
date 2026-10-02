"""
Demo of SONATI imputer on two real spatio-temporal benchmark datasets
(not a tuned benchmark, settings are made to keep the script fast).

Usage
-----
    python examples/real_world_test.py  # both datasets
    python examples/real_world_test.py air  # air quality only
    python examples/real_world_test.py traffic  # traffic only

Requirements (beyond sonati deps)
------------------------------------------
    pip install requests tables

Datasets
--------
1.  Air quality - AQI-36 (small, 36 PM2.5 sensors in Beijing)
    Hourly PM2.5 readings, May 2014 - April 2015, from the Urban Air project
    (https://www.microsoft.com/en-us/research/project/ruban-air/).

2.  Traffic - MetrLA (Los Angeles highway sensors)
    Readings every 5 min, March 2012 - June 2012, from the paper
    Yaguang Li, Rose Yu, Cyrus Shababi, Yan Liu, Diffusion Convolutional
    Recurrent Neural Network: Data-Driven Traffic Forecasting, ICLR 2018.

Models
------
  Mean fill - Per-feature column mean.
  KNN (feat) - sklearn KNNImputer(k=5) in feature space.
               Finds nearest neighbours by feature similarity, ignores lat/lon/time.
  SONATI - Spatio-temporal cross-attention (this package).

Metrics
-------
  MAE - Mean absolute error in original units.
  MSE - Mean squared error in original units squared.
  NMAE - Normalised MAE = MAE / mean(|true values|) * 100 %.
         Allows comparison across features with different scales
         and with published benchmarks (equivalent to MRE in Cini et al. 2022).

Comparability to published benchmarks
--------------------------------------
Both reference tables are a rough sanity check. The benchmarked
numbers result from training on the full datasets (here we only use a subset)
and properly tuned architectures (here only small, briefly tuned SONATI).
Benchmark numbers originate from:
    Cini et al 2022: Andrea Cini, Ivan Marisca, & Cesare Alippi (2022).
    Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks.
    In International Conference on Learning Representations.

    Liu et al 2023: Liu, M., Huang, H., Feng, H., Sun, L., Du, B., & Fu, Y. (2023).
    PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation.
    arXiv preprint arXiv:2302.09746.

"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split

from sonati import SONATIImputer
from _tsl_helpers import load_metr_la, load_aqi36

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


# -- Evaluation helpers ------------------------------------------------------- #

def _mask_fraction(X: np.ndarray, frac: float, seed: int = 0):
    """Randomly hide `frac` of the observed (non-NaN) values."""
    rng = np.random.default_rng(seed)
    held = ~np.isnan(X) & (rng.random(X.shape) < frac)
    Xm = X.copy()
    Xm[held] = np.nan
    return Xm, held


def _split_rows(n: int, test_frac: float = 0.20, val_frac: float = 0.20, seed: int = 0):
    """Returns row indices for a test set fully withheld from training and a validation
    set used only for early stopping."""
    train_val_idx, test_idx = train_test_split(np.arange(n), test_size=test_frac, random_state=seed)
    _, val_idx = train_test_split(train_val_idx, test_size=val_frac, random_state=seed)
    return val_idx, test_idx


def _split_val(n: int, val_frac: float = 0.20, seed: int = 0):
    """Split dataset randomly."""
    _, val_idx = train_test_split(np.arange(n), test_size=val_frac, random_state=seed)
    return val_idx


def _compute_metrics(X_true: np.ndarray, X_pred: np.ndarray,
                     held_out: np.ndarray,
                     feature_names: list[str]) -> dict[str, dict[str, float]]:
    """MAE, MSE and NMAE per feature and overall, evaluated on held-out positions."""
    mae: dict[str, float] = {}
    mse: dict[str, float] = {}
    nmae: dict[str, float] = {}

    # Iterate over features
    for i, name in enumerate(feature_names):
        # Filter values
        col_mask = held_out[:, i]
        true_vals = X_true[col_mask, i]
        pred_vals = X_pred[col_mask, i]

        # Compute per-feature metrics
        mae[name] = float(np.abs(pred_vals - true_vals).mean())
        mse[name] = float(((pred_vals - true_vals) ** 2).mean())
        nmae[name] = float(mae[name] / np.abs(true_vals).mean() * 100)

    # Compute overall metrics
    mae["overall"] = float(np.abs(X_pred[held_out] - X_true[held_out]).mean())
    mse["overall"] = float(((X_pred[held_out] - X_true[held_out]) ** 2).mean())
    nmae["overall"] = float(mae["overall"] / np.abs(X_true[held_out]).mean() * 100)
    return {"mae": mae, "mse": mse, "nmae": nmae}


def _print_tables(results: dict[str, dict],
                  feature_names: list[str],
                  mae_label: str = "MAE  (lower is better)",
                  mse_label: str = "MSE  (lower is better)",
                  nmae_label: str = "NMAE %  = MAE / mean(|true|) * 100  (lower is better)",
                  ref_note: str = "") -> None:
    """Print results."""
    col_names = feature_names + ["overall"]
    col_w = max(10, max(len(n) for n in col_names) + 2)
    meth_w = max(len(m) for m in results) + 2

    def _table(metric: str, fmt: str, suffix: str = "") -> None:
        # Print the results table for one metric
        # Header
        header = f"  {'method':<{meth_w}}" + "".join(f"{n:>{col_w}}" for n in col_names)
        sep = "  " + "-" * (meth_w + col_w * len(col_names))
        print(header)
        print(sep)
        # Results
        for method, metrics in results.items():
            vals = metrics[metric]
            row = f"  {method:<{meth_w}}"
            for name in col_names:
                row += f"{vals[name]:>{col_w}{fmt}}{suffix}"
            print(row)

    print(f"\n  {mae_label}")
    _table("mae", ".3f")
    print(f"\n  {mse_label}")
    _table("mse", ".3f")
    print(f"\n  {nmae_label}")
    _table("nmae", ".1f", "%")
    if ref_note:
        print(f"\n  {ref_note}")


# -- 1. Air quality - AQI36 ---------------------------------------------------- #
AQI36_DAYS = 14  # subsample to 14 days (for the small demo to be fast)
AQ_VARS = ["pm25"]


def run_air_quality() -> None:
    print("\n" + "=" * 64)
    print("AIR QUALITY - 36 Beijing PM2.5 stations")
    print("=" * 64)

    pm25_df, eval_mask_df, stations_df = load_aqi36(CACHE_DIR)

    # Subset: First AQI36_DAYS days of the one-year hourly series
    n_hours = AQI36_DAYS * 24
    pm25_sub = pm25_df.iloc[:n_hours]
    eval_mask_sub = eval_mask_df.iloc[:n_hours]
    natural_nan_pct = pm25_sub.isna().mean().mean()

    # Extract station IDs, latitudes and longitudes
    station_ids = pm25_sub.columns
    lats = stations_df.loc[station_ids, "latitude"].values  # (36,)
    lons = stations_df.loc[station_ids, "longitude"].values  # (36,)

    # Hours since the first sample (the numeric time coordinate SONATI requires)
    t0 = pm25_sub.index[0]
    t_h = np.array([(t - t0).total_seconds() / 3600 for t in pm25_sub.index])

    # Build long df (only one feature column)
    n_t, n_s = pm25_sub.shape  # 336, 36
    t_flat = np.tile(t_h, n_s)  # hour for each row
    X = pm25_sub.values.T.flatten().reshape(-1, 1)  # (12096, 1)

    # Build coordinates (one empty column as SONATI expects depth)
    lats_flat = np.repeat(lats, n_t)  # station lat for each row
    lons_flat = np.repeat(lons, n_t)
    coords = np.column_stack([lats_flat, lons_flat, np.zeros(len(X)), t_flat])

    # Hold out test data (evaluation mask is shipped along with the dataset)
    held_out = eval_mask_sub.values.T.flatten().reshape(-1, 1).astype(bool)
    X_masked = X.copy()
    X_masked[held_out] = np.nan

    # val_idx is only needed for early stopping
    val_idx = _split_val(len(X), seed=42)

    # Stats
    print(f"  {len(X)} rows ({n_s} stations x {AQI36_DAYS} days hourly) | "
          f"natural NaN {natural_nan_pct:.1%} | total masked: {np.isnan(X_masked).mean():.1%}")
    print(f"  {held_out.sum()} cells held out for evaluation (AQI36's published eval_mask), "
          f"{len(val_idx)} rows for early stopping")
    print(f"  Unit: μg/m3")

    results: dict[str, dict] = {}

    print("\n  Running baselines ...", end="", flush=True)
    # Mean imputation
    X_mean = SimpleImputer(strategy="mean").fit_transform(X_masked)
    results["Mean fill"] = _compute_metrics(X, X_mean, held_out, AQ_VARS)

    # KNN
    X_knn = KNNImputer(n_neighbors=5).fit_transform(X_masked)
    results["KNN feat-space (k=5)"] = _compute_metrics(X, X_knn, held_out, AQ_VARS)
    print(" done")

    # SONATI (simple config)
    print("  Running SONATI ...")
    imputer = SONATIImputer(
        val_indices=val_idx,
        n_neighbours_space=15, n_neighbours_time=15, max_epochs=20, patience=5,
        d_model=32, dim_feedforward=64,
        n_spatial_layers=2, n_temporal_layers=2,
        device="auto", seed=42,
    )
    X_sonati = imputer.fit_transform(X_masked, coords)
    obs_mae = np.abs(X_sonati[~np.isnan(X_masked)] - X_masked[~np.isnan(X_masked)]).mean()
    print(f"  MAE on observed values: {obs_mae:.4f}  [should be ~0]")  # sanity check
    results["SONATI"] = _compute_metrics(X, X_sonati, held_out, AQ_VARS)

    # Print results (and actual benchmark results)
    _print_tables(
        results, AQ_VARS,
        ref_note=(
            "AQI-36: Real benchmark results (results from this script are only a demo) \n"
            "Based on AQI's 'simulated failure' scenario (~24.6 % missing: ~13 % natural + ~11 % eval_mask): \n"
            "    Mean fill : MAE 53.48  |  MSE 4578.08 (Cini et al., 2022)\n"
            "    KNN (k=5) : MAE 30.21  |  MSE 2892.31 (Cini et al., 2022)\n"
            "    GRIN      : MAE 12.08  |  MSE  523.14   (Cini et al., 2022)\n"
            "    PriSTI    : MAE  9.03  |  MSE  310.39   (Liu et al., 2023)\n"
        ),
    )


# -- 2. Traffic - METR-LA ----------------------------------------------------------- #
METR_LA_HOURS = 72  # subsample to 3 days
TRAFFIC_VARS = ["speed_mph"]


def run_traffic() -> None:
    print("\n" + "=" * 64)
    print("TRAFFIC - METR-LA, 207 LA freeway sensors")
    print("=" * 64)

    speed_df, locations = load_metr_la(CACHE_DIR)

    # Subset: First METR_LA_HOURS/24 days at hourly resolution (every 12th 5-min sample)
    speed_sub = speed_df.iloc[:METR_LA_HOURS * 12: 12].copy()  # (72, 207)

    # Zeros encode sensor failures ("missing")
    speed_sub = speed_sub.replace(0.0, np.nan)
    natural_nan_pct = speed_sub.isna().mean().mean()

    # Build (n_samples, n_features) arrays in long format
    locations = locations.set_index("sensor_id")
    sensor_ids = speed_sub.columns.astype(int)
    lats = locations.loc[sensor_ids, "latitude"].values  # (207,)
    lons = locations.loc[sensor_ids, "longitude"].values  # (207,)

    # Hours since the first sample (the numeric time coordinate SONATI expects)
    t0 = speed_sub.index[0]
    t_h = np.array([(t - t0).total_seconds() / 3600 for t in speed_sub.index])

    # Reshape the (time x sensor) matrix into one (sensor, time) row per observation
    n_t, n_s = speed_sub.shape  # 72, 207
    lats_flat = np.repeat(lats, n_t)  # sensor lat for each row
    lons_flat = np.repeat(lons, n_t)
    t_flat = np.tile(t_h, n_s)  # hour for each row
    X = speed_sub.values.T.flatten().reshape(-1, 1)  # (14904, 1)

    # Build coordinates (one empty column as SONATI expects depth)
    coords = np.column_stack([lats_flat, lons_flat, np.zeros(len(X)), t_flat])

    # Hold out 25 % of observed readings for evaluation (test data)
    X_masked, held_out = _mask_fraction(X, frac=0.25, seed=1)
    val_idx, test_idx = _split_rows(len(X), seed=0)
    is_test_row = np.zeros(len(X), dtype=bool)
    is_test_row[test_idx] = True
    held_out_test = held_out & is_test_row[:, None]

    # Stats
    print(f"  {len(X)} rows ({n_s} sensors x {METR_LA_HOURS} h hourly) | "
          f"natural NaN {natural_nan_pct:.1%} | total masked: {np.isnan(X_masked).mean():.1%}")
    print(f"  {len(test_idx)} rows fully held out as the test set "
          f"({held_out_test.sum()} evaluable cells), {len(val_idx)} rows for early stopping")
    print(f"  Unit: Speed in mph")

    # Hide test indexes (for baselines, SONATI hides them internally)
    X_masked_baseline = X_masked.copy()
    X_masked_baseline[test_idx] = np.nan

    results: dict[str, dict] = {}

    print("\n  Running baselines ...", end="", flush=True)
    # Mean imputation
    X_mean = SimpleImputer(strategy="mean").fit_transform(X_masked_baseline)
    results["Mean fill"] = _compute_metrics(X, X_mean, held_out_test, TRAFFIC_VARS)

    # KNN
    X_knn = KNNImputer(n_neighbors=5).fit_transform(X_masked_baseline)
    results["KNN feat-space (k=5)"] = _compute_metrics(X, X_knn, held_out_test, TRAFFIC_VARS)
    print(" done")

    # SONATI (small demo)
    print("  Running SONATI ...")
    imputer = SONATIImputer(
        val_indices=val_idx, test_idx=test_idx,
        n_neighbours_space=15, n_neighbours_time=15, max_epochs=20, patience=5,
        d_model=32, dim_feedforward=64,
        n_spatial_layers=2, n_temporal_layers=2,
        device="auto", seed=0,
    )
    X_sonati = imputer.fit_transform(X_masked, coords)
    obs_mae = np.abs(X_sonati[~np.isnan(X_masked)] - X_masked[~np.isnan(X_masked)]).mean()
    print(f"  MAE on observed values: {obs_mae:.4f}  [should be ~0]")  # sanity check
    results["SONATI"] = _compute_metrics(X, X_sonati, held_out_test, TRAFFIC_VARS)

    # Print results (and actual benchmark results)
    _print_tables(
        results, TRAFFIC_VARS,
        ref_note=(
            "MetrLA: Real benchmark results (results from this script are only a demo) \n"
            "Reference: GRIN (Cini et al., 2022) \n"
            "    25 % random missing:\n"
            "      MAE ~1.91 mph  |  MSE ~10.41 mph²  |  NMAE ~3.3 %\n"
            "    5 % random + block missing:\n"
            "      MAE ~2.03 mph  |  MSE ~13.26 mph²  |  NMAE ~3.52 %\n"
        ),
    )


# Entry point
if __name__ == "__main__":
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    if target in ("airquality", "air", "both"):
        run_air_quality()
    if target in ("traffic", "both"):
        run_traffic()
