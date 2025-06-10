import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pykrige.ok import OrdinaryKriging
from A1_Data_loader import DataLoader
from A2_elevation_smoothing import compute_smoothed_elevation, get_elevation_at_coords

GRID_RES = 1000  # meters


def compute_nse(predicted, observed):
    """Compute Nash-Sutcliffe Efficiency"""
    mean_obs = np.mean(observed)
    return 1 - np.sum((observed - predicted) ** 2) / np.sum((observed - mean_obs) ** 2)


def main():
    # === Load Data ===
    dem, transform, _ = DataLoader.load_dem()
    stations = DataLoader.load_elevation_data()
    ppt_df = DataLoader.load_precipitation_data()
    ppt_df = ppt_df[stations.index]

    # Mean daily precipitation
    ppt_mean = ppt_df.mean().values
    xs, ys = stations["X"].values, stations["Y"].values

    # Extract raw and smoothed elevation at station points
    raw_elev = get_elevation_at_coords(dem, transform, xs, ys)
    smoothed = compute_smoothed_elevation(dem, 135, 55, 8)
    smooth_elev = get_elevation_at_coords(smoothed, transform, xs, ys)

    # === Filter valid stations ===
    valid = ~np.isnan(ppt_mean) & ~np.isnan(raw_elev) & ~np.isnan(smooth_elev)
    ppt = ppt_mean[valid]
    x_valid, y_valid = xs[valid], ys[valid]
    elev_raw = raw_elev[valid]
    elev_smooth = smooth_elev[valid]
    coords = np.column_stack([x_valid, y_valid])

    # === Train-Test Split ===
    ppt_train, ppt_test, raw_train, raw_test, smooth_train, smooth_test, coord_train, coord_test = train_test_split(
        ppt, elev_raw, elev_smooth, coords, test_size=0.4, random_state=42
    )

    # === 1. Raw Elevation Model ===
    model_raw = LinearRegression().fit(raw_train.reshape(-1, 1), ppt_train)
    pred_raw = model_raw.predict(raw_test.reshape(-1, 1))
    rmse_raw = np.sqrt(np.mean((ppt_test - pred_raw) ** 2))
    nse_raw = compute_nse(pred_raw, ppt_test)

    # === 2. Smoothed Elevation Model ===
    model_smooth = LinearRegression().fit(smooth_train.reshape(-1, 1), ppt_train)
    pred_smooth = model_smooth.predict(smooth_test.reshape(-1, 1))
    rmse_smooth = np.sqrt(np.mean((ppt_test - pred_smooth) ** 2))
    nse_smooth = compute_nse(pred_smooth, ppt_test)

    # === 3. Hybrid Model: raw for low elevation (<500m), smoothed for others ===
    pred_hybrid = np.full_like(ppt_test, np.nan)
    for lo, hi, use_raw in [(0, 500, True), (500, 1000, False), (1000, np.inf, False)]:
        mask = (raw_test >= lo) & (raw_test < hi)
        x_train = raw_train[(raw_train >= lo) & (raw_train < hi)] if use_raw else smooth_train[(raw_train >= lo) & (raw_train < hi)]
        y_train_local = ppt_train[(raw_train >= lo) & (raw_train < hi)]
        if len(x_train) < 3:
            continue
        model = LinearRegression().fit(x_train.reshape(-1, 1), y_train_local)
        x_test = raw_test[mask] if use_raw else smooth_test[mask]
        pred_hybrid[mask] = model.predict(x_test.reshape(-1, 1))

    rmse_hybrid = np.sqrt(np.mean((ppt_test - pred_hybrid) ** 2))
    nse_hybrid = compute_nse(pred_hybrid, ppt_test)

    # === 4. Stratified Regression + Kriging ===
    drift_train = np.full_like(ppt_train, np.nan)
    residuals_train = np.full_like(ppt_train, np.nan)

    # Fit regression models in each elevation band and compute residuals
    for lo, hi, use_raw in [(0, 500, True), (500, 1000, False), (1000, np.inf, False)]:
        mask = (raw_train >= lo) & (raw_train < hi)
        x = raw_train[mask] if use_raw else smooth_train[mask]
        y = ppt_train[mask]
        if np.sum(mask) < 3:
            continue
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        drift_train[mask] = model.predict(x.reshape(-1, 1))
        residuals_train[mask] = y - drift_train[mask]

    # Kriging on regression residuals
    ok_model = OrdinaryKriging(
        coord_train[:, 0], coord_train[:, 1], residuals_train,
        variogram_model="spherical", verbose=False, enable_plotting=False
    )

    drift_test = np.full_like(ppt_test, np.nan)
    res_kriged = np.full_like(ppt_test, np.nan)
    for lo, hi, use_raw in [(0, 500, True), (500, 1000, False), (1000, np.inf, False)]:
        mask_test = (raw_test >= lo) & (raw_test < hi)
        mask_train = (raw_train >= lo) & (raw_train < hi)
        if np.sum(mask_train) < 3:
            continue
        x_train = raw_train[mask_train] if use_raw else smooth_train[mask_train]
        y_train_local = ppt_train[mask_train]
        model = LinearRegression().fit(x_train.reshape(-1, 1), y_train_local)
        x_test = raw_test[mask_test] if use_raw else smooth_test[mask_test]
        drift_test[mask_test] = model.predict(x_test.reshape(-1, 1))

        x_coords = coord_test[mask_test, 0].astype(float)
        y_coords = coord_test[mask_test, 1].astype(float)
        z_kriged, _ = ok_model.execute("points", x_coords, y_coords)
        res_kriged[mask_test] = z_kriged

    stratified_pred = drift_test + res_kriged
    rmse_strat = np.sqrt(np.mean((ppt_test - stratified_pred) ** 2))
    nse_strat = compute_nse(stratified_pred, ppt_test)

    # === Print Comparison Results ===
    print("\n=== Model Comparison ===")
    print(f"Raw Elevation     : RMSE = {rmse_raw:.3f}, NSE = {nse_raw:.3f}")
    print(f"Smoothed Elevation: RMSE = {rmse_smooth:.3f}, NSE = {nse_smooth:.3f}")
    print(f"Hybrid Elevation  : RMSE = {rmse_hybrid:.3f}, NSE = {nse_hybrid:.3f}")
    print(f"Stratified Kriging: RMSE = {rmse_strat:.3f}, NSE = {nse_strat:.3f}")

    # === Plot Results ===
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    models = [
        ("Raw Elevation", pred_raw, rmse_raw, nse_raw, axs[0, 0]),
        ("Smoothed Elevation", pred_smooth, rmse_smooth, nse_smooth, axs[0, 1]),
        ("Hybrid Elevation", pred_hybrid, rmse_hybrid, nse_hybrid, axs[1, 0]),
        ("Stratified + Kriging", stratified_pred, rmse_strat, nse_strat, axs[1, 1])
    ]

    for title, pred, rmse, nse, ax in models:
        ax.scatter(ppt_test, pred, alpha=0.7, edgecolor='k')
        ax.plot([ppt_test.min(), ppt_test.max()], [ppt_test.min(), ppt_test.max()], 'r--')
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Observed Precipitation", fontsize=11)
        ax.set_ylabel("Predicted Precipitation", fontsize=11)
        ax.text(0.05, 0.95, f"RMSE = {rmse:.3f}\nNSE = {nse:.3f}",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))
        ax.grid(True)

    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    plt.show()


if __name__ == "__main__":
    main()
