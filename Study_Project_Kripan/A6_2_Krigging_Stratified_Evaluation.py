import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from A1_Data_loader import DataLoader
from A2_elevation_smoothing import compute_smoothed_elevation, get_elevation_at_coords

GRID_RES = 1000  # DEM resolution in meters


def load_best_variogram(path="best_variogram_params.npy"):
    """
    Load the best variogram model and parameters from a saved .npy file.
    """
    data = np.load(path, allow_pickle=True).item()
    return data["model"].lower(), data["params"]


def compute_nse(predicted, observed):
    """
    Compute Nash-Sutcliffe Efficiency (NSE) between predicted and observed values.
    """
    mean_obs = np.mean(observed)
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - mean_obs) ** 2)
    return 1 - numerator / denominator if denominator != 0 else -np.inf


def stratified_metrics(elevations, observed, pred_raw, pred_smooth):
    """
    Compute RMSE and NSE for different elevation bands.
    """
    bins = [0, 500, 1000, np.inf]
    labels = ["Low (<500m)", "Mid (500–1000m)", "High (>1000m)"]
    results = []

    for i in range(3):
        mask = (elevations >= bins[i]) & (elevations < bins[i + 1])
        if np.sum(mask) < 10:
            continue  # Skip small sample groups

        obs = observed[mask]
        raw = pred_raw[mask]
        smooth = pred_smooth[mask]

        rmse_r = np.sqrt(np.mean((raw - obs) ** 2))
        rmse_s = np.sqrt(np.mean((smooth - obs) ** 2))
        nse_r = compute_nse(raw, obs)
        nse_s = compute_nse(smooth, obs)

        results.append((labels[i], len(obs), rmse_r, nse_r, rmse_s, nse_s))

    return results


def plot_stratified_bar_chart(results):
    """
    Plot RMSE and NSE per elevation band as bar charts.
    """
    labels = [r[0] for r in results]
    rmse_raw = [r[2] for r in results]
    nse_raw = [r[3] for r in results]
    rmse_smooth = [r[4] for r in results]
    nse_smooth = [r[5] for r in results]
    x = np.arange(len(labels))
    width = 0.35

    # RMSE Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x - width / 2, rmse_raw, width, label='Raw', color='orange')
    ax1.bar(x + width / 2, rmse_smooth, width, label='Smoothed', color='green')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE by Elevation Band')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True)
    plt.tight_layout()
    plt.show()

    # NSE Plot
    fig, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(x - width / 2, nse_raw, width, label='Raw', color='orange')
    ax2.bar(x + width / 2, nse_smooth, width, label='Smoothed', color='green')
    ax2.set_ylabel('NSE')
    ax2.set_title('NSE by Elevation Band')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # === Load data ===
    dem, transform, _ = DataLoader.load_dem()
    stations = DataLoader.load_elevation_data()
    ppt_df = DataLoader.load_precipitation_data()

    ppt_df = ppt_df[stations.index]
    ppt_mean = ppt_df.mean().values
    xs, ys = stations["X"].values, stations["Y"].values

    # === Compute elevations ===
    smoothed = compute_smoothed_elevation(dem, 135, 55, 8)
    raw_elev = get_elevation_at_coords(dem, transform, xs, ys)
    smooth_elev = get_elevation_at_coords(smoothed, transform, xs, ys)

    # === Filter valid records ===
    valid = ~np.isnan(ppt_mean) & ~np.isnan(raw_elev) & ~np.isnan(smooth_elev)
    z_valid = ppt_mean[valid].astype(float)
    elev_raw = raw_elev[valid]
    elev_smooth = smooth_elev[valid]

    # === Fit regression models ===
    model_raw = LinearRegression().fit(elev_raw.reshape(-1, 1), z_valid)
    pred_raw = model_raw.predict(elev_raw.reshape(-1, 1))

    model_smooth = LinearRegression().fit(elev_smooth.reshape(-1, 1), z_valid)
    pred_smooth = model_smooth.predict(elev_smooth.reshape(-1, 1))

    # === Overall performance ===
    rmse_raw = np.sqrt(np.mean((pred_raw - z_valid) ** 2))
    rmse_smooth = np.sqrt(np.mean((pred_smooth - z_valid) ** 2))
    nse_raw = compute_nse(pred_raw, z_valid)
    nse_smooth = compute_nse(pred_smooth, z_valid)

    print("=== Overall Model Performance ===")
    print(f"Raw Elevation:     RMSE = {rmse_raw:.3f}, NSE = {nse_raw:.3f}")
    print(f"Smoothed Elevation: RMSE = {rmse_smooth:.3f}, NSE = {nse_smooth:.3f}\n")

    # === Plot scatter comparison ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].scatter(z_valid, pred_raw, alpha=0.6, color="orange")
    axs[0].plot([z_valid.min(), z_valid.max()], [z_valid.min(), z_valid.max()], 'k--')
    axs[0].set_title(f"Raw Elevation (RMSE={rmse_raw:.2f}, NSE={nse_raw:.3f})")
    axs[0].set_xlabel("Observed Precipitation")
    axs[0].set_ylabel("Predicted Precipitation")

    axs[1].scatter(z_valid, pred_smooth, alpha=0.6, color="green")
    axs[1].plot([z_valid.min(), z_valid.max()], [z_valid.min(), z_valid.max()], 'k--')
    axs[1].set_title(f"Smoothed Elevation (RMSE={rmse_smooth:.2f}, NSE={nse_smooth:.3f})")
    axs[1].set_xlabel("Observed Precipitation")
    axs[1].set_ylabel("Predicted Precipitation")

    plt.tight_layout()
    plt.show()

    # === Plot elevation distribution ===
    plt.figure(figsize=(8, 4))
    plt.hist(elev_raw, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("Station Elevation (m)")
    plt.ylabel("Frequency")
    plt.title("Elevation Distribution of Stations")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Stratified Evaluation ===
    print("\n=== Stratified Evaluation by Elevation Class ===")
    stratified = stratified_metrics(elev_raw, z_valid, pred_raw, pred_smooth)

    for label, count, rmse_r, nse_r, rmse_s, nse_s in stratified:
        print(f"{label} ({count} stations)")
        print(f"  Raw:     RMSE={rmse_r:.3f}, NSE={nse_r:.3f}")
        print(f"  Smoothed: RMSE={rmse_s:.3f}, NSE={nse_s:.3f}\n")

    plot_stratified_bar_chart(stratified)


if __name__ == "__main__":
    main()
