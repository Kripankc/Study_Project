import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from A1_Data_loader import DataLoader
from A2_elevation_smoothing import compute_smoothed_elevation, get_elevation_at_coords

GRID_RES = 1000  # Grid resolution (not used in this script but kept for consistency)


def load_best_variogram(path="best_variogram_params.npy"):
    """
    Load the best variogram model and parameters from file.
    """
    data = np.load(path, allow_pickle=True).item()
    return data["model"].lower(), data["params"]


def compute_nse(predicted, observed):
    """
    Compute the Nash-Sutcliffe Efficiency coefficient.
    """
    mean_obs = np.mean(observed)
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - mean_obs) ** 2)
    return 1 - numerator / denominator if denominator != 0 else -np.inf


def main():
    # Load DEM, station coordinates, and precipitation data
    dem, transform, _ = DataLoader.load_dem()
    stations = DataLoader.load_elevation_data()
    ppt_df = DataLoader.load_precipitation_data()

    # Ensure alignment of station and precipitation records
    ppt_df = ppt_df[stations.index]
    ppt_mean = ppt_df.mean().values
    xs = stations["X"].values
    ys = stations["Y"].values

    # Compute raw and smoothed elevation at station locations
    smoothed = compute_smoothed_elevation(dem, 135, 55, 8)
    raw_elev = get_elevation_at_coords(dem, transform, xs, ys)
    smooth_elev = get_elevation_at_coords(smoothed, transform, xs, ys)

    # Filter valid stations with non-NaN data
    valid = ~np.isnan(ppt_mean) & ~np.isnan(raw_elev) & ~np.isnan(smooth_elev)
    x_valid = xs[valid].astype(float)
    y_valid = ys[valid].astype(float)
    z_valid = ppt_mean[valid].astype(float)
    elev_raw = raw_elev[valid]
    elev_smooth = smooth_elev[valid]

    # Fit linear regression models
    model_raw = LinearRegression().fit(elev_raw.reshape(-1, 1), z_valid)
    pred_raw = model_raw.predict(elev_raw.reshape(-1, 1))

    model_smooth = LinearRegression().fit(elev_smooth.reshape(-1, 1), z_valid)
    pred_smooth = model_smooth.predict(elev_smooth.reshape(-1, 1))

    # Calculate RMSE and NSE for both cases
    rmse_raw = np.sqrt(np.mean((pred_raw - z_valid) ** 2))
    rmse_smooth = np.sqrt(np.mean((pred_smooth - z_valid) ** 2))
    nse_raw = compute_nse(pred_raw, z_valid)
    nse_smooth = compute_nse(pred_smooth, z_valid)

    # Plot observed vs predicted precipitation
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

    # Print evaluation metrics
    print("\n=== Comparison Results ===")
    print(f"RMSE (Raw Residuals):     {rmse_raw:.4f}")
    print(f"RMSE (Smoothed Residuals): {rmse_smooth:.4f}")
    print(f"NSE (Raw Residuals):      {nse_raw:.4f}")
    print(f"NSE (Smoothed Residuals): {nse_smooth:.4f}")


if __name__ == "__main__":
    main()
