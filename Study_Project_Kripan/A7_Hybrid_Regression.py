import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from A1_Data_loader import DataLoader
from A2_elevation_smoothing import compute_smoothed_elevation, get_elevation_at_coords


def compute_nse(predicted, observed):
    """
    Compute the Nash–Sutcliffe Efficiency (NSE) coefficient.
    Measures predictive skill of a model (1 = perfect, 0 = mean prediction).
    """
    mean_obs = np.mean(observed)
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - mean_obs) ** 2)
    return 1 - numerator / denominator if denominator != 0 else -np.inf


def main():
    # === Load precipitation and station metadata ===
    dem, transform, _ = DataLoader.load_dem()
    stations = DataLoader.load_elevation_data()
    ppt_df = DataLoader.load_precipitation_data()

    ppt_df = ppt_df[stations.index]
    ppt_mean = ppt_df.mean().values
    xs = stations["X"].values
    ys = stations["Y"].values

    # === Extract elevation values at station coordinates ===
    raw_elev = get_elevation_at_coords(dem, transform, xs, ys)
    smoothed_dem = compute_smoothed_elevation(dem, 135, 55, 8)  # best smoothing parameters
    smooth_elev = get_elevation_at_coords(smoothed_dem, transform, xs, ys)

    # === Filter valid data points ===
    valid = ~np.isnan(ppt_mean) & ~np.isnan(raw_elev) & ~np.isnan(smooth_elev)
    ppt = ppt_mean[valid]
    elev_raw = raw_elev[valid]
    elev_smooth = smooth_elev[valid]

    # === Create hybrid elevation: ===
    # Raw elevation is more accurate in lowlands (<500 m) where topography is simple.
    # Smoothed elevation is better in complex terrains (≥500 m) to reduce noise.
    elev_hybrid = np.where(elev_raw < 500, elev_raw, elev_smooth)

    # === Fit linear regression models for each elevation version ===
    model_raw = LinearRegression().fit(elev_raw.reshape(-1, 1), ppt)
    pred_raw = model_raw.predict(elev_raw.reshape(-1, 1))

    model_smooth = LinearRegression().fit(elev_smooth.reshape(-1, 1), ppt)
    pred_smooth = model_smooth.predict(elev_smooth.reshape(-1, 1))

    model_hybrid = LinearRegression().fit(elev_hybrid.reshape(-1, 1), ppt)
    pred_hybrid = model_hybrid.predict(elev_hybrid.reshape(-1, 1))

    # === Report RMSE and NSE for each model ===
    def print_stats(name, y_pred):
        rmse = np.sqrt(np.mean((y_pred - ppt) ** 2))
        nse = compute_nse(y_pred, ppt)
        print(f"{name:12}: RMSE = {rmse:.3f}, NSE = {nse:.3f}")
        return rmse, nse

    print("=== Model Comparison ===")
    r_raw, n_raw = print_stats("Raw Elevation", pred_raw)
    r_smooth, n_smooth = print_stats("Smoothed Elevation", pred_smooth)
    r_hybrid, n_hybrid = print_stats("Hybrid Elevation", pred_hybrid)

    # === Scatter plots for observed vs predicted precipitation ===
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    axs[0].scatter(ppt, pred_raw, alpha=0.6, color="orange")
    axs[0].plot([ppt.min(), ppt.max()], [ppt.min(), ppt.max()], 'k--')
    axs[0].set_title(f"Raw (RMSE={r_raw:.2f}, NSE={n_raw:.3f})")
    axs[0].set_xlabel("Observed")
    axs[0].set_ylabel("Predicted")

    axs[1].scatter(ppt, pred_smooth, alpha=0.6, color="green")
    axs[1].plot([ppt.min(), ppt.max()], [ppt.min(), ppt.max()], 'k--')
    axs[1].set_title(f"Smoothed (RMSE={r_smooth:.2f}, NSE={n_smooth:.3f})")
    axs[1].set_xlabel("Observed")

    # Third plot includes raw elevation color as context
    sc = axs[2].scatter(ppt, pred_hybrid, c=elev_raw, cmap="viridis", alpha=0.7)
    axs[2].plot([ppt.min(), ppt.max()], [ppt.min(), ppt.max()], 'k--')
    axs[2].set_title(f"Hybrid (RMSE={r_hybrid:.2f}, NSE={n_hybrid:.3f})")
    axs[2].set_xlabel("Observed")

    # Add colorbar indicating elevation
    cbar = fig.colorbar(sc, ax=axs[2])
    cbar.set_label("Raw Elevation (m)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
