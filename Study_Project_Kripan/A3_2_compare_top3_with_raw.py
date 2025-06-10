import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from A1_Data_loader import DataLoader
from A3_1_finding_best import compute_smoothed_elevation, get_elevation_at_coords


def plot_raw_vs_top3_smoothing():
    """
    Compare correlation between raw elevation and precipitation vs. top 3 directional
    smoothing combinations. Displays scatter plots and Pearson r values.
    """
    # Load DEM and station data
    dem, transform, _ = DataLoader.load_dem()
    stations = DataLoader.load_elevation_data()
    ppt_df = DataLoader.load_precipitation_data()

    # Match station data with precipitation
    stations = stations.loc[stations.index.intersection(ppt_df.columns)]
    ppt_df = ppt_df[stations.index]
    ppt_mean = ppt_df.mean()

    xs = stations["X"].values
    ys = stations["Y"].values
    ppt = ppt_mean.values
    elev_raw = stations["Elevation"].values

    # Top 3 smoothing configurations (based on correlation results)
    top_3 = [
        (135, 55, 8),  # (direction_deg, h_s_km, width_px)
        (90, 45, 4),
        (180, 65, 2)
    ]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    # --- Plot Raw Elevation ---
    mask = ~np.isnan(elev_raw) & ~np.isnan(ppt)
    r_raw = pearsonr(elev_raw[mask], ppt[mask])[0]
    axs[0].scatter(elev_raw[mask], ppt[mask], alpha=0.6, color="orange")
    axs[0].set_title(f"Raw Elevation vs Precipitation (r = {r_raw:.2f})")
    axs[0].set_xlabel("Elevation (m)")
    axs[0].set_ylabel("Precipitation (mm)")

    # --- Plot Smoothed Elevation using top 3 settings ---
    for i, (deg, dist_km, width_px) in enumerate(top_3, start=1):
        smoothed_dem = compute_smoothed_elevation(dem, deg, dist_km, width_px)
        elev_smooth = get_elevation_at_coords(smoothed_dem, transform, xs, ys)
        mask = ~np.isnan(elev_smooth) & ~np.isnan(ppt)

        if np.sum(mask) < 2:
            axs[i].set_title("Insufficient data for correlation")
            continue

        r = pearsonr(elev_smooth[mask], ppt[mask])[0]
        axs[i].scatter(elev_smooth[mask], ppt[mask], alpha=0.6, color="teal")
        axs[i].set_title(f"Smoothed ({deg}°, {dist_km}km, w={width_px}) (r = {r:.2f})")
        axs[i].set_xlabel("Smoothed Elevation (m)")
        axs[i].set_ylabel("Precipitation (mm)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_raw_vs_top3_smoothing()
