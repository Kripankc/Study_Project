import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from A1_Data_loader import DataLoader
from A3_finding_best import compute_smoothed_elevation, get_elevation_at_coords


def plot_summary():
    # Load all required data
    dem, transform, _ = DataLoader.load_dem()
    stations = DataLoader.load_elevation_data()
    ppt_df = DataLoader.load_precipitation_data()

    if dem is None or stations is None or ppt_df is None:
        print("Missing data files.")
        return

    # Align data
    stations = stations.loc[stations.index.intersection(ppt_df.columns)]
    ppt_df = ppt_df[stations.index]
    ppt_mean = ppt_df.mean()

    xs = stations["X"].values
    ys = stations["Y"].values
    elev_raw = stations["Elevation"].values
    ppt = ppt_mean.values

    # Compute smoothed elevation using best parameters (replace if needed)
    smoothed_dem = compute_smoothed_elevation(dem, transform, direction_deg=135, h_s_km=55, width_pixels=8)
    elev_smooth = get_elevation_at_coords(smoothed_dem, transform, xs, ys)

    # Validity mask
    valid = ~np.isnan(elev_raw) & ~np.isnan(ppt) & ~np.isnan(elev_smooth)
    xs, ys, ppt, elev_raw, elev_smooth = xs[valid], ys[valid], ppt[valid], elev_raw[valid], elev_smooth[valid]

    # Correlation calculations
    r_raw = pearsonr(elev_raw, ppt)[0]
    r_smooth = pearsonr(elev_smooth, ppt)[0]

    # Plot setup
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()

    # 1. Raw elevation vs ppt
    axs[0].scatter(elev_raw, ppt, alpha=0.6, color='orange')
    axs[0].set_title(f"Raw Elevation vs Precipitation (r = {r_raw:.2f})")
    axs[0].set_xlabel("Elevation (m)")
    axs[0].set_ylabel("Precipitation (mm)")

    # 2. Smoothed elevation vs ppt
    axs[1].scatter(elev_smooth, ppt, alpha=0.6, color='green')
    axs[1].set_title(f"Smoothed Elevation (135°, 55km) vs Precip (r = {r_smooth:.2f})")
    axs[1].set_xlabel("Smoothed Elevation (m)")
    axs[1].set_ylabel("Precipitation (mm)")

    # 3. Semivariogram
    coords = np.column_stack((xs, ys))
    dists = pdist(coords)
    diffs = pdist(ppt.reshape(-1, 1))
    semi_vars = 0.5 * (diffs ** 2)
    bins = np.linspace(0, np.max(dists), 50)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    gamma_vals = [np.mean(semi_vars[(dists >= b0) & (dists < b1)])
                  if np.any((dists >= b0) & (dists < b1)) else np.nan
                  for b0, b1 in zip(bins[:-1], bins[1:])]
    axs[2].plot(bin_centers, gamma_vals, 'o-', color='blue')
    axs[2].set_title("Empirical Semivariogram")
    axs[2].set_xlabel("Lag Distance")
    axs[2].set_ylabel("Semivariance")

    # 4. Histogram of elevation
    axs[3].hist(elev_raw, bins=20, color='skyblue', edgecolor='black')
    axs[3].set_title("Histogram of Station Elevation")
    axs[3].set_xlabel("Elevation (m)")
    axs[3].set_ylabel("Frequency")

    # 5. Boxplot of mean ppt
    axs[4].boxplot(ppt, vert=True)
    axs[4].set_title("Boxplot of Mean Precipitation")
    axs[4].set_ylabel("Precipitation (mm)")

    # 6. Blank or additional plot
    axs[5].axis("off")
    axs[5].text(0.5, 0.5, "Add your own plot here", ha="center", va="center", fontsize=12, color="gray")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_summary()
