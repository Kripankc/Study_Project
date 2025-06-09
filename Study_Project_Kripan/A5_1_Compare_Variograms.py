from A1_Data_loader import DataLoader
from A2_elevation_smoothing import compute_smoothed_elevation, get_elevation_at_coords
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np


def compute_empirical_variogram(x, y, residuals, n_bins=15, max_dist=None):
    coords = np.column_stack((x, y))
    dists = pairwise_distances(coords)
    diffs = np.subtract.outer(residuals, residuals)
    semi = 0.5 * (diffs ** 2)

    if max_dist is None:
        max_dist = np.percentile(dists, 90)

    bins = np.linspace(0, max_dist, n_bins + 1)
    lag_distances = []
    semivariances = []

    for i in range(n_bins):
        mask = (dists >= bins[i]) & (dists < bins[i + 1])
        if np.any(mask):
            lag_distances.append(np.mean(dists[mask]))
            semivariances.append(np.mean(semi[mask]))

    return np.array(lag_distances), np.array(semivariances)


def compute_residuals(elev, coords, ppt):
    X_train, _, y_train, _, elev_train, _ = train_test_split(
        coords, ppt, elev, test_size=0.4, random_state=42
    )
    model = LinearRegression().fit(elev_train.reshape(-1, 1), y_train)
    predicted = model.predict(elev_train.reshape(-1, 1))
    residuals = y_train - predicted
    return X_train[:, 0], X_train[:, 1], residuals


def main():
    dem, transform, _ = DataLoader.load_dem()
    stations = DataLoader.load_elevation_data()
    ppt_df = DataLoader.load_precipitation_data()

    stations = stations.loc[stations.index.intersection(ppt_df.columns)]
    ppt_df = ppt_df[stations.index]
    ppt = ppt_df.mean().values
    xs = stations["X"].values
    ys = stations["Y"].values
    raw_elev = stations["Elevation"].values

    coords = np.column_stack((xs, ys))
    mask = ~np.isnan(xs) & ~np.isnan(ys) & ~np.isnan(ppt) & ~np.isnan(raw_elev)
    coords, ppt, raw_elev = coords[mask], ppt[mask], raw_elev[mask]

    smoothed_dem = compute_smoothed_elevation(dem, direction_deg=135, h_s_pixels=55, width_pixels=8)
    smoothed_elev = get_elevation_at_coords(smoothed_dem, transform, coords[:, 0], coords[:, 1])
    smoothed_elev = smoothed_elev[mask]

    x_r, y_r, res_raw = compute_residuals(raw_elev, coords, ppt)
    x_s, y_s, res_smooth = compute_residuals(smoothed_elev, coords, ppt)

    lags_raw, semis_raw = compute_empirical_variogram(x_r, y_r, res_raw)
    lags_smooth, semis_smooth = compute_empirical_variogram(x_s, y_s, res_smooth)

    plt.figure(figsize=(8, 5))
    plt.plot(lags_raw, semis_raw, 'o-', label='Raw Elevation Residuals')
    plt.plot(lags_smooth, semis_smooth, 'o-', label='Smoothed Elevation Residuals')
    plt.xlabel("Lag Distance")
    plt.ylabel("Semivariance")
    plt.title("Empirical Variogram Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
