import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from A1_Data_loader import DataLoader
from A2_elevation_smoothing import compute_smoothed_elevation, get_elevation_at_coords


def prepare_data():
    """
    Loads DEM, station coordinates, and precipitation data.

    Returns:
        dem (np.ndarray): DEM raster.
        transform (Affine): Affine transform for the DEM.
        stations (DataFrame): Station metadata including X, Y, Elevation.
        ppt_mean (np.ndarray): Mean precipitation per station.
        xs, ys (np.ndarray): Station coordinate arrays.
    """
    dem, transform, _ = DataLoader.load_dem()
    stations = DataLoader.load_elevation_data()
    ppt_df = DataLoader.load_precipitation_data()

    if dem is None or stations is None or ppt_df is None:
        raise RuntimeError("One or more required data sources could not be loaded.")

    # Match station IDs between elevation and precipitation data
    stations = stations.loc[stations.index.intersection(ppt_df.columns)]
    ppt_df = ppt_df[stations.index]
    ppt_mean = ppt_df.mean()

    xs = stations["X"].values
    ys = stations["Y"].values

    return dem, transform, stations, ppt_mean.values, xs, ys


def evaluate_smoothing_combinations(dem, transform, xs, ys, ppt, directions, distances_px, widths_px):
    """
    Tests multiple directional smoothing configurations and evaluates them
    using Pearson correlation between smoothed elevation and precipitation.

    Args:
        dem (np.ndarray): Digital elevation model.
        transform (Affine): Affine transform for mapping.
        xs, ys (np.ndarray): Station coordinates.
        ppt (np.ndarray): Mean precipitation at stations.
        directions (list): List of azimuth angles (degrees).
        distances_px (list): Smoothing distances in pixels.
        widths_px (list): Lateral widths in pixels.

    Returns:
        DataFrame: Ranked smoothing configurations with Pearson correlations.
    """
    results = []

    for width_px in widths_px:
        for dist_px in distances_px:
            for direction in directions:
                smoothed_dem = compute_smoothed_elevation(dem, direction, dist_px, width_px)
                elev = get_elevation_at_coords(smoothed_dem, transform, xs, ys)
                mask = ~np.isnan(elev) & ~np.isnan(ppt)

                if np.sum(mask) >= 2 and np.std(elev[mask]) > 0 and np.std(ppt[mask]) > 0:
                    corr = pearsonr(elev[mask], ppt[mask])[0]
                    results.append((direction, dist_px, width_px, corr))

    df = pd.DataFrame(results, columns=["Direction (°)", "Distance (px)", "Width (px)", "Correlation"])
    return df.sort_values("Correlation", ascending=False)


def main():
    dem, transform, stations, ppt, xs, ys = prepare_data()

    # Define parameter ranges
    directions = [0, 45, 90, 135, 180, 225, 270, 315]
    distances_px = [20, 25, 35, 45, 55, 65, 75]
    widths_px = [0, 2, 4, 8]

    results_df = evaluate_smoothing_combinations(dem, transform, xs, ys, ppt, directions, distances_px, widths_px)

    print("\nTop 10 Smoothing Combinations by Correlation:")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
