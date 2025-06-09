import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from A1_Data_loader import DataLoader
from A2_elevation_smoothing import compute_smoothed_elevation, get_elevation_at_coords


def compute_nse(predicted, observed):
    mean_obs = np.mean(observed)
    numerator = np.sum((observed - predicted) ** 2)
    denominator = np.sum((observed - mean_obs) ** 2)
    return 1 - numerator / denominator if denominator != 0 else -np.inf


def main():
    dem, transform, _ = DataLoader.load_dem()
    stations = DataLoader.load_elevation_data()
    ppt_df = DataLoader.load_precipitation_data()

    ppt_df = ppt_df[stations.index]
    ppt_mean = ppt_df.mean().values
    xs = stations["X"].values
    ys = stations["Y"].values

    raw_elev = get_elevation_at_coords(dem, transform, xs, ys)
    smoothed_dem = compute_smoothed_elevation(dem, 135, 55, 8)
    smooth_elev = get_elevation_at_coords(smoothed_dem, transform, xs, ys)

    valid = ~np.isnan(ppt_mean) & ~np.isnan(raw_elev) & ~np.isnan(smooth_elev)
    ppt = ppt_mean[valid]
    elev_raw = raw_elev[valid]
    elev_smooth = smooth_elev[valid]
    hybrid_elev = np.where(elev_raw < 500, elev_raw, elev_smooth)

    model_raw = LinearRegression().fit(elev_raw.reshape(-1, 1), ppt)
    pred_raw = model_raw.predict(elev_raw.reshape(-1, 1))

    model_smooth = LinearRegression().fit(elev_smooth.reshape(-1, 1), ppt)
    pred_smooth = model_smooth.predict(elev_smooth.reshape(-1, 1))

    model_hybrid = LinearRegression().fit(hybrid_elev.reshape(-1, 1), ppt)
    pred_hybrid = model_hybrid.predict(hybrid_elev.reshape(-1, 1))

    pred_stratified = np.full_like(ppt, np.nan)

    for label, lo, hi, use_raw in [("Low", 0, 500, True), ("Mid", 500, 1000, False), ("High", 1000, np.inf, False)]:
        band_mask = (elev_raw >= lo) & (elev_raw < hi)
        x_band = elev_raw[band_mask] if use_raw else elev_smooth[band_mask]
        y_band = ppt[band_mask]
        model = LinearRegression().fit(x_band.reshape(-1, 1), y_band)
        pred = model.predict(x_band.reshape(-1, 1))
        pred_stratified[band_mask] = pred

    def print_stats(label, pred):
        rmse = np.sqrt(np.mean((pred - ppt) ** 2))
        nse = compute_nse(pred, ppt)
        print(f"{label:18}: RMSE = {rmse:.3f}, NSE = {nse:.3f}")
        return rmse, nse

    print("=== Model Comparison ===")
    r1, n1 = print_stats("Raw Elevation", pred_raw)
    r2, n2 = print_stats("Smoothed Elevation", pred_smooth)
    r3, n3 = print_stats("Hybrid Elevation", pred_hybrid)
    r4, n4 = print_stats("Stratified Linear", pred_stratified)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    all_preds = [pred_raw, pred_smooth, pred_hybrid, pred_stratified]
    titles = [
        f"Raw (RMSE={r1:.2f}, NSE={n1:.3f})",
        f"Smoothed (RMSE={r2:.2f}, NSE={n2:.3f})",
        f"Hybrid (RMSE={r3:.2f}, NSE={n3:.3f})",
        f"Stratified (RMSE={r4:.2f}, NSE={n4:.3f})"
    ]

    for ax, pred, title in zip(axs.flatten(), all_preds, titles):
        ax.scatter(ppt, pred, alpha=0.6)
        ax.plot([ppt.min(), ppt.max()], [ppt.min(), ppt.max()], 'k--')
        ax.set_xlabel("Observed Precipitation")
        ax.set_ylabel("Predicted Precipitation")
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
