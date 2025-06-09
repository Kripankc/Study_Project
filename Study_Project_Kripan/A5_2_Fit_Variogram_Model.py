import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from A1_Data_loader import DataLoader
from A2_elevation_smoothing import compute_smoothed_elevation, get_elevation_at_coords
from A5_1_Compare_Variograms import compute_empirical_variogram


def spherical_model(h, nugget, sill, range_):
    return np.where(
        h <= range_,
        nugget + sill * (1.5 * (h / range_) - 0.5 * (h / range_) ** 3),
        nugget + sill)


def exponential_model(h, nugget, sill, range_):
    return nugget + sill * (1 - np.exp(-3 * h / range_))


def gaussian_model(h, nugget, sill, range_):
    return nugget + sill * (1 - np.exp(-3 * (h / range_) ** 2))


def fit_model(lags, semis, model_func):
    initial = [np.min(semis), np.max(semis) - np.min(semis), np.max(lags) / 2]
    bounds = (0, [np.max(semis), np.max(semis), np.max(lags) * 2])
    params, _ = curve_fit(model_func, lags, semis, p0=initial, bounds=bounds)
    fitted = model_func(lags, *params)
    rmse = np.sqrt(mean_squared_error(semis, fitted))
    return params, fitted, rmse


def save_best_variogram(model_name, params, output_path="best_variogram_params.npy"):
    np.save(output_path, {"model": model_name, "params": params})


def main():
    dem, transform, _ = DataLoader.load_dem()
    stations = DataLoader.load_elevation_data()
    ppt_df = DataLoader.load_precipitation_data()

    ppt_df = ppt_df[stations.index]
    ppt_mean = ppt_df.mean().values
    xs, ys = stations["X"].values, stations["Y"].values

    smoothed = compute_smoothed_elevation(dem, 135, 55, 8)
    elev_s = get_elevation_at_coords(smoothed, transform, xs, ys)

    valid = ~np.isnan(elev_s) & ~np.isnan(ppt_mean)
    ppt_clean = ppt_mean[valid]
    elev_clean = elev_s[valid]
    xs_clean = xs[valid]
    ys_clean = ys[valid]

    model = LinearRegression().fit(elev_clean.reshape(-1, 1), ppt_clean)
    residuals = ppt_clean - model.predict(elev_clean.reshape(-1, 1))

    lags, semis = compute_empirical_variogram(xs_clean, ys_clean, residuals)

    models = {
        "Spherical": spherical_model,
        "Exponential": exponential_model,
        "Gaussian": gaussian_model
    }

    fine_lags = np.linspace(0, np.max(lags), 200)
    plt.figure(figsize=(10, 6))
    plt.plot(lags, semis, "ko", label="Empirical Variogram")

    best_model = None
    lowest_rmse = np.inf

    for name, func in models.items():
        params, fitted, rmse = fit_model(lags, semis, func)
        plt.plot(fine_lags, func(fine_lags, *params), label=f"{name} Fit (RMSE: {rmse:.4f})")
        print(f"{name} model: nugget={params[0]:.4f}, sill={params[1]:.4f}, range={params[2]:.2f}, RMSE={rmse:.4f}")

        if rmse < lowest_rmse:
            lowest_rmse = rmse
            best_model = (name, params)

    plt.xlabel("Lag Distance")
    plt.ylabel("Semivariance")
    plt.title("Fitted Variogram Models")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if best_model:
        print(f"\nBest Model: {best_model[0]}")
        print(f"Parameters: nugget={best_model[1][0]:.4f}, sill={best_model[1][1]:.4f}, range={best_model[1][2]:.2f}")
        save_best_variogram(best_model[0], best_model[1])


if __name__ == "__main__":
    main()
