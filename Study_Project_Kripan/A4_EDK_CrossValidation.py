import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pykrige.ok import OrdinaryKriging

from A1_Data_loader import DataLoader
from A2_elevation_smoothing import compute_smoothed_elevation, get_elevation_at_coords


def evaluate(y_true, y_pred):
    """
    Compute RMSE and Nash-Sutcliffe Efficiency (NSE).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nse = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return rmse, nse


def run_edk(xs_train, ys_train, ppt_train, drift_train, xs_test, ys_test, drift_test):
    """
    Run External Drift Kriging using elevation as drift.
    """
    xs_train = np.asarray(xs_train, dtype=np.float64)
    ys_train = np.asarray(ys_train, dtype=np.float64)
    xs_test = np.asarray(xs_test, dtype=np.float64)
    ys_test = np.asarray(ys_test, dtype=np.float64)

    drift_train = drift_train.reshape(-1, 1)
    drift_test = drift_test.reshape(-1, 1)

    regression = LinearRegression().fit(drift_train, ppt_train)
    residuals = ppt_train - regression.predict(drift_train)

    ok_model = OrdinaryKriging(xs_train, ys_train, residuals, variogram_model="linear", verbose=False)
    predicted_residuals, _ = ok_model.execute("points", xs_test, ys_test)

    ppt_pred = regression.predict(drift_test) + predicted_residuals
    return ppt_pred


def plot_predictions(y_test, predictions_dict, evaluation_dict):
    """
    Plot observed vs predicted precipitation with RMSE and NSE annotations.
    """
    num_plots = len(predictions_dict)
    fig, axs = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5), sharey=True)

    if num_plots == 1:
        axs = [axs]

    min_val = min(np.min(y_test), *(np.min(p) for p in predictions_dict.values()))
    max_val = max(np.max(y_test), *(np.max(p) for p in predictions_dict.values()))

    for ax, (label, y_pred) in zip(axs, predictions_dict.items()):
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([min_val, max_val], [min_val, max_val], 'k--')

        rmse, nse = evaluation_dict[label]
        ax.set_title(f"{label}\nRMSE = {rmse:.2f}, NSE = {nse:.3f}")
        ax.set_xlabel("Observed PPT")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    axs[0].set_ylabel("Predicted PPT")
    plt.tight_layout()
    plt.show()


def build_drift_model(xs, ys, ppt, elevation):
    """Fits a linear regression model and returns residuals and fitted model."""
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(elevation.reshape(-1, 1), ppt)
    residuals = ppt - model.predict(elevation.reshape(-1, 1))
    return model, residuals


def main():
    # Load and prepare data
    dem, transform, _ = DataLoader.load_dem()
    stations = DataLoader.load_elevation_data()
    ppt_df = DataLoader.load_precipitation_data()

    stations = stations.loc[stations.index.intersection(ppt_df.columns)]
    ppt_df = ppt_df[stations.index]
    ppt_mean = ppt_df.mean()

    xs = stations["X"].values
    ys = stations["Y"].values
    ppt = ppt_mean.values
    elev_raw = stations["Elevation"].values

    # Smooth elevation configs
    top_configs = [
        {"direction": 135, "distance_px": 55, "width_px": 8},
        {"direction": 135, "distance_px": 65, "width_px": 8},
        {"direction": 135, "distance_px": 45, "width_px": 8}
    ]

    smoothed_elevations = []
    for config in top_configs:
        smoothed_dem = compute_smoothed_elevation(
            dem, config["direction"], config["distance_px"], config["width_px"]
        )
        elev = get_elevation_at_coords(smoothed_dem, transform, xs, ys)
        smoothed_elevations.append(elev)

    # Filter valid stations
    valid = ~np.isnan(xs) & ~np.isnan(ys) & ~np.isnan(ppt) & ~np.isnan(elev_raw)
    for elev in smoothed_elevations:
        valid &= ~np.isnan(elev)

    xs, ys, ppt = xs[valid], ys[valid], ppt[valid]
    elev_raw = elev_raw[valid]
    smoothed_elevations = [e[valid] for e in smoothed_elevations]

    # Split
    X = np.column_stack((xs, ys))
    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        X, ppt, elev_raw, test_size=0.4, random_state=42
    )

    evaluation_scores = {}  # ← Add this line
    predictions = {}

    # Evaluate raw elevation
    pred_raw = run_edk(X_train[:, 0], X_train[:, 1], y_train, raw_train,
                       X_test[:, 0], X_test[:, 1], raw_test)
    rmse_r, nse_r = evaluate(y_test, pred_raw)
    evaluation_scores["Raw Elevation"] = (rmse_r, nse_r)
    predictions["Raw Elevation"] = pred_raw
    print("Raw Elevation as Drift:")
    print(f"  RMSE: {rmse_r:.2f}, NSE: {nse_r:.3f}")

    # Evaluate smoothed elevations
    for config, elev in zip(top_configs, smoothed_elevations):
        smooth_train, smooth_test = train_test_split(elev, test_size=0.4, random_state=42)
        pred = run_edk(X_train[:, 0], X_train[:, 1], y_train, smooth_train,
                       X_test[:, 0], X_test[:, 1], smooth_test)
        label = f"Smoothed {config['direction']}° {config['distance_px']}px {config['width_px']}px"
        rmse, nse = evaluate(y_test, pred)
        evaluation_scores[label] = (rmse, nse)
        predictions[label] = pred
        print(f"{label}:\n  RMSE: {rmse:.2f}, NSE: {nse:.3f}")

    plot_predictions(y_test, predictions, evaluation_scores)


if __name__ == "__main__":
    main()
