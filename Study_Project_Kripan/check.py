import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pykrige.ok import OrdinaryKriging


def compute_nse(predicted, observed):
    mean_obs = np.mean(observed)
    return 1 - np.sum((observed - predicted) ** 2) / np.sum((observed - mean_obs) ** 2)


def stratified_regression_prediction(train_elev, train_ppt, test_elev, bands):
    pred = np.full_like(test_elev, np.nan)
    for lo, hi, use_raw in bands:
        train_mask = (train_elev >= lo) & (train_elev < hi)
        test_mask = (test_elev >= lo) & (test_elev < hi)
        x_train = train_elev[train_mask]
        y_train = train_ppt[train_mask]
        if len(x_train) < 3:
            continue
        model = LinearRegression().fit(x_train.reshape(-1, 1), y_train)
        x_test = test_elev[test_mask]
        pred[test_mask] = model.predict(x_test.reshape(-1, 1))
    return pred


# === Generate mock data ===
np.random.seed(42)
n_points = 100
xs = np.random.uniform(0, 1000, n_points)
ys = np.random.uniform(0, 1000, n_points)
elev_raw = np.random.uniform(0, 1500, n_points)
elev_smooth = elev_raw + np.random.normal(0, 50, n_points)
ppt = 0.01 * elev_raw + np.random.normal(0, 5, n_points)
coords = np.column_stack((xs, ys))

# === 60/40 split ===
ppt_train, ppt_test, raw_train, raw_test, smooth_train, smooth_test, coord_train, coord_test = train_test_split(
    ppt, elev_raw, elev_smooth, coords, test_size=0.4, random_state=42
)

# === Stratified regression (drift) ===
bands = [(0, 500, True), (500, 1000, False), (1000, np.inf, False)]
drift_train = stratified_regression_prediction(raw_train, ppt_train, raw_train, bands)
residuals_train = ppt_train - drift_train

# === Kriging residuals ===
ok_model = OrdinaryKriging(
    coord_train[:, 0], coord_train[:, 1], residuals_train,
    variogram_model="spherical", verbose=False, enable_plotting=False
)

# === Predict drift and kriged residuals for test ===
drift_test = stratified_regression_prediction(raw_train, ppt_train, raw_test, bands)
kriged_residuals, _ = ok_model.execute("points", coord_test[:, 0], coord_test[:, 1])

# === Final prediction ===
final_pred = drift_test + kriged_residuals

# === Evaluation ===
rmse = np.sqrt(np.mean((ppt_test - final_pred) ** 2))
nse = compute_nse(final_pred, ppt_test)

# === Plot ===
plt.figure(figsize=(6, 6))
plt.scatter(ppt_test, final_pred, alpha=0.7, edgecolor='k')
plt.plot([ppt_test.min(), ppt_test.max()], [ppt_test.min(), ppt_test.max()], 'r--')
plt.xlabel("Observed Precipitation")
plt.ylabel("Predicted Precipitation")
plt.title(f"Stratified Regression + Kriging\nRMSE = {rmse:.3f}, NSE = {nse:.3f}")
plt.grid(True)
plt.tight_layout()
plt.show()
