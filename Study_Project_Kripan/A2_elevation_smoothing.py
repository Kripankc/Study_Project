import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from math import cos, sin, radians
from A1_Data_loader import DataLoader


def create_directional_kernel(direction_deg, h_s_pixels, width_pixels):
    """
    Creates a directional smoothing kernel with decaying weights along the
    direction and lateral width.

    Parameters:
        direction_deg (float): Smoothing direction in degrees (azimuth).
        h_s_pixels (int): Smoothing radius in pixels.
        width_pixels (int): Lateral width of smoothing in pixels.

    Returns:
        np.ndarray: Normalized smoothing kernel.
    """
    theta = radians(direction_deg)
    dx, dy = cos(theta), -sin(theta)  # Image coordinates: y-axis is downward

    size = 2 * (h_s_pixels + width_pixels) + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2

    for r in range(1, h_s_pixels + 1):
        dir_weight = 1 - (r / h_s_pixels)
        for w in range(-width_pixels, width_pixels + 1):
            lat_weight = 1 - (abs(w) / width_pixels) if width_pixels > 0 else 1.0
            if lat_weight <= 0:
                continue
            x = int(round(cx + dx * r + (-dy * w)))
            y = int(round(cy + dy * r + (dx * w)))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = dir_weight * lat_weight

    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel


def compute_smoothed_elevation(dem, direction_deg, h_s_pixels, width_pixels):
    """
    Applies directional smoothing to the DEM using a custom kernel.

    Parameters:
        dem (np.ndarray): Digital Elevation Model.
        direction_deg (float): Smoothing direction in degrees.
        h_s_pixels (int): Smoothing radius.
        width_pixels (int): Lateral width.

    Returns:
        np.ndarray: Smoothed DEM.
    """
    kernel = create_directional_kernel(direction_deg, h_s_pixels, width_pixels)
    filled_dem = np.nan_to_num(dem, nan=np.nanmean(dem))
    return convolve(filled_dem, kernel, mode='nearest')


def get_smoothed_elevation(xs, ys, method='default', direction_deg=135, h_s_pixels=65, width_pixels=8):
    """
    Retrieves smoothed elevation values at given station coordinates.

    Parameters:
        xs, ys (np.ndarray): Station coordinates.
        method (str): Placeholder for future strategies.
        direction_deg (float): Direction for smoothing.
        h_s_pixels (int): Smoothing radius.
        width_pixels (int): Lateral width.

    Returns:
        np.ndarray: Smoothed elevation values at station locations.
    """
    dem, transform, _ = DataLoader.load_dem()
    if dem is None:
        raise RuntimeError("DEM could not be loaded.")

    smoothed_dem = compute_smoothed_elevation(dem, direction_deg, h_s_pixels, width_pixels)
    return get_elevation_at_coords(smoothed_dem, transform, xs, ys)


def get_elevation_at_coords(dem, transform, xs, ys):
    """
    Samples elevation values from the DEM at specified coordinates.

    Parameters:
        dem (np.ndarray): DEM raster array.
        transform (Affine): Rasterio affine transform.
        xs, ys (np.ndarray): Coordinate arrays.

    Returns:
        np.ndarray: Elevation values at input coordinates.
    """
    inv_transform = ~transform
    rc_coords = inv_transform * (xs, ys)
    rows, cols = np.round(rc_coords[1]).astype(int), np.round(rc_coords[0]).astype(int)
    valid = (rows >= 0) & (rows < dem.shape[0]) & (cols >= 0) & (cols < dem.shape[1])
    elevation_values = np.full_like(xs, np.nan, dtype=np.float64)
    elevation_values[valid] = dem[rows[valid], cols[valid]]
    return elevation_values


if __name__ == "__main__":
    # Visualize the effect of directional smoothing for selected angles
    dem, transform, _ = DataLoader.load_dem()
    if dem is None:
        exit("DEM not loaded.")

    directions = [0, 90, 135, 180, 270]
    h_s_px = 65
    width_px = 8

    plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(dem, cmap="terrain")
    ax1.set_title("Original DEM")
    plt.colorbar(ax1.images[0], ax=ax1, shrink=0.7)

    for i, deg in enumerate(directions, 2):
        smoothed = compute_smoothed_elevation(dem, deg, h_s_px, width_px)
        ax = plt.subplot(2, 3, i)
        im = ax.imshow(smoothed, cmap="terrain")
        ax.set_title(f"Direction: {deg}°")
        plt.colorbar(im, ax=ax, shrink=0.7)

        # Add direction arrow
        arrow_length = 20
        theta = radians(deg)
        dx, dy = arrow_length * cos(theta), -arrow_length * sin(theta)
        cx, cy = smoothed.shape[1] // 2, smoothed.shape[0] - 30
        ax.annotate('', xy=(cx + dx, cy + dy), xytext=(cx, cy),
                    arrowprops=dict(facecolor='black', width=2, headwidth=8))
        ax.text(cx + dx + 5, cy + dy, f'{deg}°', fontsize=9, color='black')

    plt.tight_layout()
    plt.show()
