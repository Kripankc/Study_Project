import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from math import cos, sin, radians
from A1_Data_loader import DataLoader


def create_directional_kernel(direction_deg, h_s_pixels, width_pixels):
    """
    Create a directional smoothing kernel using a pixel-based radius and width.
    The weight decays linearly in both the smoothing direction and lateral width.

    direction_deg: azimuth angle in degrees (0° = North, 90° = East, etc.)
    h_s_pixels: smoothing radius in pixels
    width_pixels: lateral width in pixels
    """
    theta = radians(direction_deg)
    dx, dy = cos(theta), -sin(theta)  # y is downward in image coords

    size = 2 * (h_s_pixels + width_pixels) + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2

    for r in range(1, h_s_pixels + 1):
        dir_weight = 1 - (r / h_s_pixels)

        for w in range(-width_pixels, width_pixels + 1):
            lat_weight = 1 - (abs(w) / width_pixels) if width_pixels > 0 else 1.0
            if lat_weight < 0:
                continue

            x = int(round(cx + dx * r + (-dy * w)))
            y = int(round(cy + dy * r + (dx * w)))

            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = dir_weight * lat_weight

    if kernel.sum() > 0:
        kernel /= kernel.sum()
    return kernel


def compute_smoothed_elevation(dem, direction_deg, h_s_pixels, width_pixels):
    """
    Smooth the DEM using a directional kernel.
    All distances are in pixels.
    """
    kernel = create_directional_kernel(direction_deg, h_s_pixels, width_pixels)
    filled_dem = np.nan_to_num(dem, nan=np.nanmean(dem))
    return convolve(filled_dem, kernel, mode='nearest')


def get_elevation_at_coords(dem, transform, xs, ys):
    """Sample DEM values at given station coordinates."""
    inv_transform = ~transform
    rc_coords = inv_transform * (xs, ys)
    rows, cols = np.round(rc_coords[1]).astype(int), np.round(rc_coords[0]).astype(int)
    valid = (rows >= 0) & (rows < dem.shape[0]) & (cols >= 0) & (cols < dem.shape[1])
    elevation_values = np.full_like(xs, np.nan, dtype=np.float64)
    elevation_values[valid] = dem[rows[valid], cols[valid]]
    return elevation_values


if __name__ == "__main__":
    dem, transform, crs = DataLoader.load_dem()
    if dem is None:
        exit("DEM not loaded.")

    directions = [0, 90, 135, 180, 270]
    h_s_px = 65  # smoothing radius in pixels
    width_px = 8  # lateral width in pixels

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

        # Compute arrow direction
        arrow_length = 20
        theta = radians(deg)
        dx = arrow_length * cos(theta)
        dy = -arrow_length * sin(theta)

        # Start position: near bottom-center of the image
        start_x = smoothed.shape[1] // 2
        start_y = smoothed.shape[0] - 30  # 30 pixels from bottom

        # Add arrow
        ax.annotate('', xy=(start_x + dx, start_y + dy), xytext=(start_x, start_y),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))

        # Add text label with angle
        ax.text(start_x + dx + 5, start_y + dy, f'{deg}°', fontsize=9, color='black')

    plt.tight_layout()
    plt.show()
