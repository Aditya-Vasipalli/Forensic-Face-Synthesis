"""Generate mask assets for the aging pipeline."""

import os

import numpy as np
from PIL import Image

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def gaussian_2d(shape, cx, cy, sx, sy):
    """Return a 2-D Gaussian array (float64, range 0-1)."""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    g = np.exp(-((x - cx) ** 2 / (2 * sx**2) + (y - cy) ** 2 / (2 * sy**2)))
    return g


# --- mask1024.jpg ---
# Soft elliptical face-region mask: 1024x1024, sigma_x=420, sigma_y=460
g1 = gaussian_2d((1024, 1024), cx=512, cy=512, sx=420, sy=460)
mask1024 = (g1 * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(mask1024, mode="L").save(os.path.join(ASSETS_DIR, "mask1024.jpg"))
print("Saved mask1024.jpg")

# --- mask512.jpg ---
# Sliding-window feathering mask: 512x512, sigma=200 (isotropic)
g2 = gaussian_2d((512, 512), cx=256, cy=256, sx=200, sy=200)
mask512 = (g2 * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(mask512, mode="L").save(os.path.join(ASSETS_DIR, "mask512.jpg"))
print("Saved mask512.jpg")

print("Done.")
