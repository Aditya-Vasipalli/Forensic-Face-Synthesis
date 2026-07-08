import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

# ── OpenCV face detector ──────────────────────────────────────────────────────
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)

# ── masks: load from assets/, generate on-the-fly if missing ─────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))


def _make_gaussian_mask(size, sigma_x=None, sigma_y=None):
    """Return a (size x size) float32 numpy array with a 2-D Gaussian, peak=1."""
    if sigma_x is None:
        sigma_x = size * 0.39
    if sigma_y is None:
        sigma_y = sigma_x
    cx = cy = size / 2.0
    y, x = np.ogrid[:size, :size]
    mask = np.exp(
        -((x - cx) ** 2 / (2 * sigma_x**2) + (y - cy) ** 2 / (2 * sigma_y**2))
    )
    return (mask * 255).clip(0, 255).astype(np.uint8)


def _ensure_asset(filename, size, sigma_x=None, sigma_y=None):
    assets_dir = os.path.join(_HERE, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    path = os.path.join(assets_dir, filename)
    if not os.path.exists(path):
        arr = _make_gaussian_mask(size, sigma_x, sigma_y)
        Image.fromarray(arr, mode="L").save(path)
    return path


_mask_path = _ensure_asset("mask1024.jpg", 1024, sigma_x=420, sigma_y=460)
_small_mask_path = _ensure_asset("mask512.jpg", 512, sigma_x=200, sigma_y=200)

mask_file = (
    torch.from_numpy(np.array(Image.open(_mask_path).convert("L"))).float() / 255
)
small_mask_file = (
    torch.from_numpy(np.array(Image.open(_small_mask_path).convert("L"))).float() / 255
)


# ── face detection via OpenCV Haar cascade ───────────────────────────────────
def _detect_faces_opencv(image_rgb: np.ndarray):
    """Return list of (top, right, bottom, left) tuples, same format as face_recognition."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return []
    locations = []
    for x, y, w, h in faces:
        # convert (x, y, w, h) -> (top, right, bottom, left)
        locations.append((y, x + w, y + h, x))
    return locations


# ── sliding window ────────────────────────────────────────────────────────────
def sliding_window_tensor(
    input_tensor,
    window_size,
    stride,
    your_model,
    mask=mask_file,
    small_mask=small_mask_file,
):
    """Apply aging on input_tensor using overlapping sliding windows."""
    device = next(your_model.parameters()).device
    input_tensor = input_tensor.to(device)
    mask = mask.to(device)
    small_mask = small_mask.to(device)

    n, c, h, w = input_tensor.size()
    output_tensor = torch.zeros((n, 3, h, w), dtype=input_tensor.dtype, device=device)
    count_tensor = torch.zeros((n, 3, h, w), dtype=torch.float32, device=device)

    add = 2 if window_size % stride != 0 else 1

    for y in range(0, h - window_size + add, stride):
        for x in range(0, w - window_size + add, stride):
            window = input_tensor[:, :, y : y + window_size, x : x + window_size]
            input_variable = Variable(window, requires_grad=False)
            with torch.no_grad():
                output = your_model(input_variable)
            output_tensor[:, :, y : y + window_size, x : x + window_size] += (
                output * small_mask
            )
            count_tensor[:, :, y : y + window_size, x : x + window_size] += small_mask

    count_tensor = torch.clamp(count_tensor, min=1.0)
    output_tensor /= count_tensor
    output_tensor *= mask

    return output_tensor.cpu()


# ── main entry ────────────────────────────────────────────────────────────────
def process_image(
    your_model, image, source_age, target_age=0, window_size=512, stride=256, steps=18
):
    input_size = (1024, 1024)

    image_arr = np.array(image.convert("RGB"))

    face_locations = _detect_faces_opencv(image_arr)
    if not face_locations:
        raise ValueError(
            "No faces detected in the image. Please ensure the image contains a clear, visible face."
        )

    fl = face_locations[0]  # (top, right, bottom, left)

    margin_y_t = int((fl[2] - fl[0]) * 0.63 * 0.85)
    margin_y_b = int((fl[2] - fl[0]) * 0.37 * 0.85)
    margin_x = int((fl[1] - fl[3]) // (2 / 0.85))
    margin_y_t += 2 * margin_x - margin_y_t - margin_y_b

    l_y = max(fl[0] - margin_y_t, 0)
    r_y = min(fl[2] + margin_y_b, image_arr.shape[0])
    l_x = max(fl[3] - margin_x, 0)
    r_x = min(fl[1] + margin_x, image_arr.shape[1])

    cropped = image_arr[l_y:r_y, l_x:r_x, :]

    cropped_t = transforms.ToTensor()(cropped)
    resized_t = transforms.Resize(
        input_size, interpolation=Image.BILINEAR, antialias=True
    )(cropped_t)

    src_ch = torch.full_like(resized_t[:1], source_age / 100)
    tgt_ch = torch.full_like(resized_t[:1], target_age / 100)
    input_tensor = torch.cat([resized_t, src_ch, tgt_ch], dim=0).unsqueeze(0)

    image_t = transforms.ToTensor()(image_arr)

    aged_crop = sliding_window_tensor(input_tensor, window_size, stride, your_model)

    orig_size = cropped.shape[:2]
    aged_crop_resized = transforms.Resize(
        orig_size, interpolation=Image.BILINEAR, antialias=True
    )(aged_crop.squeeze(0).unsqueeze(0)).squeeze(0)

    image_t[:, l_y:r_y, l_x:r_x] += aged_crop_resized
    image_t = torch.clamp(image_t, 0, 1)

    return transforms.functional.to_pil_image(image_t)
