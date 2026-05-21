"""
Flask REST API for face restoration via GFPGAN.
Endpoint: POST /api/clarify/enhance
Body:     multipart/form-data { image: File, scale: int, denoise: bool, sharpen: bool }
Response: JSON { output_url: "data:image/png;base64,..." }

Run with:
  D:/Study/college/SEM4/DV/complete-pandas-tutorial/venv/Scripts/python.exe flask_api_clarify.py

Note: First run downloads facexlib detection weights (~104 MB) into models/GFPGAN/weights/.
"""

import base64
import io
import os
import sys

import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Tell facexlib to store its auto-downloaded weights next to GFPGANv1.4.pth
os.environ.setdefault(
    "FACEXLIB_WEIGHTS_DIR", os.path.join(_HERE, "models", "GFPGAN", "weights")
)

from facexlib.utils.face_restoration_helper import (
    FaceRestoreHelper,  # noqa: F401 (triggers path setup)
)

from gfpgan import GFPGANer

app = Flask(__name__)

GFPGAN_MODEL = os.path.join(_HERE, "models", "GFPGAN", "GFPGANv1.4.pth")
WEIGHTS_DIR = os.path.join(_HERE, "models", "GFPGAN", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

print("[clarify] Loading GFPGANer…")
restorer = GFPGANer(
    model_path=GFPGAN_MODEL,
    upscale=2,  # default; overridden per-request
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None,
)
print("[clarify] GFPGANer ready.")


# ── CORS ──────────────────────────────────────────────────────────────────
@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/api/clarify/enhance", methods=["OPTIONS"])
def preflight():
    return "", 204


# ── Helpers ───────────────────────────────────────────────────────────────
def apply_denoise(bgr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(
        bgr, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21
    )


def apply_sharpen(bgr: np.ndarray) -> np.ndarray:
    """Unsharp mask sharpening."""
    blurred = cv2.GaussianBlur(bgr, (0, 0), 3)
    return cv2.addWeighted(bgr, 1.5, blurred, -0.5, 0)


# ── Route ─────────────────────────────────────────────────────────────────
@app.route("/api/clarify/enhance", methods=["POST"])
def enhance():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        scale = max(1, min(4, int(request.form.get("scale", 2))))
        denoise = request.form.get("denoise", "false").lower() == "true"
        sharpen = request.form.get("sharpen", "false").lower() == "true"
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid parameters"}), 400

    img_bytes = request.files["image"].read()
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    # PIL RGB -> OpenCV BGR
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Optional pre-processing
    if denoise:
        bgr = apply_denoise(bgr)

    try:
        # GFPGANer.enhance returns (cropped_faces, restored_faces, restored_img)
        restorer.upscale = scale
        _, _, restored_bgr = restorer.enhance(
            bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5,
        )
    except Exception as e:
        return jsonify({"error": f"Restoration failed: {e}"}), 500

    if restored_bgr is None:
        # No face detected — fall back to plain upscale via PIL
        h, w = bgr.shape[:2]
        restored_bgr = cv2.resize(
            bgr, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4
        )

    # Optional post-processing
    if sharpen:
        restored_bgr = apply_sharpen(restored_bgr)

    # BGR -> RGB -> PNG -> base64
    restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(restored_rgb)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({"output_url": f"data:image/png;base64,{b64}"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "GFPGANv1.4"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7003, debug=False)
