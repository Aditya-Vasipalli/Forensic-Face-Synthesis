"""
Flask REST API for the Face Aging model.
Endpoint: POST /api/aging/generate
Body:     multipart/form-data  { image: <file>, source_age: int, target_age: int }
Response: JSON { output_url: "data:image/png;base64,..." }

Run with:
  D:/Study/college/SEM4/DV/complete-pandas-tutorial/venv/Scripts/python.exe flask_api.py
"""

import base64
import io
import os
import sys

import torch
from flask import Flask, jsonify, request
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from models import UNet
from test_functions import process_image

# ── app ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/api/aging/generate", methods=["OPTIONS"])
def preflight():
    return "", 204


# ── model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(_HERE, "best_unet_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[aging] Loading model on {device}…")

model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.to(device)
model.eval()
print("[aging] Model ready.")


# ── route ─────────────────────────────────────────────────────────────────────
@app.route("/api/aging/generate", methods=["POST"])
def generate():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    try:
        source_age = int(request.form.get("source_age", 25))
        target_age = int(request.form.get("target_age", 60))
    except (ValueError, TypeError):
        return jsonify({"error": "source_age and target_age must be integers"}), 400

    img_bytes = request.files["image"].read()
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    try:
        result: Image.Image = process_image(model, image, source_age, target_age)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({"output_url": f"data:image/png;base64,{b64}"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(device)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7001, debug=False)
