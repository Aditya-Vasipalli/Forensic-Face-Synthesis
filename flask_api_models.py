"""
Flask REST API for Pix2Pix and CycleGAN sketch-to-face models.
Endpoints:
  POST /api/cyclegan/generate        { image } -> { output_url, inference_ms }
  POST /api/pix2pix/generate         { image } -> { output_url, inference_ms }
  POST /api/pix2pix-arcface/generate { image } -> { output_url, inference_ms }
  GET  /health

Run with:
  D:/Study/college/SEM4/DV/complete-pandas-tutorial/venv/Scripts/python.exe flask_api_models.py
"""

import base64
import io
import os
import sys
import time

import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from model_cyclegan import ResnetGenerator
from model_pix2pix import Generator as Pix2PixGenerator

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[models-api] Device: {DEVICE}")

# ── Load "CycleGAN" tab model (cycle-gan-arcface folder, U-Net arch, key G) ──
CG_CKPT = os.path.join(_HERE, "models", "cycle-gan-arcface", "checkpoint_epoch_200.pt")
print("[models-api] Loading CycleGAN (U-Net)…")
cg_gen = Pix2PixGenerator(in_ch=3, out_ch=3).to(DEVICE)
_ck = torch.load(CG_CKPT, map_location=DEVICE, weights_only=False)
cg_gen.load_state_dict(_ck["G"])
cg_gen.eval()
print("[models-api] CycleGAN ready.")

# ── Load "Pix2Pix" tab model (pix2pix-no-arcface folder, ResNet arch, key G_S2P) ──
P2P_CKPT = os.path.join(
    _HERE, "models", "pix2pix-no-arcface", "checkpoint_epoch_200.pt"
)
print("[models-api] Loading Pix2Pix (ResNet)…")
p2p_gen = ResnetGenerator(in_ch=3, out_ch=3, n_res=9).to(DEVICE)
_ck2 = torch.load(P2P_CKPT, map_location=DEVICE, weights_only=False)
p2p_gen.load_state_dict(_ck2["G_S2P"])
p2p_gen.eval()
print("[models-api] Pix2Pix ready.")

# ── Load "Pix2Pix+ID" tab model (pix2pix-with-arcface folder, U-Net arch, key G) ──
P2P_ARC_CKPT = os.path.join(
    _HERE, "models", "pix2pix-with-arcface", "checkpoint_epoch_50.pt"
)
print("[models-api] Loading Pix2Pix+ArcFace (U-Net)…")
p2p_arc_gen = Pix2PixGenerator(in_ch=3, out_ch=3).to(DEVICE)
_ck3 = torch.load(P2P_ARC_CKPT, map_location=DEVICE, weights_only=False)
p2p_arc_gen.load_state_dict(_ck3["G"])
p2p_arc_gen.eval()
print("[models-api] Pix2Pix+ArcFace ready.")

# ── Transform ─────────────────────────────────────────────────────────────
_tf = transforms.Compose(
    [
        transforms.Resize((256, 256), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def run_inference(generator, image_bytes: bytes):
    """Returns (PIL Image, inference_ms)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _tf(img).unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = generator(tensor)
    ms = int((time.perf_counter() - t0) * 1000)

    out = (out.clamp(-1, 1) + 1.0) / 2.0  # -> [0,1]
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = (out * 255).astype(np.uint8)
    return Image.fromarray(out), ms


def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── CORS ──────────────────────────────────────────────────────────────────
@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/api/cyclegan/generate", methods=["OPTIONS"])
@app.route("/api/pix2pix/generate", methods=["OPTIONS"])
@app.route("/api/pix2pix-arcface/generate", methods=["OPTIONS"])
def preflight():
    return "", 204


# ── Routes ────────────────────────────────────────────────────────────────
def _generate_endpoint(generator):
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    img_bytes = request.files["image"].read()
    try:
        result, ms = run_inference(generator, img_bytes)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500
    return jsonify(
        {"output_url": f"data:image/png;base64,{to_b64(result)}", "inference_ms": ms}
    )


@app.route("/api/cyclegan/generate", methods=["POST"])
def cyclegan_generate():
    return _generate_endpoint(cg_gen)


@app.route("/api/pix2pix/generate", methods=["POST"])
def pix2pix_generate():
    return _generate_endpoint(p2p_gen)


@app.route("/api/pix2pix-arcface/generate", methods=["POST"])
def pix2pix_arcface_generate():
    return _generate_endpoint(p2p_arc_gen)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "device": str(DEVICE),
            "models": ["cyclegan", "pix2pix", "pix2pix-arcface"],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7002, debug=False)
