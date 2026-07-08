"""ArcFace identity-preserving loss for face generation.

Uses the pre-trained ArcFace w600k_r50 model from insightface,
auto-converted from ONNX to pure PyTorch so gradients flow through
to the generator during training.

Usage:
    arcface = ArcFaceID(device)
    loss = arcface.identity_loss(generated_photo, real_photo)
"""

import os
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx2torch import convert


ONNX_PATH = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")


def _ensure_onnx_downloaded():
    """Download the ArcFace ONNX model via insightface if not present."""
    if os.path.exists(ONNX_PATH):
        return
    print("Downloading ArcFace model via insightface (first time only)...")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(112, 112))
    if not os.path.exists(ONNX_PATH):
        raise RuntimeError(f"ArcFace ONNX not found at {ONNX_PATH}")


class ArcFaceID(nn.Module):
    """Pre-trained ArcFace feature extractor for identity loss.

    - Input:  images in [-1, 1], any size (resized internally to 112x112)
    - Output: 512-d normalized embedding
    """

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

        _ensure_onnx_downloaded()
        onnx_model = onnx.load(ONNX_PATH)
        self.backbone = convert(onnx_model).to(device).eval()

        for p in self.backbone.parameters():
            p.requires_grad = False

        print("ArcFace loaded successfully.")

    def forward(self, img):
        """Extract 512-d embedding. Input: [B, 3, H, W] in [-1, 1]."""
        # ArcFace expects 112x112, normalize from [-1,1] to insightface range
        x = F.interpolate(img, size=(112, 112), mode="bilinear", align_corners=False)
        x = (x + 1.0) * 127.5 / 128.0  # [-1,1] -> approx [0, ~1.99] (insightface norm)
        emb = self.backbone(x)
        return F.normalize(emb, dim=1)

    def identity_loss(self, generated, target):
        """1 - cosine_similarity between generated and target face embeddings.
        Returns 0 when faces are identical, 2 when maximally different."""
        emb_gen = self.forward(generated)
        emb_tgt = self.forward(target)
        return (1.0 - F.cosine_similarity(emb_gen, emb_tgt)).mean()
