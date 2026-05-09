"""
Demo: generate a face photo from a sketch + custom text attributes.

Change CUSTOM_ATTRIBUTES below to any values listed in config_attributes.json.
The script generates one output image per attribute set and saves them side-by-side.

NOTE: the model here is randomly initialised (untrained), so the output is noise.
The point is to show the attribute -> embedding -> image pipeline working end-to-end.
Swap in a trained checkpoint to get real results.
"""

import os
import sys
import torch
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(__file__))
from model_pix2pix_attr import load_attr_config, AttributeEmbedder, Generator

# ──────────────────────────────────────────────────────────────────────────────
# EDIT THIS — pick any values from config_attributes.json
# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_ATTRIBUTES = [
    {"age": "child",       "hair_color": "black",  "gender": "male",   "skin_tone": "dark"},
    {"age": "young_adult", "hair_color": "blonde",  "gender": "female", "skin_tone": "light"},
    {"age": "senior",      "hair_color": "gray",    "gender": "female", "skin_tone": "brown"},
    {"age": "teenager",    "hair_color": "red",     "gender": "female", "skin_tone": "medium"},
]
# ──────────────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_attributes.json")
OUT_PATH    = os.path.join(os.path.dirname(__file__), "example_output.png")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg           = load_attr_config(CONFIG_PATH)
attr_cfg      = cfg["text_attributes"]
total_emb_dim = cfg["total_embed_dim"]

def attr_name_to_idx(attr_name, value):
    values = attr_cfg[attr_name]["values"]
    assert value in values, f"'{value}' not in {attr_name} values: {values}"
    return values.index(value)

embedder = AttributeEmbedder(cfg).to(DEVICE)
G        = Generator(attr_embed_dim=total_emb_dim).to(DEVICE)
G.eval()

print(f"Device : {DEVICE}")
print(f"Config : {CONFIG_PATH}\n")

results = []

with torch.no_grad():
    for attrs_dict in CUSTOM_ATTRIBUTES:
        # convert names → indices
        attr_tensors = {
            name: torch.tensor([attr_name_to_idx(name, val)]).to(DEVICE)
            for name, val in attrs_dict.items()
        }

        # random sketch (stand-in for a real sketch image)
        sketch   = torch.randn(1, 3, 256, 256).to(DEVICE)
        attr_vec = embedder(attr_tensors)           # (1, 48)
        output   = G(sketch, attr_vec)              # (1, 3, 256, 256)

        label = " | ".join(f"{k}={v}" for k, v in attrs_dict.items())
        print(f"  Generated for: {label}")
        results.append(output)

# save all outputs as a grid: one column per attribute set
grid = torch.cat(results, dim=0)   # (N, 3, 256, 256)
save_image(grid, OUT_PATH, nrow=len(CUSTOM_ATTRIBUTES), normalize=True)
print(f"\nSaved → {OUT_PATH}")
