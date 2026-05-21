import os
import sys
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from model_pix2pix import Generator

# ---------- Config ----------
CHECKPOINT = "output_pix2pix_arcface/checkpoint_epoch_200.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "output_pix2pix_arcface/test_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load model ----------
G = Generator().to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
G.load_state_dict(ckpt["G"])
G.eval()
print(f"Loaded checkpoint: {CHECKPOINT}")

# ---------- Transform (same as training) ----------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ---------- Inference ----------
# Usage: python test_pix2pix_arcface.py sketch1.jpg sketch2.jpg ...
# If no args, test on all sketches in dataset/sketches/

if len(sys.argv) > 1:
    sketch_paths = sys.argv[1:]
else:
    sketch_dir = "dataset/sketches"
    sketch_paths = [os.path.join(sketch_dir, f) for f in sorted(os.listdir(sketch_dir)) if f.endswith(".jpg")]
    print(f"No input specified, testing on all {len(sketch_paths)} sketches from {sketch_dir}/")

with torch.no_grad():
    for path in sketch_paths:
        sketch = Image.open(path).convert("RGB")
        sketch_tensor = transform(sketch).unsqueeze(0).to(DEVICE)
        
        fake_photo = G(sketch_tensor)
        
        # Save side-by-side: sketch | generated photo
        result = torch.cat([sketch_tensor, fake_photo], dim=3)  # concat width-wise
        
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = f"{OUT_DIR}/{name}.png"
        save_image(result, out_path, normalize=True)

print(f"Done! {len(sketch_paths)} results saved to {OUT_DIR}/")
