import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model_pix2pix_attr import Generator, Discriminator, AttributeEmbedder, load_attr_config
from dataset_loader import SketchPhotoDataset

# ─────────────────────────── Config ──────────────────────────────────────────

CONFIG_PATH = "config_attributes.json"
SKETCH_DIR  = "dataset/sketches"
PHOTO_DIR   = "dataset/photos"
BATCH_SIZE  = 2
EPOCHS      = 200
LR          = 2e-4
LAMBDA_L1   = 100
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR     = "output_pix2pix_attr"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/samples", exist_ok=True)

# ─────────────────────────── Attributes ──────────────────────────────────────

cfg           = load_attr_config(CONFIG_PATH)
attr_names    = list(cfg["text_attributes"].keys())
attr_vocab    = {name: len(meta["values"]) for name, meta in cfg["text_attributes"].items()}
total_emb_dim = cfg["total_embed_dim"]

print(f"Attributes loaded from {CONFIG_PATH}: {attr_names}")
print(f"Total embedding dim: {total_emb_dim}")

# ─────────────────────────── Data ────────────────────────────────────────────

dataset = SketchPhotoDataset(SKETCH_DIR, PHOTO_DIR, img_size=256)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ─────────────────────────── Models ──────────────────────────────────────────

embedder = AttributeEmbedder(cfg).to(DEVICE)
G        = Generator(attr_embed_dim=total_emb_dim).to(DEVICE)
D        = Discriminator(attr_embed_dim=total_emb_dim).to(DEVICE)

# Embedder params are optimised together with the Generator
optimizer_G = torch.optim.Adam(
    list(G.parameters()) + list(embedder.parameters()), lr=LR, betas=(0.5, 0.999)
)
optimizer_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_L1  = torch.nn.L1Loss()

# ─────────────────────────── Attribute helper ───────────────────────────────
# Random varying attrs each batch — this is what forces the model to actually
# learn conditioning. Fixed attrs = model ignores the attribute channel.

def get_attrs(batch_size):
    return {
        name: torch.randint(0, attr_vocab[name], (batch_size,)).to(DEVICE)
        for name in attr_names
    }

# ─────────────────────────── Training loop ───────────────────────────────────

for epoch in range(EPOCHS):
    for i, (sketch, photo) in enumerate(loader):
        print(f"  [Epoch {epoch+1}/{EPOCHS}  Batch {i+1}/{len(loader)}] loading...", flush=True)
        sketch = sketch.to(DEVICE)
        photo  = photo.to(DEVICE)
        B      = sketch.size(0)

        attrs    = get_attrs(B)
        attr_vec = embedder(attrs)                  # (B, total_emb_dim)

        # ── Generator ──────────────────────────────────────────────────────
        fake_photo = G(sketch, attr_vec)
        pred_fake  = D(sketch, fake_photo, attr_vec)

        loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_L1  = criterion_L1(fake_photo, photo)
        loss_G   = loss_GAN + LAMBDA_L1 * loss_L1

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # ── Discriminator ──────────────────────────────────────────────────
        attr_vec_d  = attr_vec.detach()

        pred_real   = D(sketch, photo, attr_vec_d)
        loss_real   = criterion_GAN(pred_real, torch.ones_like(pred_real))

        pred_fake_d = D(sketch, fake_photo.detach(), attr_vec_d)
        loss_fake   = criterion_GAN(pred_fake_d, torch.zeros_like(pred_fake_d))

        loss_D = (loss_real + loss_fake) / 2

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        print(
            f"  [Epoch {epoch+1}/{EPOCHS}  Batch {i+1}/{len(loader)}] "
            f"D: {loss_D.item():.4f}  G: {loss_G.item():.4f}  L1: {loss_L1.item():.4f}",
            flush=True
        )

    print(
        f"[Epoch {epoch+1}/{EPOCHS} DONE] "
        f"D: {loss_D.item():.4f}  G: {loss_G.item():.4f}  L1: {loss_L1.item():.4f}",
        flush=True
    )

    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            sample = torch.cat([sketch[:4], fake_photo[:4], photo[:4]], dim=0)
            save_image(sample, f"{OUT_DIR}/samples/epoch_{epoch+1}.png", nrow=4, normalize=True)

    if (epoch + 1) % 50 == 0:
        torch.save({
            "G":           G.state_dict(),
            "D":           D.state_dict(),
            "embedder":    embedder.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
            "epoch":       epoch,
        }, f"{OUT_DIR}/checkpoint_epoch_{epoch+1}.pt")

print("Training complete.")
