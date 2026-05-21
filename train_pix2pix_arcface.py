import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model_pix2pix import Generator, Discriminator
from dataset_loader import SketchPhotoDataset
from arcface import ArcFaceID

# ---------- Config ----------
SKETCH_DIR = "dataset/sketches"
PHOTO_DIR = "dataset/photos"
BATCH_SIZE = 4
EPOCHS = 200
LR = 2e-4
LAMBDA_L1 = 100
LAMBDA_ID = 5.0          # ArcFace identity preservation weight
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "output_pix2pix_arcface"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/samples", exist_ok=True)

# ---------- Data ----------
dataset = SketchPhotoDataset(SKETCH_DIR, PHOTO_DIR, img_size=256)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ---------- Models ----------
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

optimizer_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_L1 = torch.nn.L1Loss()

# ---------- ArcFace (identity preservation) ----------
arcface = ArcFaceID(device=DEVICE)

# ---------- Training ----------
for epoch in range(EPOCHS):
    for i, (sketch, photo) in enumerate(loader):
        sketch = sketch.to(DEVICE)
        photo = photo.to(DEVICE)

        # ---- Train Generator ----
        fake_photo = G(sketch)
        pred_fake = D(sketch, fake_photo)

        loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_L1 = criterion_L1(fake_photo, photo)
        loss_id = arcface.identity_loss(fake_photo, photo)

        loss_G = loss_GAN + LAMBDA_L1 * loss_L1 + LAMBDA_ID * loss_id

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # ---- Train Discriminator ----
        pred_real = D(sketch, photo)
        loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

        pred_fake = D(sketch, fake_photo.detach())
        loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        loss_D = (loss_real + loss_fake) / 2

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

    # Log every epoch
    print(f"[Epoch {epoch+1}/{EPOCHS}] D_loss: {loss_D.item():.4f}  G_loss: {loss_G.item():.4f}  L1: {loss_L1.item():.4f}  ID: {loss_id.item():.4f}")

    # Save samples every 10 epochs
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            sample = torch.cat([sketch[:4], fake_photo[:4], photo[:4]], dim=0)
            save_image(sample, f"{OUT_DIR}/samples/epoch_{epoch+1}.png", nrow=4, normalize=True)

    # Save checkpoint every 50 epochs
    if (epoch + 1) % 50 == 0:
        torch.save({
            "G": G.state_dict(),
            "D": D.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
            "epoch": epoch,
        }, f"{OUT_DIR}/checkpoint_epoch_{epoch+1}.pt")

print("Training complete.")
