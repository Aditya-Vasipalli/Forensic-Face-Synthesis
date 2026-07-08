import os
import itertools
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model_cyclegan import ResnetGenerator, NLayerDiscriminator, ImageBuffer
from dataset_loader import SketchPhotoDataset
from arcface import ArcFaceID

# ---------- Config ----------
SKETCH_DIR = "dataset/sketches"
PHOTO_DIR = "dataset/photos"
BATCH_SIZE = 1           # CycleGAN standard: batch_size=1
EPOCHS = 200
LR = 2e-4
LAMBDA_CYCLE = 10.0
LAMBDA_ID = 5.0       # ArcFace identity preservation weight
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "output_cyclegan"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/samples", exist_ok=True)

# ---------- Data ----------
dataset = SketchPhotoDataset(SKETCH_DIR, PHOTO_DIR, img_size=256)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ---------- Models ----------
# G_S2P: sketch -> photo,  G_P2S: photo -> sketch
G_S2P = ResnetGenerator().to(DEVICE)
G_P2S = ResnetGenerator().to(DEVICE)
D_sketch = NLayerDiscriminator().to(DEVICE)
D_photo = NLayerDiscriminator().to(DEVICE)

# ---------- Optimizers ----------
optimizer_G = torch.optim.Adam(
    itertools.chain(G_S2P.parameters(), G_P2S.parameters()),
    lr=LR, betas=(0.5, 0.999),
)
optimizer_D = torch.optim.Adam(
    itertools.chain(D_sketch.parameters(), D_photo.parameters()),
    lr=LR, betas=(0.5, 0.999),
)

# Linear LR decay: constant for first 100 epochs, linear decay to 0 for remaining
def lr_lambda(epoch):
    decay_start = EPOCHS // 2
    if epoch < decay_start:
        return 1.0
    return 1.0 - (epoch - decay_start) / (EPOCHS - decay_start)

scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)

# ---------- Losses ----------
criterion_GAN = torch.nn.MSELoss()   # LSGAN
criterion_cycle = torch.nn.L1Loss()

# ---------- ArcFace (identity preservation) ----------
arcface = ArcFaceID(device=DEVICE)

# ---------- Buffers ----------
buffer_fake_sketch = ImageBuffer()
buffer_fake_photo = ImageBuffer()

# ---------- Training ----------
for epoch in range(EPOCHS):
    for i, (sketch, photo) in enumerate(loader):
        sketch = sketch.to(DEVICE)
        photo = photo.to(DEVICE)

        # ============================================================
        #  Train Generators  (G_S2P and G_P2S)
        # ============================================================
        optimizer_G.zero_grad()

        # -- Adversarial loss --
        fake_photo = G_S2P(sketch)
        pred_fake_photo = D_photo(fake_photo)
        loss_GAN_S2P = criterion_GAN(pred_fake_photo, torch.ones_like(pred_fake_photo))

        fake_sketch = G_P2S(photo)
        pred_fake_sketch = D_sketch(fake_sketch)
        loss_GAN_P2S = criterion_GAN(pred_fake_sketch, torch.ones_like(pred_fake_sketch))

        # -- Cycle-consistency loss --
        recovered_sketch = G_P2S(fake_photo)
        loss_cycle_sketch = criterion_cycle(recovered_sketch, sketch)

        recovered_photo = G_S2P(fake_sketch)
        loss_cycle_photo = criterion_cycle(recovered_photo, photo)

        # -- Identity preservation loss (ArcFace) --
        # Generated photo should have same identity as real photo
        loss_id = arcface.identity_loss(fake_photo, photo)

        # -- Total generator loss --
        loss_G = (
            loss_GAN_S2P + loss_GAN_P2S
            + LAMBDA_CYCLE * (loss_cycle_sketch + loss_cycle_photo)
            + LAMBDA_ID * loss_id
        )

        loss_G.backward()
        optimizer_G.step()

        # ============================================================
        #  Train Discriminators  (D_photo and D_sketch)
        # ============================================================
        optimizer_D.zero_grad()

        # -- D_photo --
        pred_real = D_photo(photo)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

        fake_photo_buf = buffer_fake_photo.query(fake_photo.detach())
        pred_fake = D_photo(fake_photo_buf)
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        loss_D_photo = (loss_D_real + loss_D_fake) / 2

        # -- D_sketch --
        pred_real = D_sketch(sketch)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

        fake_sketch_buf = buffer_fake_sketch.query(fake_sketch.detach())
        pred_fake = D_sketch(fake_sketch_buf)
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        loss_D_sketch = (loss_D_real + loss_D_fake) / 2

        loss_D = loss_D_photo + loss_D_sketch
        loss_D.backward()
        optimizer_D.step()

    # Step schedulers
    scheduler_G.step()
    scheduler_D.step()

    # Log
    print(
        f"[Epoch {epoch+1}/{EPOCHS}] "
        f"D: {loss_D.item():.4f}  "
        f"G: {loss_G.item():.4f}  "
        f"cyc: {(loss_cycle_sketch + loss_cycle_photo).item():.4f}  "
        f"id: {loss_id.item():.4f}  "
        f"lr: {scheduler_G.get_last_lr()[0]:.6f}"
    )

    # Save samples every 10 epochs
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            # sketch -> fake_photo -> recovered_sketch
            sample = torch.cat([sketch[:4], fake_photo[:4], recovered_sketch[:4], photo[:4]], dim=0)
            save_image(sample, f"{OUT_DIR}/samples/epoch_{epoch+1}.png", nrow=4, normalize=True)

    # Save checkpoint every 50 epochs
    if (epoch + 1) % 50 == 0:
        torch.save({
            "G_S2P": G_S2P.state_dict(),
            "G_P2S": G_P2S.state_dict(),
            "D_sketch": D_sketch.state_dict(),
            "D_photo": D_photo.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
            "epoch": epoch,
        }, f"{OUT_DIR}/checkpoint_epoch_{epoch+1}.pt")

print("Training complete.")
