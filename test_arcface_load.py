import torch
from arcface import ArcFaceID

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = ArcFaceID(device=device)

x = torch.randn(2, 3, 256, 256, device=device)
y = torch.randn(2, 3, 256, 256, device=device)

print("identity_loss:", m.identity_loss(x, y).item())
