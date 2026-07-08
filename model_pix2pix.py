import torch
import torch.nn as nn


class UNetDown(nn.Module):
    """Encoder block: Conv -> BatchNorm -> LeakyReLU"""

    def __init__(self, in_ch, out_ch, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Decoder block: ConvTranspose -> BatchNorm -> Dropout(optional) -> ReLU"""

    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat([x, skip], dim=1)


class Generator(nn.Module):
    """U-Net generator with skip connections (256x256 -> 256x256)"""

    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()

        # Encoder
        self.down1 = UNetDown(in_ch, 64, normalize=False)  # 128
        self.down2 = UNetDown(64, 128)                       # 64
        self.down3 = UNetDown(128, 256)                      # 32
        self.down4 = UNetDown(256, 512)                      # 16
        self.down5 = UNetDown(512, 512)                      # 8
        self.down6 = UNetDown(512, 512)                      # 4
        self.down7 = UNetDown(512, 512)                      # 2
        self.down8 = UNetDown(512, 512, normalize=False)     # 1

        # Decoder (in_ch is doubled because of skip connections)
        self.up1 = UNetUp(512, 512, dropout=True)   # 2
        self.up2 = UNetUp(1024, 512, dropout=True)  # 4
        self.up3 = UNetUp(1024, 512, dropout=True)  # 8
        self.up4 = UNetUp(1024, 512)                # 16
        self.up5 = UNetUp(1024, 256)                # 32
        self.up6 = UNetUp(512, 128)                 # 64
        self.up7 = UNetUp(256, 64)                  # 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_ch, 4, 2, 1),  # 256
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class Discriminator(nn.Module):
    """PatchGAN discriminator (conditional: takes sketch + photo as 6-channel input)"""

    def __init__(self, in_ch=6):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, sketch, photo):
        x = torch.cat([sketch, photo], dim=1)
        return self.model(x)