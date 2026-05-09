import json
import torch
import torch.nn as nn


def load_attr_config(config_path="config_attributes.json"):
    with open(config_path) as f:
        return json.load(f)


class AttributeEmbedder(nn.Module):
    """
    Embeds categorical text attributes defined in config_attributes.json.
    Each attribute gets its own nn.Embedding; outputs are concatenated.

    Config keys used:
        text_attributes.<name>.values   -> vocabulary size
        text_attributes.<name>.embed_dim -> embedding dimension
        total_embed_dim                 -> sum of all embed_dims
    """

    def __init__(self, config):
        super().__init__()
        attrs = config["text_attributes"]
        self.attr_names = list(attrs.keys())
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(len(meta["values"]), meta["embed_dim"])
            for name, meta in attrs.items()
        })
        self.total_dim = config["total_embed_dim"]

    def forward(self, attr_dict):
        # attr_dict: {attr_name: LongTensor(B,)}
        parts = [self.embeddings[name](attr_dict[name]) for name in self.attr_names]
        return torch.cat(parts, dim=1)   # (B, total_embed_dim)


def tile_to_spatial(vec, H, W):
    """Broadcast a flat vector to a spatial feature map: (B, D) -> (B, D, H, W)."""
    return vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)


# ─────────────────────────── U-Net blocks ────────────────────────────────────

class UNetDown(nn.Module):
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


# ─────────────────────────── Generator ───────────────────────────────────────

class Generator(nn.Module):
    """
    Attribute-conditioned U-Net (256x256).

    Injection: text attributes are spatially tiled and concatenated to the
    sketch before the first encoder layer.
        in_ch = 3  (sketch RGB)
              + attr_embed_dim  (from config total_embed_dim)
    """

    def __init__(self, attr_embed_dim=48, out_ch=3):
        super().__init__()
        in_ch = 3 + attr_embed_dim

        self.down1 = UNetDown(in_ch, 64, normalize=False)   # 128
        self.down2 = UNetDown(64, 128)                        # 64
        self.down3 = UNetDown(128, 256)                       # 32
        self.down4 = UNetDown(256, 512)                       # 16
        self.down5 = UNetDown(512, 512)                       # 8
        self.down6 = UNetDown(512, 512)                       # 4
        self.down7 = UNetDown(512, 512)                       # 2
        self.down8 = UNetDown(512, 512, normalize=False)      # 1

        # bottleneck injection: d8(512) + tiled attr(attr_embed_dim) -> up1
        self.up1 = UNetUp(512 + attr_embed_dim, 512, dropout=True)    # 2
        self.up2 = UNetUp(1024, 512, dropout=True)   # 4
        self.up3 = UNetUp(1024, 512, dropout=True)   # 8
        self.up4 = UNetUp(1024, 512)                 # 16
        self.up5 = UNetUp(1024, 256)                 # 32
        self.up6 = UNetUp(512, 128)                  # 64
        self.up7 = UNetUp(256, 64)                   # 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_ch, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, sketch, attr_vec):
        # attr_vec: (B, attr_embed_dim)
        H, W = sketch.shape[2], sketch.shape[3]
        attr_map = tile_to_spatial(attr_vec, H, W)         # (B, D, H, W)
        x = torch.cat([sketch, attr_map], dim=1)           # (B, 3+D, H, W)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # inject attributes a second time at the 1×1 bottleneck
        attr_bottleneck = tile_to_spatial(attr_vec, 1, 1)       # (B, D, 1, 1)
        d8_cond = torch.cat([d8, attr_bottleneck], dim=1)        # (B, 512+D, 1, 1)

        u1 = self.up1(d8_cond, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


# ─────────────────────────── Discriminator ───────────────────────────────────

class Discriminator(nn.Module):
    """
    Attribute-conditioned PatchGAN discriminator.

    Input channels = 3 (sketch) + 3 (photo) + attr_embed_dim = 6 + attr_embed_dim
    """

    def __init__(self, attr_embed_dim=48):
        super().__init__()
        in_ch = 6 + attr_embed_dim

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

    def forward(self, sketch, photo, attr_vec):
        H, W = sketch.shape[2], sketch.shape[3]
        attr_map = tile_to_spatial(attr_vec, H, W)           # (B, D, H, W)
        x = torch.cat([sketch, photo, attr_map], dim=1)      # (B, 6+D, H, W)
        return self.model(x)
