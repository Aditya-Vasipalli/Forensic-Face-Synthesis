import random
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """ResNet-based generator: c7s1-64, d128, d256, R256x9, u128, u64, c7s1-3"""

    def __init__(self, in_ch=3, out_ch=3, n_res=9, base_filters=64):
        super().__init__()

        # Initial conv block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, base_filters, 7, bias=False),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_f = base_filters
        for _ in range(2):
            out_f = in_f * 2
            model += [
                nn.Conv2d(in_f, out_f, 3, 2, 1, bias=False),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True),
            ]
            in_f = out_f

        # Residual blocks
        for _ in range(n_res):
            model.append(ResidualBlock(in_f))

        # Upsampling
        for _ in range(2):
            out_f = in_f // 2
            model += [
                nn.ConvTranspose2d(in_f, out_f, 3, 2, 1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_f),
                nn.ReLU(inplace=True),
            ]
            in_f = out_f

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_filters, out_ch, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (unconditional, 70x70 receptive field)"""

    def __init__(self, in_ch=3, base_filters=64, n_layers=3):
        super().__init__()

        layers = [
            nn.Conv2d(in_ch, base_filters, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf = base_filters
        for i in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Second-to-last layer (stride 1)
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Output layer (1 channel prediction map)
        layers.append(nn.Conv2d(nf, 1, 4, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ImageBuffer:
    """Stores up to `max_size` previously generated images.
    Returns a mix of new and buffered images to stabilize D training."""

    def __init__(self, max_size=50):
        self.max_size = max_size
        self.images = []

    def query(self, images):
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.images) < self.max_size:
                self.images.append(img.clone())
                result.append(img)
            elif random.random() > 0.5:
                idx = random.randint(0, self.max_size - 1)
                old = self.images[idx].clone()
                self.images[idx] = img.clone()
                result.append(old)
            else:
                result.append(img)
        return torch.cat(result, dim=0)
