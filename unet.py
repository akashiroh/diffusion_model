
from functools import partial

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        y = self.activation(x)
        return y

class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()

        upsample = partial(nn.ConvTranspose2d, kernel_size=2, stride=2)
        pool = partial(nn.MaxPool2d, kernel_size=2, stride=2)

        # List to store our residual connections in
        self.residuals = []
        
        # Helper Module classes to handle residual connections
        class AddResidual(nn.Module):
            def __init__(inner_self):
                super(AddResidual, inner_self).__init__()

            def forward(inner_self, x: torch.Tensor):
                self.residuals.append(x)
                return x

        class UseResidual(nn.Module):
            def __init__(inner_self):
                super(UseResidual, inner_self).__init__()

            def forward(inner_self, x: torch.Tensor):
                res = self.residuals.pop()
                crop_amount = x.shape[-1]
                cropped = F.center_crop(res, [crop_amount, crop_amount])
                return torch.concat([cropped, x], dim=1)
 
        # Layers for encoding input data into a latent
        self.encoder = nn.Sequential(
                ConvBlock(input_channels, 64),
                ConvBlock(64, 64),
                # Add a residual connection using the x at this point
                AddResidual(),
                # Downsample data with maxpool
                pool(),
                ConvBlock(64, 128),
                ConvBlock(128, 128),
                AddResidual(),
                pool(),
                ConvBlock(128, 256),
                ConvBlock(256, 256),
                )
        # Layers for decoding latents into image(-like) data
        self.decoder = nn.Sequential(
                # Upsample data with transposed convolution
                upsample(256, 128),
                # Add residual connection to this point
                UseResidual(),
                # Convolute over the residual + x
                ConvBlock(256, 128), # in_channels=256 is to account for the residual connection being concat'd to the input here
                ConvBlock(128, 128),
                upsample(128, 64),
                UseResidual(),
                ConvBlock(128, 64),
                ConvBlock(64, 64),
                nn.Conv2d(64, output_channels, kernel_size=1)
                )

    def forward(self, x):
        latent = self.encoder(x)
        y = self.decoder(latent)
        return y

if __name__ == '__main__':
    x = torch.ones(1, 3, 256, 256)
    model = UNet(input_channels=3, output_channels=3)

    y = model(x)

    print(f'Input: {x}')
    print(f'Input Shape: {x.shape}')
    print(f'Output: {y}')
    print(f'Output: {y.shape}')
