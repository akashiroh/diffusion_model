import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math


class DownConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DownConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class UpConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(UpConvNet, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Unet, self).__init__()
        
        self.blocks = nn.Sequential(
            DownConvNet(in_channels, 64),
            DownConvNet(64, 128),
            DownConvNet(128, 256),
            DownConvNet(256, 512),
            DownConvNet(512, 1024),
            UpConvNet(1024, 512),
            UpConvNet(512, 256),
            UpConvNet(256, 128),
            UpConvNet(128, 64),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.blocks(x)

    

