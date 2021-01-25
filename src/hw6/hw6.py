import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np


class Generator(nn.Module):
    def __init__(self, start_shape=2, latent_channels=32, start_channels=1024, upsamplings=5):
        super(Generator, self).__init__()
        self.start_channels = start_channels
        self.start_shape = start_shape
        self.latent_channels = latent_channels
        self.upsamplings = upsamplings

        self.configure_model()

    def configure_model(self):
        self.conv1 = nn.Conv2d(in_channels=self.latent_channels,
                               out_channels=self.start_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0
                               )
        create_upblock = lambda c: [
            nn.ConvTranspose2d(
                in_channels=c,
                out_channels=c//2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=c//2),
            nn.ReLU()
        ]
        seq = []
        c = self.start_channels
        for _ in range(self.upsamplings):
            seq += create_upblock(c)
            c //= 2

        self.upsampling_block = nn.Sequential(*seq)

        self.conv2 = nn.Conv2d(
            in_channels=c,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        res = self.conv1(x)
        res = self.upsampling_block(res)
        res = self.conv2(res)
        return res


class Discriminator(nn.Module):
    def __init__(self, start_shape=128, downsamplings=5, start_channels=8):
        super().__init__()
        self.downsamplings = downsamplings
        self.start_channels = start_channels
        self.start_shape = start_shape

        self.configure_model()

    def configure_model(self):
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=self.start_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0
                               )

        create_upblock = lambda c: [
            nn.Conv2d(
                in_channels=c,
                out_channels=c*2,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=c*2),
            nn.ReLU()
        ]
        seq = []
        c = self.start_channels
        for _ in range(self.downsamplings):
            seq += create_upblock(c)
            c *= 2

        self.upsampling_block = nn.Sequential(*seq)

        self.linear = nn.Linear(
            in_features=c * (self.start_shape // 2 ** self.downsamplings) ** 2,
            out_features=1
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        res = self.conv1(x)
        res = self.upsampling_block(res)
        res = torch.flatten(res, start_dim=1)
        res = self.linear(res)
        res = self.activation(res)
        return res
