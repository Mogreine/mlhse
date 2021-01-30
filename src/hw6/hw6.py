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
                out_channels=c // 2,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=c // 2),
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
                out_channels=c * 2,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(num_features=c * 2),
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


class VAE(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, img_size=128, downsamplings=5, latent_size=512, down_channels=8):
            super().__init__()
            self.img_size = img_size
            self.downsamplings = downsamplings
            self.latent_size = latent_size
            self.down_channels = down_channels

            self.configure_model()

        def configure_model(self):
            conv1 = nn.Conv2d(in_channels=3,
                              out_channels=self.down_channels,
                              kernel_size=1)

            create_upblock = lambda c: [
                nn.Conv2d(
                    in_channels=c,
                    out_channels=c * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(num_features=c * 2),
                nn.ReLU()
            ]
            seq = []
            c = self.down_channels
            for _ in range(self.downsamplings):
                seq += create_upblock(c)
                c *= 2

            conv2 = nn.Conv2d(c, 2 * self.latent_size, 1)

            seq = [conv1] + seq + [conv2]
            self.seq = nn.Sequential(*seq)

        def forward(self, x):
            tmp = self.seq(x)
            m = tmp[:, :self.latent_size]
            s = torch.exp(tmp[:, self.latent_size:])
            return m, s

    class Decoder(nn.Module):
        def __init__(self, img_size=128, upsamplings=5, latent_size=512, up_channels=16):
            super().__init__()
            self.img_size = img_size
            self.upsamplings = upsamplings
            self.latent_size = latent_size
            self.up_channels = up_channels

            self.configure_model()

        def configure_model(self):
            conv1 = nn.Conv2d(in_channels=self.latent_size,
                              out_channels=self.up_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0
                              )
            create_upblock = lambda c: [
                nn.ConvTranspose2d(
                    in_channels=c,
                    out_channels=c // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(num_features=c // 2),
                nn.ReLU()
            ]
            seq = []
            c = self.up_channels * 2 ** self.upsamplings
            for _ in range(self.upsamplings):
                seq += create_upblock(c)
                c //= 2

            self.conv2 = nn.Conv2d(
                in_channels=c,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0
            )

            self.active = nn.Tanh()

            self.seq = [conv1] + seq
            self.seq = nn.Sequential(*seq)

        def forward(self, x):
            x = self.seq(x)
            # print(f'before: {x.shape}')
            x = self.conv2(x)
            # print(f'after: {x.shape}')
            x = self.active(x)
            return x

    def __init__(self, img_size=128, downsamplings=5, latent_size=512, down_channels=8, up_channels=16):
        super().__init__()
        self.encoder = self.Encoder(img_size, downsamplings, latent_size, down_channels)
        self.decoder = self.Decoder(img_size, downsamplings, latent_size, up_channels)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        x = mu + torch.randn_like(sigma) * sigma

        pred = self.decoder(x)
        kld = 1 / 2 * (mu ** 2 + sigma ** 2 - 2 * torch.log(sigma + 1e-16) - 1)

        # print(kld)

        return pred, kld

    def encode(self, x):
        mu, sigma = self.encoder(x)
        x = mu + torch.randn_like(sigma) * sigma
        return x

    def decode(self, z):
        return self.decoder(z)
