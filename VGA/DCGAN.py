import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

b1 = 0.5
b2 = 0.999
lr = 0.0002
imgSize = 64
channels = 1
batchSize = 64
latentDim = 100


dataloader = torch.utils.data.DataLoader(
    tv.datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=tv.transforms.Compose(
            [tv.transforms.Resize(imgSize), tv.transforms.ToTensor(), tv.transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batchSize,
    shuffle=True,
)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.reLU = nn.ReLU(True)
        self.tanh = nn.Tanh()

        self.convTranspose1 = nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(512)

        self.convTranspose2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(256)

        self.convTranspose3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.convTranspose4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.convTranspose5 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        out = self.convTranspose1(x)
        out = self.batchnorm1(out)
        out = self.reLU(out)

        out = self.convTranspose2(out)
        out = self.batchnorm2(out)
        out = self.reLU(out)

        out = self.convTranspose3(out)
        out = self.batchnorm3(out)
        out = self.reLU(out)

        out = self.convTranspose4(out)
        out = self.batchnorm4(out)
        out = self.reLU(out)

        out = self.convTranspose5(out)
        out = self.tanh(out)

        return out




class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.leakyReLU(out)

        out = self.conv2(out)
        out = self.batchnorm1(out)
        out = self.leakyReLU(out)

        out = self.conv3(out)
        out = self.batchnorm2(out)
        out = self.leakyReLU(out)

        out = self.conv4(out)
        out = self.batchnorm3(out)
        out = self.leakyReLU(out)

        out = self.conv5(out)
        out = self.sigmoid(out)

        out = self.flatten(out)
        return out

generator = Generator()
discriminator = Discriminator()

adversarial_loss = torch.nn.BCELoss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

for epoch in range(200):
    i = 0
    for img, label in (pbar := tqdm(dataloader)):
        batch_size = img.size(0)
        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)

        optimizer_G.zero_grad()

        z = torch.randn(batch_size, latentDim, 1, 1)
        genImg = generator(z)
        i = genImg[0]
        g_loss = adversarial_loss(discriminator(genImg), valid)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(img), valid)
        fake_loss = adversarial_loss(discriminator(genImg.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        pbar.set_description(f'epoch -> {epoch}')
    plt.imshow(i.detach().numpy().squeeze(), cmap='gray')
    plt.show()


torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
