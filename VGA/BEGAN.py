import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

import numpy as np
import os
import matplotlib.pyplot as plt
import torch.autograd as autograd
from tqdm.autonotebook import tqdm


b1 = 0.5
b2 = 0.999
lr = 0.00005
imgSize = 32
channels = 1
nClasses = 10
batchSize = 64
latentDim = 100
nCritic = 5
clipValue = 0.01

imgShape = (channels, imgSize, imgSize)

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

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.initSize = imgSize // 4
        self.l1 = nn.Linear(latentDim, 128 * self.initSize ** 2)
        self.sec = nn.Sequential()
        self.upsample = nn.Upsample(scale_factor=2)
        self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        out = self.l1(x)
        out = self.sec(out)
        out = out.view(out.shape[0], 128, self.initSize, self.initSize)

        out = self.upsample(out)
        out = self.conv1(out)
        out = self.batchnorm1(out)
        out = self.leakyReLU(out)

        out = self.upsample(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.leakyReLU(out)

        out = self.conv3(out)
        out = self.tanh(out)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=2, padding=1)

        self.down_size = imgSize // 2
        downDim = 64 * (self.down_size // 2) ** 2
        self.fc1 = nn.Linear(downDim, 32)
        self.batchnorm1 = nn.BatchNorm1d(32, 0.8)

        self.fc2 = nn.Linear(32, downDim)
        self.batchnorm2 = nn.BatchNorm1d(downDim)

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.fc1(out.view(out.shape[0], -1))
        out = self.batchnorm1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)

        out = self.upsample(out.view(out.shape[0], 64, self.downSize, self.downSize))
        out = self.conv2(out)

        return out


generator = Generator()
discriminator = Discriminator()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

gamma = 0.75
lambda_k = 0.001
k = 0.0

for epoch in range(200):
    i = 0
    dLossVal = 0
    for img, label in (pbar := tqdm(dataloader)):
        batch_size = img.size(0)

        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latentDim).normal_(0, 1)

        gen = generator(z)
        g_loss = torch.mean(torch.abs(discriminator(gen) - gen))
        g_loss.backward()
        optimizer_G.step()

        dReal = discriminator(img)
        dFake = discriminator(gen.detach())

        dLossReal = torch.mean(torch.abs(dReal - img))
        dLossFake = torch.mean(torch.abs(dFake - gen.detach()))
        dLoss = dLossReal - k * dLossFake

        dLoss.backward()
        optimizer_D.step()

        diff = torch.mean(gamma * dLossReal - dLossFake)
        k = k + lambda_k * diff.item()
        k = min(max(k, 0), 1)
        i = gen[0]
        M = (dLossReal + torch.abs(diff)).data[0]

        pbar.set_description(f'epoch -> {epoch} ')
    plt.imshow(i.detach().numpy().squeeze(), cmap='gray')
    plt.show()

torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')