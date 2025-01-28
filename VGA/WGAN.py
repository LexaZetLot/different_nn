import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv
from torch import Tensor

from torch.autograd import Variable

import numpy as np
import os
import matplotlib.pyplot as plt
from sympy.physics.paulialgebra import epsilon
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

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

        self.linear1 = nn.Linear(latentDim, 128)

        self.linear2 = nn.Linear(128, 256)
        self.batchNorm1 = nn.BatchNorm1d(256)

        self.linear3 = nn.Linear(256, 512)
        self.batchNorm2 = nn.BatchNorm1d(512)

        self.linear4 = nn.Linear(512, 1024)
        self.batchNorm3 = nn.BatchNorm1d(1024)

        self.linear5 = nn.Linear(1024, int(np.prod(imgShape)))

    def forward(self, x):
        out = self.linear1(x)
        out = self.leakyReLU(out)

        out = self.linear2(out)
        out = self.batchNorm1(out)
        out = self.leakyReLU(out)

        out = self.linear3(out)
        out = self.batchNorm2(out)
        out = self.leakyReLU(out)

        out = self.linear4(out)
        out = self.batchNorm3(out)
        out = self.leakyReLU(out)

        out = self.linear5(out)
        out = self.tanh(out)

        return out.view(out.shape[0], *imgShape)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)

        self.linear1 = nn.Linear(int(np.prod(imgShape)), 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, x):
        out = self.linear1(x.view(x.shape[0], -1))
        out = self.leakyReLU(out)
        out = self.linear2(out)
        out = self.leakyReLU(out)
        out = self.linear3(out)
        return out


generator = Generator()
discriminator = Discriminator()

optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

for epoch in range(200):
    i = 0
    imgPlt = None
    for img, _ in (pbar := tqdm(dataloader)):
        batch_size = img.shape[0]

        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latentDim)
        fakeImg = generator(z)
        lossD = -torch.mean(discriminator(img)) + torch.mean(discriminator(fakeImg))
        lossD.backward()
        optimizer_D.step()

        for p in discriminator.parameters():
            p.data.clamp_(min=-clipValue, max=clipValue)

        if i % nCritic == 0:
            optimizer_G.zero_grad()
            genImg = generator(z)
            lossG = -torch.mean(discriminator(genImg))

            imgPlt = genImg[0]
            lossG.backward()
            optimizer_G.step()

        i += 1

        pbar.set_description(f'epoch -> {epoch}')

    plt.imshow(imgPlt.detach().numpy().squeeze(), cmap='gray')
    plt.show()
