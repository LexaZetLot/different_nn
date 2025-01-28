import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm


b1 = 0.5
b2 = 0.999
lr = 0.0002
imgSize = 28
channels = 1
batchSize = 64
latentDim = 100


imgShape = (channels, imgSize, imgSize)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

        self.linear1 = nn.Linear(latentDim, 128)

        self.linear2 = nn.Linear(128, 256)
        self.batchNorm2 = nn.BatchNorm1d(256)

        self.linear3 = nn.Linear(256, 512)
        self.batchNorm3 = nn.BatchNorm1d(512)

        self.linear4 = nn.Linear(512, 1024)
        self.batchNorm4 = nn.BatchNorm1d(1024)

        self.linear5 = nn.Linear(1024, int(np.prod(imgShape)))

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)

        out = self.linear2(out)
        out = self.batchNorm2(out)
        out = self.relu(out)

        out = self.linear3(out)
        out = self.batchNorm3(out)
        out = self.relu(out)

        out = self.linear4(out)
        out = self.batchNorm4(out)
        out = self.relu(out)

        out = self.linear5(out)
        out = self.tanh(out)

        return out.view(out.size(0), *imgShape)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.linear1 = nn.Linear(int(np.prod(imgShape)), 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, x):
        out = x.view(x.size(0), -1)

        out = self.linear1(out)
        out = self.relu(out)

        out = self.linear2(out)
        out = self.sigmoid(out)

        out = self.linear3(out)
        out = self.sigmoid(out)

        return out

adversarial_loss = torch.nn.BCELoss()
generator = Generator()
discriminator = Discriminator()

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

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

for epoch in range(200):
    i = 0
    for img, label in (pbar := tqdm(dataloader)):
        batch_size = img.size(0)
        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)

        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latentDim)
        gen_imgs = generator(z)
        i = gen_imgs[0]
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(img), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        pbar.set_description(f'epoch -> {epoch}')
    plt.imshow(i.detach().numpy().squeeze(), cmap='gray')
    plt.show()

torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')