import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

from WGAN import nCritic

b1 = 0.5
b2 = 0.999
lr = 0.0002
imgSize = 64
channels = 1
nClasses = 10
batchSize = 64
latentDim = 100

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

        self.labelEmd = nn.Embedding(nClasses, nClasses)
        self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

        self.linear1 = nn.Linear(latentDim + nClasses, 128)

        self.linear2 = nn.Linear(128, 256)
        self.batchNorm1 = nn.BatchNorm1d(256)

        self.linear3 = nn.Linear(256, 512)
        self.batchNorm2 = nn.BatchNorm1d(512)

        self.linear4 = nn.Linear(512, 1024)
        self.batchNorm3 = nn.BatchNorm1d(1024)

        self.linear5 = nn.Linear(1024, int(np.prod(imgShape)))


    def forward(self, x, labels):
        out = torch.cat([self.labelEmd(labels), x], -1)
        out = self.linear1(out)
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

        out = out.view(out.size(0), *imgShape)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.labelEmd = nn.Embedding(nClasses, nClasses)
        self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.4)

        self.linear1 = nn.Linear(nClasses + int(np.prod(imgShape)), 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 1)


    def forward(self, x, labels):
        out = torch.cat([self.labelEmd(labels), x.view(x.size(0), -1)], -1)

        out = self.linear1(out)
        out = self.leakyReLU(out)

        out = self.linear2(out)
        out = self.dropout(out)
        out = self.leakyReLU(out)

        out = self.linear3(out)
        out = self.dropout(out)
        out = self.leakyReLU(out)

        out = self.linear4(out)

        return out


adversarial_loss = torch.nn.MSELoss()

generator = Generator()
discriminator = Discriminator()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
for epoch in range(200):
    i = 0
    for img, labels in (pbar := tqdm(dataloader)):
        batch_size = img.size(0)

        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)

        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latentDim)
        gen_labels = torch.randint(0, nClasses, (batch_size,))

        gen_imgs = generator(z, gen_labels)
        i = gen_imgs[0]
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(img, labels), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()


        pbar.set_description(f'epoch -> {epoch}')
    plt.imshow(i.detach().numpy().squeeze(), cmap='gray')
    plt.show()

