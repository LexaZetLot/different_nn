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

        self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.25)

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32, 0.8)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64, 0.8)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128, 0.8)

        dsSize = imgSize // 2 ** 4
        self.fc1 = nn.Linear(128 * dsSize ** 2, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.leakyReLU(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.leakyReLU(out)
        out = self.dropout(out)
        out = self.batchnorm2(out)

        out = self.conv3(out)
        out = self.leakyReLU(out)
        out = self.dropout(out)
        out = self.batchnorm3(out)

        out = self.conv4(out)
        out = self.leakyReLU(out)
        out = self.dropout(out)
        out = self.batchnorm4(out)

        out = out.view(out.shape[0], -1)
        out = self.fc1(out)

        return out

adversarial_loss = torch.nn.MSELoss()

generator = Generator()
discriminator = Discriminator()

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
