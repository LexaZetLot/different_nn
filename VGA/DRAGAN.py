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

        self.leakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = nn.Sigmoid()
        self.sequential = nn.Sequential()

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
        out = self.sigmoid(out)
        out = self.sequential(out)

        return out

adversarial_loss = nn.BCELoss()

lambda_gp = 10

generator = Generator()
discriminator = Discriminator()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

def computeGradientPenalty(discriminator, real):
    alpha = torch.rand(size=real.shape)

    interpolates = (alpha * real + ((1 - alpha) * (real + 0.5 * real.std() * torch.rand(real.shape)))).requires_grad_(True)
    disc_interpolates = discriminator(interpolates)
    fake = torch.tensor(1.0, dtype=disc_interpolates.dtype, device=disc_interpolates.device).expand(real.shape[0], 1).requires_grad_(False)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

for epoch in range(200):
    i = 0
    dLossVal = 0
    for img, label in (pbar := tqdm(dataloader)):
        batch_size = img.size(0)
        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)


        discriminator.zero_grad()
        predReal = discriminator(img)
        lossDReal = adversarial_loss(predReal, valid)
        lossDReal.backward()

        z = torch.randn(batch_size, latentDim).normal_(0, 1)
        fakeImg = generator(z).detach()
        predFake = discriminator(fakeImg)
        lossDFake = adversarial_loss(predFake, fake)
        lossDFake.backward()

        alpha = torch.rand(size=img.size())
        interpolates = (alpha * img.data + ((1 - alpha) * (img.data + 0.5 * img.data.std() * torch.rand(img.data.shape)))).requires_grad_(True)
        predHat = discriminator(interpolates)
        gradients = autograd.grad(outputs=predHat,
                                  inputs=interpolates,
                                  grad_outputs=torch.ones(predHat.size()),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True,)[0]
        gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty.backward()

        lossD = lossDReal + lossDFake + gradient_penalty
        dLossVal += lossD
        optimizer_D.step()

        generator.zero_grad()
        z = torch.randn(batch_size, latentDim).normal_(0, 1)
        gen = generator(z)
        i = gen[0]
        predGen = discriminator(gen)
        lossG = adversarial_loss(predGen, valid)
        lossG.backward()
        optimizer_G.step()


        pbar.set_description(f'epoch -> {epoch} \t d_loss -> {(dLossVal / len(dataloader)):.4f}')
    plt.imshow(i.detach().numpy().squeeze(), cmap='gray')
    plt.show()

torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
