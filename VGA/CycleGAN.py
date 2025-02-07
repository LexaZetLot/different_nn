import itertools
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision as tv

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

b1 = 0.5
b2 = 0.999
lr = 0.0002
imgSize = 64
channels = 1
batchSize = 64
latentDim = 100
nEpochs = 200
lambdaCyc = 10.0
lambdaId = 5.0


class DataSet2Class(torch.utils.data.Dataset):
    def __init__(self, path_dir1: str, path_dir2: str):
        super().__init__()

        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2

        self.dir1_list = sorted(os.listdir(path_dir1))
        self.dir2_list = sorted(os.listdir(path_dir2))

    def __len__(self):
        return min(len(self.dir1_list), len(self.dir2_list))

    def __getitem__(self, idx):
        img_pathA = os.path.join(self.path_dir1, self.dir1_list[idx])
        img_pathB = os.path.join(self.path_dir2, self.dir2_list[idx])

        imgA = cv2.imread(img_pathA, cv2.IMREAD_COLOR)
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
        imgA = imgA.astype(np.float32)
        imgA = imgA / 255.0

        imgA = cv2.resize(imgA, (300, 300), interpolation=cv2.INTER_AREA)
        imgA = imgA.transpose((2, 0, 1))

        imgB = cv2.imread(img_pathB, cv2.IMREAD_COLOR)
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
        imgB = imgB.astype(np.float32)
        imgB = imgB / 255.0

        imgB = cv2.resize(imgB, (300, 300), interpolation=cv2.INTER_AREA)
        imgB = imgB.transpose((2, 0, 1))

        t_imgA = torch.from_numpy(imgA)
        t_imgB = torch.from_numpy(imgB)

        return {'imgA': t_imgA, 'imgB': t_imgB}

trainDogsPath = '/home/lexa/Desktop/pythonProject/horse2zebra/trainA/'
trainCatsPath = '/home/lexa/Desktop/pythonProject/horse2zebra/trainB/'


batch_size = 16
DataSetCatsDogs = DataSet2Class(trainDogsPath, trainCatsPath)
trainLoader = torch.utils.data.DataLoader(DataSetCatsDogs, shuffle=True,
                                          batch_size=batch_size, num_workers=0,
                                          drop_last=True)

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction_ratio)
        self.fc2 = nn.Linear(channels // reduction_ratio, channels)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        squeeze = self.global_avg_pool(x).view(batch_size, channels)

        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation)).view(batch_size, channels, 1, 1)

        scaled_output = x * excitation
        return scaled_output

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitationBlock(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.se(out)
        out = self.relu(out)
        return out

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitationBlock(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.se(out)
        out = self.relu(out)
        return out


class Stem(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = ConvBlock(channels, 32, kernel_size=3, stride=2, padding=0)
        self.conv2 = ConvBlock(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)

        self.conv4 = ConvBlock(64, 96, kernel_size=3, stride=2, padding=0)
        self.maxPool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.branch1_1 = nn.Sequential(
            ConvBlock(160, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock(64, 96, kernel_size=3, stride=1, padding=0),
        )
        self.branch1_2 = nn.Sequential(
            ConvBlock(160, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock(64, 96, kernel_size=3, stride=1, padding=0),
        )

        self.branch2_1 = ConvBlock(192, 192, 3, 2, 0)
        self.branch2_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = torch.cat([self.conv4(out), self.maxPool4(out)], dim=1)

        out = torch.cat([self.branch1_1(out), self.branch1_2(out)], dim=1)

        out = torch.cat([self.branch2_1(out), self.branch2_2(out)], dim=1)
        return out

class StemRevers(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1_1 =  ConvTransposeBlock(channels // 2, 96, kernel_size=3, stride=2, padding=0)
        self.conv1_2 = nn.Sequential(nn.Upsample(size=(71, 71), mode='bilinear', align_corners=True),
                                     nn.Conv2d(channels // 2, 96, kernel_size=1))

        self.conv2_1 = nn.Sequential(
            ConvBlock(96, 96, kernel_size=1, stride=1, padding=1),
            ConvBlock(96, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
        )

        self.conv2_2 = nn.Sequential(
            ConvBlock(96, 96, kernel_size=1, stride=1, padding=1),
            ConvBlock(96, 96, kernel_size=3, stride=1, padding=1),
        )

        self.conv3_1 = ConvTransposeBlock(80, 32, kernel_size=3, stride=2, padding=0)
        self.conv3_2 = nn.Sequential(nn.Upsample(size=(147, 147), mode='bilinear', align_corners=True),
                                     nn.Conv2d(80, 32, kernel_size=1))

        self.conv4 = ConvTransposeBlock(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvTransposeBlock(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv6 = ConvTransposeBlock(32, 3, kernel_size=3, stride=2, padding=0, output_padding=1)

    def forward(self, x):
        out = torch.cat([self.conv1_1(x[:, :x.shape[1] // 2, :, :]), self.conv1_2(x[:, :x.shape[1] // 2, :, :])], dim=1)

        out = torch.cat([self.conv2_1(out[:, :out.shape[1] // 2, :, :]), self.conv2_2(out[:, :out.shape[1] // 2, :, :])], dim=1)

        out = torch.cat([self.conv3_1(out[:, :out.shape[1] // 2, :, :]), self.conv3_2(out[:, :out.shape[1] // 2, :, :])], dim=1)

        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        return out


class InceptionBlockA(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(channels, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock(64, 96, kernel_size=3, stride=1, padding=1),
            ConvBlock(96, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            ConvBlock(channels, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock(64, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(channels, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch4 = ConvBlock(channels, 96, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return torch.add(x, torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1))

class InceptionBlockB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(channels, 192 , kernel_size=1, stride=1, padding=0),
            ConvBlock(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
        )

        self.branch2 = nn.Sequential(
            ConvBlock(channels, 192, kernel_size=1, stride=1, padding=0),
            ConvBlock(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )

        self.branch3 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                     ConvBlock(channels, 128, kernel_size=1, stride=1, padding=0),
                                     )

        self.branch4 = ConvBlock(channels, 384, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return torch.add(x, torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1))

class InceptionBlockC(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(channels, 384, kernel_size=1, stride=1, padding=0),
            ConvBlock(384, 512, kernel_size=3, stride=1, padding=1),
        )
        self.branch1_1 = ConvBlock(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_2 = ConvBlock(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch2 = ConvBlock(channels, 384, kernel_size=1, stride=1, padding=0)
        self.branch2_1 = ConvBlock(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_2 = ConvBlock(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(channels, 256, kernel_size=3, stride=1, padding=1),
        )

        self.branch4 = ConvBlock(channels, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch1 = torch.cat([self.branch1_1(branch1), self.branch1_2(branch1)], dim=1)

        branch2 = self.branch2(x)
        branch2 = torch.cat([self.branch2_1(branch2), self.branch2_2(branch2)], dim=1)

        return torch.add(x, torch.cat([branch1, branch2, self.branch3(x), self.branch4(x)], dim=1))

class ReductionBlockA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.branch1 = nn.Sequential(ConvBlock(channels, 192, kernel_size=1, stride=1, padding=0),
                                     ConvBlock(192, 224, kernel_size=3, stride=1, padding=1),
                                     ConvBlock(224, 256, kernel_size=3, stride=2, padding=0),
                                     )
        self.branch2 = ConvBlock(channels, 384, kernel_size=3, stride=2, padding=0)
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.relu(torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1))

class ReductionBlockB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.branch1 = nn.Sequential(ConvBlock(channels, 256, kernel_size=1, stride=1, padding=0),
                                     ConvBlock(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
                                     ConvBlock(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)),
                                     ConvBlock(320, 320, kernel_size=3, stride=2, padding=0),
                                     )

        self.branch2 = nn.Sequential(
            ConvBlock(channels, 192, kernel_size=1, stride=1, padding=0),
            ConvBlock(192, 192, kernel_size=3, stride=2, padding=0),
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.relu(torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1))


class ReductionBlockARevers(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.branch1 = nn.Sequential(ConvTransposeBlock(channels, 256, kernel_size=1, stride=1, padding=0),
                                     ConvTransposeBlock(256, 224, kernel_size=3, stride=1, padding=1),
                                     ConvTransposeBlock(224, 192, kernel_size=3, stride=2, padding=0),
                                     )
        self.branch2 = ConvTransposeBlock(channels, 96, kernel_size=3, stride=2, padding=0)
        self.branch3 = nn.Sequential(nn.Upsample(size=(35, 35), mode='bilinear', align_corners=True),
                                     nn.Conv2d(channels, 96, kernel_size=1))

    def forward(self, x):
        return self.relu(torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1))


class ReductionBlockBRevers(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.branch1 = nn.Sequential(ConvTransposeBlock(channels, 320, kernel_size=1, stride=1, padding=0),
                                     ConvTransposeBlock(320, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
                                     ConvTransposeBlock(256, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)),
                                     ConvTransposeBlock(256, 256, kernel_size=3, stride=2, padding=0),
                                     )

        self.branch2 = nn.Sequential(
            ConvTransposeBlock(channels, 256, kernel_size=1, stride=1, padding=0),
            ConvTransposeBlock(256, 256, kernel_size=3, stride=2, padding=0),
        )

        self.branch3 = nn.Sequential(nn.Upsample(size=(17, 17), mode='bilinear', align_corners=True),
                                     nn.Conv2d(1536, 512, kernel_size=1))

    def forward(self, x):
        return self.relu(torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1))



class InterfaceInceptionBlockA(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()

        self.seq = nn.Sequential()
        for i in range(depth):
            self.seq.add_module(f'inceptionBlockA{i}', InceptionBlockA(channels=channels))
            self.seq.add_module(f'relu{i}', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.seq(x)

class InterfaceInceptionBlockB(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()

        self.seq = nn.Sequential()
        for i in range(depth):
            self.seq.add_module(f'inceptionBlockA{i}', InceptionBlockB(channels=channels))
            self.seq.add_module(f'relu{i}', nn.ReLU(inplace=True))
    def forward(self, x):
        return self.seq(x)

class InterfaceInceptionBlockC(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()

        self.seq = nn.Sequential()
        for i in range(depth):
            self.seq.add_module(f'inceptionBlockA{i}', InceptionBlockC(channels=channels))
            self.seq.add_module(f'relu{i}', nn.ReLU(inplace=True))
    def forward(self, x):
        return self.seq(x)


class GoogLeNetSEGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.stem = Stem(3)
        self.interfaceInceptionBlockA = InterfaceInceptionBlockA(channels=384, depth=4)
        self.reductionBlockA = ReductionBlockA(channels=384)

        self.interfaceInceptionBlockB = InterfaceInceptionBlockB(channels=1024, depth=7)
        self.reductionBlockB = ReductionBlockB(channels=1024)

        self.interfaceInceptionBlockC = InterfaceInceptionBlockC(channels=1536, depth=6)
        self.reductionBlockBRevers = ReductionBlockBRevers(channels=1536)

        self.interfaceInceptionBlockBRevers = InterfaceInceptionBlockB(channels=1024, depth=7)
        self.reductionBlockARevers = ReductionBlockARevers(channels=1024)

        self.stemRevers = StemRevers(384)


    def forward(self, x):
        out = self.stem(x)
        out = self.relu(out)

        out = self.interfaceInceptionBlockA(out)
        out = self.reductionBlockA(out)

        out = self.interfaceInceptionBlockB(out)
        out = self.reductionBlockB(out)

        out = self.interfaceInceptionBlockC(out)
        out = self.reductionBlockBRevers(out)

        out = self.interfaceInceptionBlockBRevers(out)
        out = self.reductionBlockARevers(out)

        out = self.stemRevers(out)
        return out

class GoogLeNetSEDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.globalPool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

        self.stem = Stem(3)
        self.interfaceInceptionBlockA = InterfaceInceptionBlockA(channels=384, depth=4)
        self.reductionBlockA = ReductionBlockA(384)
        self.interfaceInceptionBlockB = InterfaceInceptionBlockB(channels=1024, depth=7)
        self.reductionBlockB = ReductionBlockB(1024)
        self.interfaceInceptionBlockC = InterfaceInceptionBlockC(channels=1536, depth=3)

        self.fc1 = nn.Linear(1536, 1536)
        self.fc2 = nn.Linear(1536, 1)

    def forward(self, x):
        out = self.stem(x)
        out = self.interfaceInceptionBlockA(out)
        out = self.reductionBlockA(out)

        out = self.interfaceInceptionBlockB(out)
        out = self.reductionBlockB(out)

        out = self.interfaceInceptionBlockC(out)

        out = self.globalPool(out)
        out = self.flatten(out)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.fc2(out)

        return self.sig(out)

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

generatorA = GoogLeNetSEGenerator()
generatorB = GoogLeNetSEGenerator()
discriminatorA = GoogLeNetSEDiscriminator()
discriminatorB = GoogLeNetSEDiscriminator()

optimizerG = torch.optim.Adam(itertools.chain(generatorA.parameters(), generatorB.parameters()), lr=lr, betas=(b1, b2))
optimizerDA = torch.optim.Adam(discriminatorA.parameters(), lr=lr, betas=(b1, b2))
optimizerDB = torch.optim.Adam(discriminatorB.parameters(), lr=lr, betas=(b1, b2))

lrSchedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=LambdaLR(nEpochs, 0, 100).step)
lrSchedulerDA = torch.optim.lr_scheduler.LambdaLR(optimizerDA, lr_lambda=LambdaLR(nEpochs, 0, 100).step)
lrSchedulerDB = torch.optim.lr_scheduler.LambdaLR(optimizerDB, lr_lambda=LambdaLR(nEpochs, 0, 100).step)

criterionGAN = torch.nn.MSELoss()
criterionCycle = torch.nn.L1Loss()
criterionIdentity = torch.nn.L1Loss()

fakeABuffer = ReplayBuffer()
fakeBBuffer = ReplayBuffer()

print(sum(p.numel() for p in generatorA.parameters()) +
      sum(p.numel() for p in generatorA.parameters()) +
      sum(p.numel() for p in discriminatorA.parameters()) +
      sum(p.numel() for p in discriminatorB.parameters()))

for epoch in range(nEpochs):
    img = []
    for sample in (pbar := tqdm(trainLoader)):
        imgA, imgB = sample['imgA'], sample['imgB']

        valid = torch.ones(batch_size, 3, 300, 300).requires_grad_(False)
        fake = torch.zeros(batch_size, 3, 300, 300).requires_grad_(False)

        generatorA.train()
        generatorB.train()
        optimizerG.zero_grad()

        lossIdA = criterionIdentity(generatorB(imgA), imgA)
        lossIdB = criterionIdentity(generatorA(imgB), imgB)
        lossIdentity = (lossIdA + lossIdB) / 2

        fakeB = generatorA(imgA)
        lossGANA = criterionGAN(discriminatorB(fakeB), valid)
        fakeA = generatorB(imgB)
        lossGANB = criterionGAN(discriminatorA(fakeA), valid)
        lossGAN = (lossGANA + lossGANB) / 2

        recovA = generatorB(imgB)
        lossCycleA = criterionCycle(recovA, imgA)
        recovB = generatorA(imgA)
        lossCycleB = criterionCycle(recovB, imgB)
        lossCycle = (lossCycleA + lossCycleB) / 2
        img.append(imgA)
        img.append(imgB)

        lossG = lossGAN + lambdaCyc * lossCycle + lambdaId * lossIdentity
        lossG.backward()
        optimizerG.step()

        optimizerDA.zero_grad()
        lossReal = criterionGAN(discriminatorA(imgA), valid)
        fakeA_ = fakeABuffer.push_and_pop(fakeA)
        lossFake = criterionGAN(fakeA_.detach(), fake)
        lossDA = (lossReal + lossFake) / 2
        lossDA.backward()
        optimizerDA.step()

        optimizerDB.zero_grad()
        lossReal = criterionGAN(discriminatorB(imgB), valid)
        fakeB_ = fakeBBuffer.push_and_pop(fakeB)
        lossFake = criterionGAN(fakeB_.detach(), fake)
        lossDB = (lossReal + lossFake) / 2
        lossDB.backward()
        optimizerDB.step()

        lossD = (lossDA + lossDB) / 2

        pbar.set_description(f'epoch -> {epoch}\tlossD -> {lossD:.4f}\tlossG -> {lossG:.4f}' )

    lrSchedulerG.step()
    lrSchedulerDA.step()
    lrSchedulerDB.step()

    plt.imshow(img[0].detach().numpy().squeeze(), cmap='gray')
    plt.imshow(img[1].detach().numpy().squeeze(), cmap='gray')
    plt.show()
    img.clear()