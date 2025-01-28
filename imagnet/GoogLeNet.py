import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm


class DataSet2Class(torch.utils.data.Dataset):
    def __init__(self, path_dir1: str, path_dir2: str):
        super().__init__()

        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2

        self.dir1_list = sorted(os.listdir(path_dir1))
        self.dir2_list = sorted(os.listdir(path_dir2))

    def __len__(self):
        return len(self.dir1_list) + len(self.dir2_list)

    def __getitem__(self, idx):
        if idx < len(self.dir1_list):
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.dir1_list[idx])
        else:
            class_id = 1
            idx -= len(self.dir1_list)
            img_path = os.path.join(self.path_dir2, self.dir2_list[idx])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img / 255.0

        img = cv2.resize(img, (299, 299), interpolation=cv2.INTER_AREA)
        img = img.transpose((2, 0, 1))

        t_img = torch.from_numpy(img)
        t_class_id = torch.tensor([class_id])

        return {'img': t_img, 'label': t_class_id}

trainDogsPath = '/home/lexa/Desktop/pythonProject/dataCatDogs/training_set/training_set/dogs/'
trainCatsPath = '/home/lexa/Desktop/pythonProject/dataCatDogs/training_set/training_set/cats/'
testDogsPath = '/home/lexa/Desktop/pythonProject/dataCatDogs/test_set/test_set/dogs/'
testCatsPath = '/home/lexa/Desktop/pythonProject/dataCatDogs/test_set/test_set/dogs/'

DataSetCatsDogs = DataSet2Class(trainDogsPath, trainCatsPath)
DataSetTrainCatsDogs = DataSet2Class(testDogsPath, testCatsPath)

batch_size = 16
trainLoader = torch.utils.data.DataLoader(DataSetCatsDogs, shuffle=True,
                                          batch_size=batch_size, num_workers=0,
                                          drop_last=True)
testLoader = torch.utils.data.DataLoader(DataSetTrainCatsDogs, shuffle=True,
                                         batch_size=batch_size, num_workers=0,
                                         drop_last=False)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
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
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


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
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


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

        return torch.cat([branch1, branch2, self.branch3(x), self.branch4(x)], dim=1)


class ReductionBlockA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Sequential(ConvBlock(channels, 192, kernel_size=1, stride=1, padding=0),
                                     ConvBlock(192, 224, kernel_size=3, stride=1, padding=1),
                                     ConvBlock(224, 256, kernel_size=3, stride=2, padding=0),
                                     )
        self.branch2 = ConvBlock(channels, 384, kernel_size=3, stride=2, padding=0)
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class ReductionBlockB(nn.Module):
    def __init__(self, channels):
        super().__init__()

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
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class InterfaceInceptionBlockA(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()

        self.seq = nn.Sequential()
        for i in range(depth):
            self.seq.add_module(f'inceptionBlockA{i}', InceptionBlockA(channels=channels))
    def forward(self, x):
        return self.seq(x)


class InterfaceInceptionBlockB(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()

        self.seq = nn.Sequential()
        for i in range(depth):
            self.seq.add_module(f'inceptionBlockA{i}', InceptionBlockB(channels=channels))
    def forward(self, x):
        return self.seq(x)


class InterfaceInceptionBlockC(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()

        self.seq = nn.Sequential()
        for i in range(depth):
            self.seq.add_module(f'inceptionBlockA{i}', InceptionBlockC(channels=channels))
    def forward(self, x):
        return self.seq(x)


class GoogLeNet(nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.globalPool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

        self.stem = Stem(3)
        self.interfaceInceptionBlockA = InterfaceInceptionBlockA(channels=384, depth=4)
        self.reductionBlockA = ReductionBlockA(384)
        self.interfaceInceptionBlockB = InterfaceInceptionBlockB(channels=1024, depth=7)
        self.reductionBlockB = ReductionBlockB(1024)
        self.interfaceInceptionBlockC = InterfaceInceptionBlockC(channels=1536, depth=3)

        self.fc1 = nn.Linear(1536, 1536)
        self.fc2 = nn.Linear(1536, labels)

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

        return self.softmax(out)


model = GoogLeNet(2)
optimazer = torch.optim.RMSprop(model.parameters(),
                                lr=0.045,
                                alpha=0.9,
                                momentum=0.9,
                                weight_decay=4e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimazer,gamma=0.94)
loss_fn = torch.nn.CrossEntropyLoss()

def accuracy(pred, label):
    if label.ndim > 1:
        label = label.argmax(dim=1)

    pred_classes = pred.detach().argmax(dim=1)

    return ((pred_classes == label).float()).mean().item()

print(f'parameters -> {sum(p.numel() for p in model.parameters())}')

epochs = 90
for i in range(epochs):
    loss_val = 0
    acc_val = 0
    for sample in (pbar := tqdm(trainLoader)):
        img, label = sample['img'], sample['label'].squeeze(1)
        optimazer.zero_grad()

        pred = model(img)

        loss = loss_fn(pred, label)
        loss_val += loss.item()
        loss.backward()
        optimazer.step()

        acc_val += accuracy(pred, label)

        pbar.set_description(f"epoch -> {i + 1}\tloss -> {loss_val / len(trainLoader):.5f}\taccuracy -> {acc_val / len(trainLoader):.5f}")

    scheduler.step()

torch.save(model.state_dict(), 'GoogLeNet.pth')