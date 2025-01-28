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

        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
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
class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)


        self.conv1 = nn.Conv2d(in_channel, out_channel // 4, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel // 4, out_channel, kernel_size=1, stride=1, padding=0)

        self.convOut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if out.shape[1] != x.shape[1]:
            x = self.convOut(x)
        return torch.add(out, x)

class Layer(nn.Module):
    def __init__(self, in_channel, basic_channel, deeph):
        super().__init__()

        self.seq = nn.Sequential()

        self.seq.add_module('bc1', Bottleneck(in_channel, basic_channel * 4))
        for i in range(2, deeph + 1):
            self.seq.add_module(f'bc{i}', Bottleneck(basic_channel * 4, basic_channel * 4) )
            self.seq.add_module(f'relu{i}', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.seq(x)

class PredLayer(nn.Module):
    def __init__(self, in_channel, basic_channel):
        super().__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channel, basic_channel, kernel_size=1, stride=2, padding=0)
        self.conv2 = nn.Conv2d(basic_channel, basic_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(basic_channel, basic_channel, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channel, basic_channel, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = torch.add(self.conv4(x), out)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        self.layer1 = Layer(64, 64, 3)

        self.predLayer2 = PredLayer(256, 128)
        self.layer2 = Layer(128, 128, 7)

        self.predLayer3 = PredLayer(512, 256)
        self.layer3 = Layer(256, 256, 35)

        self.predLayer4 = PredLayer(1024, 512)
        self.layer4 = Layer(512, 512, 3)

        self.fc1 = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)

        out = self.predLayer2(out)
        out = self.layer2(out)

        out = self.predLayer3(out)
        out = self.layer3(out)

        out = self.predLayer4(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc1(out)

        return out

model = ResNet(2)
optimazer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
schedule = torch.optim.lr_scheduler.StepLR(optimazer, step_size=30, gamma=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

def accuracy(pred, label):
    if label.ndim > 1:
        label = label.argmax(dim=1)

    pred_classes = F.softmax(pred.detach(), dim=1).argmax(dim=1)

    return ((pred_classes == label).float()).mean().item()

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

    schedule.step(loss_val)

torch.save(model.state_dict(), 'ResNet.pth')