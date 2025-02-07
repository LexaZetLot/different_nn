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

class VGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.conv1_1(x)
        out = self.relu(out)
        out = self.conv1_2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv2_1(out)
        out = self.relu(out)
        out = self.conv2_2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv3_1(out)
        out = self.relu(out)
        out = self.conv3_2(out)
        out = self.relu(out)
        out = self.conv3_3(out)
        out = self.relu(out)
        out = self.conv3_4(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv4_1(out)
        out = self.relu(out)
        out = self.conv4_2(out)
        out = self.relu(out)
        out = self.conv4_3(out)
        out = self.relu(out)
        out = self.conv4_4(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv5_1(out)
        out = self.relu(out)
        out = self.conv5_2(out)
        out = self.relu(out)
        out = self.conv5_3(out)
        out = self.relu(out)
        out = self.conv5_4(out)
        out = self.maxpool(out)

        out = self.flatten(out)

        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)

        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)

        return out


model = VGG(2)
optimazer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimazer, mode='min', factor=0.1, patience=3)
loss_fn = torch.nn.CrossEntropyLoss()

def accuracy(pred, label):
    if label.ndim > 1:
        label = label.argmax(dim=1)

    pred_classes = F.softmax(pred.detach(), dim=1).argmax(dim=1)

    return ((pred_classes == label).float()).mean().item()


epochs = 100
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

torch.save(model.state_dict(), 'ZFNet.pth')