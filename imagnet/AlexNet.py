import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from torchgen.api.translate import out_tensor_ctype
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

        img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_AREA)
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

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(6 * 6 * 256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.relu(out)
        out = self.maxpool(out)



        out = self.flat(out)

        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)

        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)

        return out



model = AlexNet()
loss_fn = nn.CrossEntropyLoss()
optimazer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
schedule = torch.optim.lr_scheduler.StepLR(optimazer, step_size=30, gamma=0.1)

def accuracy(pred, label):
    if label.ndim > 1:
        label = label.argmax(dim=1)

    pred_classes = F.softmax(pred.detach(), dim=1).argmax(dim=1)

    correct = (pred_classes == label).float()
    return correct.mean().item()

epoch = 500
for i in range(epoch):
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

    schedule.step()


torch.save(model.state_dict(), "model_weights.pth")