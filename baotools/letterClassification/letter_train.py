#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :letter_train.py.py
# @Time      :2021/9/6 下午12:49
# @Author    :Yangliang
import os
import torch
import torchvision.models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class BaoClassicationDataset(Dataset):  # for training/testing
    def __init__(self, dataDir, transform):
        data = []
        label = []
        classes = os.listdir(dataDir)
        for i, c in enumerate(classes):
            img_names = os.listdir(os.path.join(dataDir, c))
            img_path = [os.path.join(dataDir, c, x) for x in img_names]
            data += img_path
            label += [i]*len(img_path)

        self.data = data
        self.labels = label
        self.transform = transform
        # print(self.data)
        # print(self.labels)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, i):
        path, label = self.data[i], self.labels[i]
        image = Image.open(path)
        w, h = image.size
        size = max(w, h)
        background = Image.new('RGB', (size, size), (127, 127, 127))
        # 矩形填充
        background.paste(image, ((size - w) // 2, (size - h) // 2))
        image = self.transform(background)
        return image, label

if __name__ == "__main__":

    epochs = 1000
    test_inter = 20
    save_inter = 100
    image_size = 640
    batch_size = 8
    brand = 'Chanel'
    part = 'sign'
    letter = 'C'
    data_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/classification_data/{letter}'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 模型
    model = torchvision.models.resnet34(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, out_features=2)
    # model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)
    model.to(device=device)

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 数据集
    train_dataset = BaoClassicationDataset(os.path.join(data_dir, 'train'), transform=transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ]))
    train_loder = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = BaoClassicationDataset(os.path.join(data_dir, 'test'), transform=transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ]))
    test_loder = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        total = 0
        correct = 0
        for i, data in enumerate(train_loder):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Epoch:{epoch} Train accuracy: {100 * correct / total}%')


        if epoch % test_inter == 0:
            with torch.no_grad():
                model.eval()
                for i, data in enumerate(test_loder):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    pred = model(inputs)
                    _, predicted = torch.max(pred.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print(f'*****------*****Test accuracy: {100 * correct / total}%')