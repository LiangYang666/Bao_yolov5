#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :letter_train.py.py
# @Time      :2021/9/6 下午12:49
# @Author    :Yangliang
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torchvision.models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from baotools.letterClassification.model import get_model


class BaoClassicationDataset(Dataset):  # for training/testing
    def __init__(self, dataDir, transform):
        data = []
        label = []
        # classes = os.listdir(dataDir)
        for i, c in enumerate(['true', 'fake']):
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



def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir:
    mode: 'a', append; 'w',
    """
    import datetime

    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    else:
        log_file = log_file + '_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def cheeck_create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print(f'create the dir {dir}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Prototypical Networks Training')
    parser.add_argument('--epochs', type=str, default=1000)
    parser.add_argument('--test_inter', type=int, default=1)
    parser.add_argument('--save_inter', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--brand', type=str, default='Chanel')
    parser.add_argument('--part', type=str, default='sign')
    parser.add_argument('--letter', type=str, default='C')

    args = parser.parse_args()
    args.data_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{args.brand}/LetterDetection/data/{args.part}/classification_data/{args.letter}'
    args.run_data_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{args.brand}/LetterDetection/data/{args.part}/classification_rundata/{args.letter}'
    args.log_dir = os.path.join(args.run_data_dir, 'log')

    cheeck_create_dir(args.run_data_dir)
    cheeck_create_dir(args.log_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 模型
    model = get_model('resnet101')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)
    model.to(device=device)

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    args.criterion = criterion

    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    args.optimizer = optimizer

    # 数据集
    train_dataset = BaoClassicationDataset(os.path.join(args.data_dir, 'train'), transform=transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0, hue=0.3),
        transforms.ToTensor(),
        normalize,
    ]))
    train_loder = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_dataset = BaoClassicationDataset(os.path.join(args.data_dir, 'test'), transform=transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize,
    ]))
    test_loder = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 初始化log文件
    logging = init_logger(log_dir=args.log_dir, log_file=model.name)
    logging.info(f"\tmodel:{model.name}")
    logging.info(f"参数")
    message_args = str(args.__dict__)
    message_args = message_args.split('{', 1)[-1]
    message_args = message_args.rsplit('}', 1)[0]
    messages = message_args.split(',')
    for s in messages:
        logging.info(s.strip())

    # 开始训练
    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()
        total = 0
        correct = 0
        class_correct = [0] * 2
        class_total = [0] * 2
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

            # 计算各个类别的准确率
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_total[label] += 1
                class_correct[label] += c[i].item()
        message_train = f'Epoch:{epoch} Train accuracy: {100 * correct / total:3.1f}% {correct}/{total}' \
                        f'\ttrue:{(100 * class_correct[0] / class_total[0]):3.1f}%  {class_correct[0]}/{class_total[0]}'
        # 测试
        if epoch % args.test_inter == 0:
            with torch.no_grad():
                model.eval()

                total = 0
                correct = 0
                class_correct = [0] * 2
                class_total = [0] * 2

                for i, data in enumerate(test_loder):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    pred = model(inputs)
                    _, predicted = torch.max(pred.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # 计算各个类别的准确率
                    c = (predicted == labels).squeeze()
                    for i in range(labels.size(0)):
                        label = labels[i]
                        class_total[label] += 1
                        class_correct[label] += c[i].item()
                message_test = f'\tTest accuracy: {100 * correct / total:3.1f}% {correct}/{total}' \
                               f'\ttrue:{(100 * class_correct[0] / class_total[0]):3.1f}%  {class_correct[0]}/{class_total[0]} ' \
                               f'\tfake:{(100 * class_correct[1] / class_total[1]):3.1f}%  {class_correct[1]}/{class_total[1]}'
                logging.info(message_train+message_test)
