#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :model.py
# @Time      :2021/9/7 下午12:43
# @Author    :Yangliang
import torch
import torchvision
import torch.nn as nn
import timm


class FirstFeature(nn.Module):
    def __init__(self, image_size, output_channels=128):
        super(FirstFeature, self).__init__()
        self.output_channels = output_channels
        self.conv11 = nn.Conv2d(3, self.output_channels, 11)
        # self.conv12 = nn.Conv2d(3, self.output_channels, 21, padding=5)
        # self.conv13 = nn.Conv2d(3, self.output_channels, 21, padding=15, dilation=2)
        self.batchnorm = nn.BatchNorm2d(self.output_channels)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # out11 = self.maxpool(self.leakyrelu(self.conv11(x)))
        out11 = self.maxpool(self.leakyrelu(self.batchnorm(self.conv11(x))))
        # out12 = self.maxpool(self.leakyrelu(self.batchnorm(self.conv12(x))))
        # out13 = self.maxpool(self.leakyrelu(self.batchnorm(self.conv13(x))))
        # return out11 + out12 + out13
        return out11


class SecondFeature(nn.Module):
    def __init__(self, in_channels=128):
        super(SecondFeature, self).__init__()
        self.in_channels = in_channels
        self.conv21 = nn.Conv2d(self.in_channels, self.in_channels, 7, padding=1)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv22 = nn.Conv2d(self.in_channels, self.in_channels, 5, padding=2)
        self.conv23 = nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1)
        self.conv30 = nn.Conv2d(self.in_channels, self.in_channels, 5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x0 = self.leakyrelu(self.conv30(x))
        x = self.leakyrelu(self.conv21(x))
        x = self.leakyrelu(self.conv22(x))
        x = self.leakyrelu(self.conv23(x))
        return self.maxpool(x0+x)


class ThirdFeature(nn.Module):
    def __init__(self, in_channels=128, num_classes=2):
        super(ThirdFeature, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, stride=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, stride=3)
        self.conv4 = nn.Conv2d(in_channels, in_channels, 3)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_channels * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MyNet1(nn.Module):
    def __init__(self):
        super(MyNet1, self).__init__()
        self.first = FirstFeature(400, 128)
        self.second = SecondFeature(128)
        self.third = ThirdFeature(128)

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        x = self.third(x)
        return x


class MyNet2(nn.Module):
    def __init__(self, num_classes=1000):
        super(MyNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def get_model(name='resnet34', pretrained=False, num_classes=2):

    # 模型
    if name.startswith('resnet'):
        if name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
            torchvision.models.resnet18()
            model = eval(f'torchvision.models.{name}()')
        else:
            raise Exception
        # model.fc = torch.nn.Linear(model.fc.in_features, out_features=num_classes)
    elif name in ['squeezenet1_0', 'squeezenet1_1']:
        torchvision.models.squeezenet1_0()
        model = eval(f'torchvision.models.{name}(pretrained=pretrained, num_classes=num_classes)')
    elif name =='alexnet':
        model = torchvision.models.alexnet(pretrained=pretrained, num_classes=num_classes)
        # model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif name in ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']:
        model = eval(f'torchvision.models.{name}(pretrained=pretrained, num_classes=num_classes)')
        # model = torchvision.models.vgg11(pretrained=pretrained, num_classes=num_classes)
    elif name == 'mynet1':
        model = MyNet1()

    model.name = name

    return model


if __name__ == "__main__":
    # 模型测试
    # model = get_model('mynet1')
    model = timm.create_model('efficientnet_b0', num_classes=2)    # timm.list_models()
    input = torch.randn(1, 3, 400, 400)
    o = model(input)
    print(o.shape)
