#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :model.py
# @Time      :2021/9/7 下午12:43
# @Author    :Yangliang
import torch
import torchvision


def get_model(name='resnet34'):
    # 模型
    if name.startswith('resnet'):
        if name == 'resnet18':
            model = torchvision.models.resnet18()
        elif name == 'resnet34':
            model = torchvision.models.resnet34()
        elif name == 'resnet50':
            model = torchvision.models.resnet50()
        elif name == 'resnet101':
            model = torchvision.models.resnet101()
        else:
            raise Exception
        model.fc = torch.nn.Linear(model.fc.in_features, out_features=2)
    model.name = name
    return model

if __name__ == "__main__":
    # 模型测试
    model = get_model()
    input = torch.randn(1, 3, 224, 224)
    o = model(input)
    print(o.shape)
