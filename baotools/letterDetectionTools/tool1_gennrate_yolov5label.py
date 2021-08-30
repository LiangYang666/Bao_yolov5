#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool1_gennrate_yolov5label.py
# @Time      :2021/8/29 下午5:38
# @Author    :Yangliang
import os
import xml.etree.ElementTree as ET

def get_boxes_from_xml(file):
    tree = ET.parse(file)  # 获取xml文件
    root = tree.getroot()
    filename = root.find('filename').text
    # object = root.find('object')
    info = []
    for object in root.findall('object'):
        name = object.find('name').text
        if name == 'loc':
            print('Change to lock', file)
            name = 'lock'
        bandbox = object.find('bndbox')
        xmin = int(bandbox.find('xmin').text)
        ymin = int(bandbox.find('ymin').text)
        xmax = int(bandbox.find('xmax').text)
        ymax = int(bandbox.find('ymax').text)
        each = [name, xmin, ymin, xmax, ymax]
        info.append(each)
    return info


if __name__ == "__main__":
    brand = 'Chanel'
    part = 'sign'
    path = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}'
    labels_path = os.path.join(path, 'labels')
    imgxmls_path = os.path.join(path, 'imgxmls')
    all_txts_path = os.path.join(path, 'all.txt')


