#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :augmentation.py
# @Time      :2021/9/29 下午9:35
# @Author    :Yangliang

import os
import shutil

import cv2
import numpy as np
from cv2 import dnn_superres


# RGB图像全局直方图均衡化
from tqdm import tqdm


def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    ycrcb = cv2.merge(channels)
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img

# 自适应直方图均衡化
def adaptiveHisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])

    ycrcb = cv2.merge(channels)
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img


brand = "LV"
part = "sign"
letter = "O"

def augment_imgs(src_dir, dst_dir):

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    # 创建sr对象
    sr = dnn_superres.DnnSuperResImpl_create()
    # 读取模型
    sr.readModel("EDSR_x4.pb")
    # 设定算法 和放大比例
    sr.setModel("edsr", 4)
    for sub_dir in os.listdir(src_dir):
        src_sub_dir_path = os.path.join(src_dir, sub_dir)
        dst_sub_dir_path = os.path.join(dst_dir, sub_dir)
        if os.path.exists(os.path.join(dst_dir, sub_dir)):
            shutil.rmtree(dst_sub_dir_path)
        os.mkdir(dst_sub_dir_path)
        for img_file in tqdm(sorted(os.listdir(src_sub_dir_path))):
            img_orgin = cv2.imread(os.path.join(src_sub_dir_path, img_file))
            # img = cv2.resize(img_orgin, dsize=None, fx=16, fy=16)
            img1 = hisEqulColor(img_orgin)
            img2 = adaptiveHisEqulColor(img_orgin)
            img3 = cv2.normalize(img_orgin, dst=None, alpha=350, beta=10, norm_type=cv2.NORM_MINMAX)
            upScalePic = sr.upsample(img3)
            # upScalePic = sr.upsample(upScalePic)
            res = np.hstack((img_orgin, img1, img2, img3))
            res = cv2.resize(res, dsize=None, fx=4, fy=4)
            res = np.hstack((res, upScalePic))
            # cv2.imshow(res)
            cv2.imwrite(os.path.join(dst_sub_dir_path, img_file), res)

if __name__ == '__main__':
    data_dir = f"/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/classification_data/{letter}"
    augment_imgs(os.path.join(data_dir, "train"), os.path.join(data_dir, "train_aug"))
    augment_imgs(os.path.join(data_dir, "test"), os.path.join(data_dir, "test_aug"))