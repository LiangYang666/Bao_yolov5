#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool0_create_dir.py
# @Time      :2021/9/24 下午4:41
# @Author    :Yangliang
import os

from tool1_gennrate_yolov5label import brand, part
def check_and_build(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

if __name__ == "__main__":
    path = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}'
    if not os.path.exists(path):
        os.makedirs(path)
    paths = []
    for dir in ['augmented_imgs_select', 'imgxmls', 'sign_1_handle', 'sign_1_handle/1detect_select_ok', 'sign_2_handle', 'sign_2_handle/1detect_select_ok']:
        paths.append(os.path.join(path, dir))
    check_and_build(*paths)
