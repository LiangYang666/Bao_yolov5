#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool3_add_aug_labels0.py
# @Time      :2021/8/31 下午4:22
# @Author    :Yangliang
import os

if __name__ == "__main__":
    brand = 'Chanel'
    part = 'sign'
    path = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}'
    augmented_imgs_dir = os.path.join(path, 'augmented_imgs_select')
    labels_path = os.path.join(path, 'labels')
    txt_path = os.path.join(path, 'aug_all.txt')

    files = os.listdir(augmented_imgs_dir)
    labels = os.listdir(labels_path)
    label_pre_names = [x.rsplit('.', 1)[0] for x in labels]
    with open(txt_path, 'w') as f:
        for file in files:
            label = file.rsplit('_', 1)[0]+'.txt'
            img_dir = os.path.basename(augmented_imgs_dir)
            strline = f'labels/{label} {img_dir}/{file}\n'
            f.write(strline)




