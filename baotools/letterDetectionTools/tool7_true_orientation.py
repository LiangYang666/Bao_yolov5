#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool7_true_orientation.py
# @Time      :2021/9/5 下午3:30
# @Author    :Yangliang
import os

from PIL import Image, ImageOps
from tqdm import tqdm

# 实现对真类图片的翻转，按照exif信息翻转保存，保存的新图片不含exif信息，避免不同读取图片模块读入结果不同

def trans_img_2_new_dir(src_img_dir, dst_img_dir):
    img_suffixs = ['jpg', 'jpeg', 'png', 'tif']
    img_names = os.listdir(src_img_dir)
    img_names = sorted([x for x in img_names if x.rsplit('.', 1)[-1].lower() in img_suffixs])
    if not os.path.exists(dst_img_dir):
        os.mkdir(dst_img_dir)
    for img_name in tqdm(img_names):
        img_path = os.path.join(src_img_dir, img_name)
        img = Image.open(img_path)
        exif_data = img._getexif()
        dst_img_path = os.path.join(dst_img_dir, img_name)
        img = ImageOps.exif_transpose(img)
        img.save(dst_img_path)


if __name__ == "__main__":
    src_img_dir = '/media/D_4TB/YL_4TB/BaoDetection/data/Chanel/LetterDetection/data/sign/sign_1_need'
    dst_img_dir = '/media/D_4TB/YL_4TB/BaoDetection/data/Chanel/LetterDetection/data/sign/sign_1_need_旋转后图片'
    trans_img_2_new_dir(src_img_dir, dst_img_dir)
