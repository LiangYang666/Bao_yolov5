#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool7_cut_true_letter.py
# @Time      :2021/9/4 下午3:28
# @Author    :Yangliang
import json
import os
import shutil
import warnings

import cv2

# 裁剪正确检测到的字母，将字母复制到指定文件夹
from tool1_gennrate_yolov5label import brand, part
# brand = 'Chanel'
# part = 'sign'

if __name__ == "__main__":

    categories_file = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/categories.txt'
    src_img_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/{part}_1_need_旋转后图片'  # 待裁剪的图片位置
    detected_selected_img_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/{part}_1_handle/1detect_select_ok' # 检测图片及检测结果json保存目录
    dst_p_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/{part}_1_handle/2裁剪后正确真类字母'     # 将要复制到目标文件夹
    if not os.path.exists(dst_p_dir):
        os.mkdir(dst_p_dir)

    with open(categories_file, 'r') as f:
        lines = f.readlines()
        categories = [line.strip() for line in lines if len(line.strip()) >= 1]
    for category in categories:
        dst_img_dir = os.path.join(dst_p_dir, category)
        if os.path.exists(dst_img_dir):
            shutil.rmtree(dst_img_dir)
        os.mkdir(dst_img_dir)
        category_id = categories.index(category)
        files = os.listdir(detected_selected_img_dir)
        json_files = [x for x in files if x.endswith('.json')]
        img_suffixs = ['jpg', 'jpeg', 'png', 'tif']
        img_names = [x for x in files if x.rsplit('.', 1)[-1].lower() in img_suffixs]
        if len(json_files) != 1:
            warnings.warn("json 文件不唯一")
        json_file_path = os.path.join(detected_selected_img_dir, json_files[0])
        with open(json_file_path, 'r') as f:
            detected_selected_info = json.load(f)   # 返回字典形如 detected_info[img_name][category_id]=[bbox1, bbox2...] 其中bbox为4元列表，存储x1,y1,x2,y2
        for img_name in detected_selected_info.keys():
            if img_name in img_names:
                img_path = os.path.join(src_img_dir, img_name)
                img = cv2.imread(img_path)
                if len(detected_selected_info[img_name][str(category_id)]) != 1:
                    warnings.warn("字母不唯一")
                x1, y1, x2, y2 = detected_selected_info[img_name][str(category_id)][0]
                img = img[y1:y2+1, x1:x2+1]
                img_save_path = os.path.join(dst_img_dir, img_name)
                cv2.imwrite(img_save_path, img)
