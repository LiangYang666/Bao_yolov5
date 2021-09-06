#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool8_copy_true_detectOk.py
# @Time      :2021/9/6 上午8:58
# @Author    :Yangliang

# 复制基本正常检测到的图片 例如复制所有字母类别检测都检测到，字母数量核对的上
import json
import os
import shutil

from tool5_get_intersect_letter_imgxml import get_json_info, get_category_names


def copy_detect_ok(src_img_dir, dst_img_dir, categories_file):
    if os.path.exists(dst_img_dir):
        shutil.rmtree(dst_img_dir)
    os.mkdir(dst_img_dir)
    categories = get_category_names(categories_file)
    info = get_json_info(src_img_dir)
    detect_ok_list = []
    for img_name in list(info.keys()):
        if len(categories) == len(info[img_name].keys()):
            detect_ok_list.append(img_name)
        else:
            del info[img_name]
    for img_name in detect_ok_list:
        src = os.path.join(src_img_dir, img_name)
        dst = os.path.join(dst_img_dir, img_name)
        shutil.copy(src, dst)
    with open(os.path.join(dst_img_dir, f'selected_imgs.json'), 'w') as f:
        json.dump(info, f, indent=2)

if __name__ == "__main__":
    brand = 'Chanel'
    part = 'sign'

    src_img_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/yolov5_rundata/dect/exp_ckp_aug_all_6c_epoch1760_2'
    dst_img_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/{part}_1_handle/1detect_select_ok/'
    categories_file = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/categories.txt'

    copy_detect_ok(src_img_dir, dst_img_dir, categories_file)



