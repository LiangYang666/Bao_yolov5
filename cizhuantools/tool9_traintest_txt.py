import os
import json
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import numpy as np
import math

from tool2_generate4points import data_dir, src_label_json_file
from tool6_by_fenge_func import bycut_imgs_save_dir
from tool7_zx_fenge_func import zxcut_imgs_save_dir
from tool8_generate_yolotxt import bycut_label_save_dir, zxcut_label_save_dir


def genarate_train_test(category='by', proportion=0.8):
    if category == 'by':
        cut_imgs_save_dir = bycut_imgs_save_dir
        cut_label_save_dir = bycut_label_save_dir
    elif category == 'zx':
        cut_imgs_save_dir = zxcut_imgs_save_dir
        cut_label_save_dir = zxcut_label_save_dir
    else:
        raise Exception()

    pre_label = cut_label_save_dir.split('CS1/')[1]
    pre_image = cut_imgs_save_dir.split('CS1/')[1]
    print(pre_label)
    print(pre_image)
    cut_txt_l = os.listdir(cut_label_save_dir)
    cut_img_l = os.listdir(cut_imgs_save_dir)
    # print(by_cut_img_l[:2])
    random.shuffle(cut_txt_l)
    # image_l = []
    # label_l = []
    txt_lines = []
    for cut_txt in tqdm(cut_txt_l):
        cut_img = cut_txt.split('.')[0] + '.jpg'
        if cut_img in cut_img_l:
            a = os.path.join(pre_label, cut_txt)
            b = os.path.join(pre_image, cut_img)
            txt_lines.append(a + ' ' + b + '\n')
    # print(pre)
    total = len(txt_lines)
    train_n = int(total * proportion)
    train_txt_lines = txt_lines[:train_n]
    test_txt_lines = txt_lines[train_n:]
    with open(data_dir + f'/all{category}.txt', 'w') as f:
        f.writelines(train_txt_lines)
    with open(data_dir + f'/train{category}.txt', 'w') as f:
        f.writelines(train_txt_lines)
    with open(data_dir + f'/test{category}.txt', 'w') as f:
        f.writelines(test_txt_lines)

if __name__ == "__main__":
    genarate_train_test(category='by', proportion=0.8)
