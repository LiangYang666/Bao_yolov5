import os
import json
import matplotlib.pyplot as plt
import cv2
import random
import logging
import sys
import time
from tqdm import tqdm

data_dir = '../../data/CS1'
src_label_json = os.path.join(data_dir, 'tile_round1_train_20201231/train_annos.json')
src_imgs_dir = os.path.join(data_dir, 'tile_round1_train_20201231/train_imgs')
mini_save_dir = src_imgs_dir + '_minisave'
cut_imgs_save_dir = src_imgs_dir + '_zxcutsave'

label_dir = cut_imgs_save_dir + '_label0'
img_dir = cut_imgs_save_dir

label_file_l = os.listdir(label_dir)
for file in label_file_l[100:]:
    label_file = os.path.join(label_dir, file)
    img_file = os.path.join(img_dir, file.split('.')[0] + '.jpg')
    with open(label_file, 'r') as f:
        l_l = f.readlines()
        l_l = [x.strip() for x in l_l]
    img = cv2.imread(img_file)
    shape = img.shape
    for l in l_l:
        c = int(l.split()[0])
        x, y, w, h = [float(x) for x in l.split()][1:]
        x0 = x - w / 2
        y0 = y - h / 2
        x1 = x + w / 2
        y1 = y + h / 2
        x0 *= shape[1]
        x1 *= shape[1]
        y0 *= shape[0]
        y1 *= shape[0]
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)

        cv2.putText(img, f'{c}', (int(x0), int(y0)), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(0, 255, 255), thickness=2)
    cv2.namedWindow(file, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(file, 800, 800)
    cv2.imshow(file, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
