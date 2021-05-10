import os
import json
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import numpy as np
import math

from tool2_generate4points import img_4points_info_save_json_file, src_imgs_dir

bycut_imgs_save_dir = src_imgs_dir + '_bycutsave'
if not os.path.exists(bycut_imgs_save_dir): os.makedirs(bycut_imgs_save_dir)


def get_line_func(A, B):
    assert len(A) == len(B) == 2, 'A and B is not point'
    x1, y1 = A
    x2, y2 = B
    dy = y2 - y1
    dx = x2 - x1
    # if dx==0:
    print(f'dx:{dx}, dy:{dy}')
    if abs(dx) > 30 and abs(dy) > 30:
        k = dy / dx
        b = y1 - k * x1
    else:
        k = 0
        b = 0
    return k, b


def wh_boundxy(w, h, x, y):
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    return x, y


def cut_single_by(src, name, i, box, cut_n=10):
    """
    每次切割一张图
    Args
        src: 图片array
        name:图片名
        i:
        box:
        cut_n:  每一条边切割快的数量
    Returns:
    """
    src_width = src.shape[1]
    src_height = src.shape[0]
    cut_single_by_list = []
    assert i < 4
    if i == 0:
        A = list(box[0])
        B = list(box[1])
    elif i == 2:
        A = list(box[3])
        B = list(box[2])
    elif i == 1:
        A = list(box[1])
        B = list(box[2])
    elif i == 3:
        A = list(box[0])
        B = list(box[3])

    if i == 0 or i == 2:
        k, b = get_line_func(A, B)
        if abs(k) < 0.005:  k = 0
        x_dist = B[0] - A[0]
        assert x_dist > 0
        # print('x_dist:', x_dist)
        jiange = x_dist / cut_n
        jiange_half = int(np.ceil(jiange / 2))
        jiange_half += 10
        print('cut_width:', jiange_half * 2)
        # print('jiange:', jiange)
        central = A.copy()
        if k == 0 and b == 0:
            central[1] = int((A[1] + B[1]) / 2)
        if i == 2:
            central[0] += jiange
            central[0] = int(central[0])
            if not (k == 0 and b == 0):
                central[1] = int(k * central[0] + b)

        for j in range(cut_n):
            y0 = central[1] - jiange_half
            x0 = central[0] - jiange_half
            y1 = central[1] + jiange_half
            x1 = central[0] + jiange_half
            x0, y0 = wh_boundxy(src_width, src_height, x0, y0)
            x1, y1 = wh_boundxy(src_width, src_height, x1, y1)

            cut_image = src[y0:y1, x0:x1]
            h, w = cut_image.shape[:2]
            cut_single_dic = {'name': f'_by_{w}_{h}_{i * cut_n + j}_{x0}_{y0}_{x1}_{y1}.'.join(name.split('.'))}
            cut_single_dic['image'] = cut_image
            cut_single_dic['xyxy'] = [x0, y0, x1, y1]
            cut_single_by_list.append(cut_single_dic)
            # cv2.rectangle(src, (x0, y0), (x1, y1), (255, 0, 0), thickness=5)
            central[0] += jiange
            central[0] = int(central[0])
            if not (k == 0 and b == 0):
                central[1] = int(k * central[0] + b)
    elif i == 1 or i == 3:
        k, b = get_line_func(A, B)
        y_dist = B[1] - A[1]
        assert y_dist > 0
        # print('y_dist:', y_dist)
        jiange = y_dist / cut_n
        jiange_half = int(np.ceil(jiange / 2))
        jiange_half += 10
        print('cut_width:', jiange_half * 2)
        # print('jiange:', jiange)
        central = A.copy()
        if k == 0 and b == 0:
            central[0] = int((A[0] + B[0]) / 2)
        if i == 3:
            central[1] += jiange
            central[1] = int(central[1])
            if not (k == 0 and b == 0):
                central[0] = int((central[1] - b) / k)
        for j in range(cut_n):
            y0 = central[1] - jiange_half
            x0 = central[0] - jiange_half
            y1 = central[1] + jiange_half
            x1 = central[0] + jiange_half
            x0, y0 = wh_boundxy(src_width, src_height, x0, y0)
            x1, y1 = wh_boundxy(src_width, src_height, x1, y1)
            cut_image = src[y0:y1, x0:x1]
            h, w = cut_image.shape[:2]
            cut_single_dic = {'name': f'_by_{w}_{h}_{i * cut_n + j}_{x0}_{y0}_{x1}_{y1}.'.join(name.split('.'))}
            cut_single_dic['image'] = cut_image
            cut_single_dic['xyxy'] = [x0, y0, x1, y1]
            cut_single_by_list.append(cut_single_dic)
            # cv2.rectangle(src, (x0, y0), (x1, y1), (255, 0, 0), thickness=5)
            central[1] += jiange
            central[1] = int(central[1])
            if not (k == 0 and b == 0):
                central[0] = int((central[1] - b) / k)
        # print(k)

    # print(f'k:{k}, b:{b}')
    return cut_single_by_list


def img_by_cut(src, name, box, save_dir):
    assert box.shape == (4, 2), 'should be numpy arr'
    # box = bos2lefttop(box)
    for i in range(4):
        cut_single_by_list = cut_single_by(src, name, i, box, cut_n=10)
        for cut in cut_single_by_list:
            save_name = os.path.join(save_dir, cut['name'])
            # print(cut['image'])
            # print(save_name)
            cv2.imwrite(save_name, cut['image'])


    return src


if __name__ == "__main__":
    with open(img_4points_info_save_json_file, 'r') as f:
        image_4points_list = json.load(f)
    for l in tqdm(image_4points_list):
        name = l['name']
        p = os.path.join(src_imgs_dir, name)
        src_orgin = cv2.imread(p, cv2.IMREAD_COLOR)
        box = l['box']
        assert len(box) == 8
        box = np.array(box).reshape((4, 2))
        image_get = img_by_cut(src_orgin, name, box, bycut_imgs_save_dir)
