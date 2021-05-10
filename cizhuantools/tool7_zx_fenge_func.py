import os
import json
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import numpy as np
import math
c
from tool2_generate4points import img_4points_info_save_json_file, src_imgs_dir

zxcut_imgs_save_dir = src_imgs_dir + '_zxcutsave'
if not os.path.exists(zxcut_imgs_save_dir): os.makedirs(zxcut_imgs_save_dir)

def img_zx_cut(src, name, box, cut_n=9, save_dir=None):
    """
    Args:
        src:  图片array
        name: 图片名
        box:
        cut_n: 割快每一行列的数量
    Returns:
    """
    assert box.shape == (4, 2), 'should be numpy arr'
    src_width = src.shape[1]
    src_height = src.shape[0]
    # cut_single_by_list = []
    # box = bos2lefttop(box)
    x0_all = np.min(box[:, 0])-20
    y0_all = np.min(box[:, 1])-20
    x1_all = np.max(box[:, 0])+20
    y1_all = np.max(box[:, 1])+20
    x0_all, x1_all = [np.clip(x, 0, src_width-1) for x in [x0_all, x1_all]]
    y0_all, y1_all = [np.clip(y, 0, src_height-1) for y in [y0_all, y1_all]]

    cut_w_s = (x1_all-x0_all)/cut_n
    cut_h_s = (y1_all-y0_all)/cut_n
    jiange_x_half = int(cut_w_s/2)+4
    jiange_y_half = int(cut_h_s/2)+4

    for i in range(cut_n):
        for j in range(cut_n):
            x_s = int(round(x0_all+(j*cut_w_s)+cut_w_s/2))
            y_s = int(round(y0_all + (i * cut_h_s) + cut_h_s / 2))
            x0 = x_s - jiange_x_half
            x1 = x_s + jiange_x_half
            y0 = y_s - jiange_y_half
            y1 = y_s + jiange_y_half

            cut_image = src[y0:y1, x0:x1]
            h, w = cut_image.shape[:2]
            cut_single_dic = {'name': f'_zx_{w}_{h}_{i * cut_n + j}_{x0}_{y0}_{x1}_{y1}.'.join(name.split('.'))}
            cut_single_dic['image'] = cut_image
            cut_single_dic['xyxy'] = [x0, y0, x1, y1]
            # cut_single_by_list.append(cut_single_dic)
            save_name = os.path.join(save_dir, cut_single_dic['name'])
            # print(cut['image'])
            # print(save_name)
            cv2.imwrite(save_name, cut_single_dic['image'])

            # cv2.rectangle(src, (x0, y0), (x1, y1), (255, 0, 0), thickness=5)


if __name__ == "__main__":
    # print('Reading json.....')
    with open(img_4points_info_save_json_file, 'r') as f:
        image_4points_list = json.load(f)
    for l in tqdm(image_4points_list):
        name = l['name']
        p = os.path.join(src_imgs_dir, name)
        src_orgin = cv2.imread(p, cv2.IMREAD_COLOR)
        box = l['box']
        assert len(box) == 8
        box = np.array(box).reshape((4, 2))
        img_zx_cut(src_orgin, name, box, cut_n=9, save_dir=zxcut_imgs_save_dir)

        # cv2.namedWindow("image_get", cv2.WINDOW_NORMAL)
        # cv2.imshow('image_get', src_orgin)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # image_get = img_by_cut(src_orgin, name, box, bycut_imgs_save_dir)
