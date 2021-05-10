import os
import json
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import numpy as np
import math
from pathlib import Path

from tool2_generate4points import data_dir
from tool1_labelall import color_list, plot_one_box
from tool3_labelresp import class_names

all_json_file = os.path.join(data_dir, 'myresult/alldect7.json')
src_imgs_dir = os.path.join(data_dir, 'tile_round1_testA_20201231/testA_imgs')
img_4points_info_save_json_file = str(Path(src_imgs_dir).parent / 'ims_4points.json')
save_imgs_dir = os.path.join(data_dir,  'myresult/imgs_dect')
if not os.path.exists(save_imgs_dir): os.makedirs(save_imgs_dir)


if __name__ == '__main__':
    with open(all_json_file, 'r') as f:
        all_result_list = json.load(f)
    image_dict = {}  # 每张图片一个字典key 其value为字典 字典中为框的标签列表
    # 标签列表中含n个框 每个框为一个列表 列表6个值 类别及两点坐标和置信度
    for label in all_result_list[:]:
        name = label['name']
        # image_height = label['image_height']
        # image_width = label['image_width']
        l_bboxes = [label['category']] + label['bbox'] + [label['score']]
        if name not in image_dict.keys():
            image_dict[name] = {'l_bboxes': []}
        image_dict[name]['l_bboxes'].append(l_bboxes)
    image_4p_dic = {}
    with open(img_4points_info_save_json_file, 'r') as f:
        load_4p = json.load(f)
    for i in load_4p:
        name = i['name']
        _4p = i['box']
        image_4p_dic[name] = _4p

    colors = color_list()  # list of colors
    print(colors)
    bl = sorted(list(image_dict.keys()))[:50]
    for name in tqdm(bl):
        p = os.path.join(src_imgs_dir, name)
        src = cv2.imread(p)
        for l_bbox in image_dict[name]['l_bboxes']:
            bbox = l_bbox[1:5]
            category = l_bbox[0]
            score = l_bbox[5]
            color = colors[category % len(colors)]
            plot_one_box(bbox, src, label=f'{class_names[category-1]} {score}', color=color, line_thickness=3)
        contours = np.array(image_4p_dic[name]).reshape(1, 4, 1, 2)
        cv2.drawContours(src, contours, -1, (0, 0, 255), 2)
        save_name = os.path.join(save_imgs_dir,'b'+name)
        cv2.imwrite(save_name, src)

        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 1200, 1200)
        cv2.imshow(name, src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()