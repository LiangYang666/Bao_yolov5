import os
import json
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import numpy as np
import math

from tool2_generate4points import src_label_json_file

from tool6_by_fenge_func import bycut_imgs_save_dir
from tool7_zx_fenge_func import zxcut_imgs_save_dir

bycut_label_save_dir = bycut_imgs_save_dir + '_label0'
zxcut_label_save_dir = zxcut_imgs_save_dir + '_label0'


def generate_label(category='by'):  # category 为 by或者zx
    if category == 'by':
        cut_imgs_save_dir = bycut_imgs_save_dir
        cut_label_save_dir = bycut_label_save_dir
    elif category == 'zx':
        cut_imgs_save_dir = zxcut_imgs_save_dir
        cut_label_save_dir = zxcut_label_save_dir
    else:
        raise Exception()
    if not os.path.exists(cut_label_save_dir): os.makedirs(cut_label_save_dir)
    cut_imgs_list = os.listdir(cut_imgs_save_dir)
    print(len(cut_imgs_list))
    cut_imgs_list = sorted(cut_imgs_list)
    # print(cut_imgs_list[:200])
    imgs_cut_dic = {}  # 用一个字典保存 每张原图名称对应的切割图名称列表
    for cut_img in cut_imgs_list:
        src_name = cut_img.split(f'_{category}_')[0] + '.' + cut_img.split('.')[-1]
        if src_name not in imgs_cut_dic.keys():
            imgs_cut_dic[src_name] = []
        imgs_cut_dic[src_name].append(cut_img)

    print(len(imgs_cut_dic))

    # bycut_label_save_file = bycut_imgs_save_dir + '_label.txt'
    # f = open(bycut_label_save_file, 'w')

    with open(src_label_json_file, 'r') as load_f:
        load_list = json.load(load_f)
    image_dict = {}  # 每张图片一个字典key 其value为字典 字典中为高宽及框的标签列表  标签列表中含n个框 每个框为一个列表 列表5个值 类别及两点坐标
    for label in load_list[:]:
        name = label['name']
        image_height = label['image_height']
        image_width = label['image_width']
        l_bboxes = [label['category']] + label['bbox']
        if name not in image_dict.keys():
            image_dict[name] = {'image_height': image_height, 'image_width': image_width, 'l_bboxes': []}
        image_dict[name]['l_bboxes'].append(l_bboxes)

    if category == 'by':
        c_want = [1, 2]
    elif category == 'zx':
        c_want = [3, 4, 5, 6]
    else:
        raise Exception()
    for name in tqdm(list(image_dict.keys())):
        temp_cut_label_dic = {}
        # # c = label['category']
        # c = image_dict[name]
        for l_bbox in image_dict[name]['l_bboxes']:
            c = l_bbox[0]
            assert c in range(1, 6 + 1)
            if c in c_want:
                bbox = l_bbox[1:]
                bbox[:2] = [int(x) for x in bbox[:2]]
                bbox[2:] = [int(np.ceil(x)) for x in bbox[2:]]
                for cut_name in imgs_cut_dic[name]:
                    cut_back = cut_name.split(f'_{category}_')[1].split('.')[0]
                    bj = cut_back.split('_')
                    assert len(bj) == 7, f'length {len(bj)}'
                    bj = [int(x) for x in bj]  # 标记信息

                    (cutw, cuth, _), boxcut = bj[:3], bj[3:]
                    assert len(boxcut) == 4
                    # print(len(boxcut))

                    box_inter = [max(boxcut[0], bbox[0]), max(boxcut[1], bbox[1]), min(boxcut[2], bbox[2]),
                                 min(boxcut[3], bbox[3])]
                    iw = box_inter[2] - box_inter[0] + 1
                    ih = box_inter[3] - box_inter[1] + 1
                    if iw > 0 and ih > 0:
                        inter_are = iw * ih  # 相交的面积
                        bbox_w = bbox[2] - bbox[0] + 1
                        bbox_h = bbox[3] - bbox[1] + 1
                        bbox_area = bbox_w * bbox_h  # 目标框的面积
                        if (inter_are / bbox_area) > 0.7:
                            # x y _trans 为转换过来的坐标 坐标原点为剪切图片的左上角 （boxcut bbox boxcut 的坐标原点均为原始未切割图的左上角）
                            x0_trans = box_inter[0] - boxcut[0]
                            y0_trans = box_inter[1] - boxcut[1]
                            x1_trans = box_inter[2] - boxcut[0]
                            y1_trans = box_inter[3] - boxcut[1]

                            # 按照yolo标准保存标签 及每张图一个txt 每行一个目标 格式为 类别序号 目标的 x y w h
                            x_trans = (x0_trans + x1_trans) / 2 / cutw
                            y_trans = (y0_trans + y1_trans) / 2 / cuth
                            w_trans = (x1_trans - x0_trans) / cutw
                            h_trans = (y1_trans - y0_trans) / cuth
                            yolo_label = [c_want.index(c), x_trans, y_trans, w_trans, h_trans]
                            if cut_name not in temp_cut_label_dic.keys():
                                temp_cut_label_dic[cut_name] = []
                            temp_cut_label_dic[cut_name].append(yolo_label)
        for cut_name in temp_cut_label_dic.keys():
            txt_file = os.path.join(cut_label_save_dir, cut_name.split('.')[0] + '.txt')
            with open(txt_file, 'w') as f:
                for l in temp_cut_label_dic[cut_name]:
                    l = [str(x) for x in l]
                    line = ' '.join(l)
                    f.write(line + '\n')


category = 'zx'
if __name__ == "__main__":
    if category == 'zx':
        generate_label(category='zx')
    elif category == 'by':
        generate_label(category='by')

# '197_100_t20201119093903298_CAM1_by_0_965_325_1515_875.jpg'
#     {
#         "name": "223_89_t20201125085855802_CAM3.jpg",
#         "image_height": 3500,
#         "image_width": 4096,
#         "category": 4,
#         "bbox": [
#             1702.79,
#             2826.53,
#             1730.79,
#             2844.53
#         ]
#     },
