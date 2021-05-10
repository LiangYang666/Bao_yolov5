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

bycut_label_save_dir = bycut_imgs_save_dir + '_label0'
if not os.path.exists(bycut_label_save_dir): os.makedirs(bycut_label_save_dir)

if __name__ == "__main__":
    bycut_imgs_list = os.listdir(bycut_imgs_save_dir)
    print(len(bycut_imgs_list))
    bycut_imgs_list = sorted(bycut_imgs_list)
    # print(bycut_imgs_list[:200])
    imgs_bycut_dic = {}  # 用一个字典保存 每张原图名称对应的切割图名称列表
    for bycut_img in bycut_imgs_list:
        src_name = bycut_img.split('_by_')[0] + '.' + bycut_img.split('.')[-1]
        if src_name not in imgs_bycut_dic.keys():
            imgs_bycut_dic[src_name] = []
        imgs_bycut_dic[src_name].append(bycut_img)

    print(len(imgs_bycut_dic))

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

    for name in tqdm(list(image_dict.keys())):
        temp_cut_label_dic = {}
        # # c = label['category']
        # c = image_dict[name]
        for l_bbox in image_dict[name]['l_bboxes']:
            c = l_bbox[0]
            if c == 1 or c == 2:
                bbox = l_bbox[1:]
                bbox[:2] = [int(x) for x in bbox[:2]]
                bbox[2:] = [int(np.ceil(x)) for x in bbox[2:]]
                for bycut_name in imgs_bycut_dic[name]:
                    cut_back = bycut_name.split('_by_')[1].split('.')[0]
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
                            yolo_label = [c - 1, x_trans, y_trans, w_trans, h_trans]
                            if bycut_name not in temp_cut_label_dic.keys():
                                temp_cut_label_dic[bycut_name] = []
                            temp_cut_label_dic[bycut_name].append(yolo_label)
        for bycut_name in temp_cut_label_dic.keys():
            txt_file = os.path.join(bycut_label_save_dir, bycut_name.split('.')[0] + '.txt')
            with open(txt_file, 'w') as f:
                for l in temp_cut_label_dic[bycut_name]:
                    l = [str(x) for x in l]
                    line = ' '.join(l)
                    f.write(line + '\n')

                # bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                # iw = bi[2] - bi[0] + 1
                # ih = bi[3] - bi[1] + 1
                # if iw > 0 and ih > 0:
                #     # compute overlap (IoU) = area of intersection / area of union
                #     ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                #                                                       + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih

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
