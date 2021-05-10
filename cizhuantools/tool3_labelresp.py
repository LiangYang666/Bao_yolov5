import os
import json
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import shutil

data_dir = '../data/CS1'
src_imdir = os.path.join(data_dir, 'labeld_trainimg')
json_file = os.path.join(data_dir, 'tile_round1_train_20201231/train_annos.json')

class_names = ['bian_yc', 'jiao_yc', 'baisedian_xc', 'qiansekuai_xc', 'shensekuai_xc', 'guangquan_xc']
if __name__ == '__main__':
    with open(json_file, 'r') as load_f:
      load_list = json.load(load_f)

    image_dict = {}    # 每张图片一个字典key 其value为字典 字典中为高宽及框的标签列表  标签列表中含n个框 每个框为一个列表 列表5个值 类别及两点坐标
    for label in load_list[:]:
        name = label['name']
        image_height = label['image_height']
        image_width = label['image_width']
        l_bboxes = [label['category']]+label['bbox']
        if name not in image_dict.keys():
            image_dict[name] = {'image_height': image_height, 'image_width': image_width, 'l_bboxes': []}
        image_dict[name]['l_bboxes'].append(l_bboxes)

    # %%
    print(type(src_imdir))
    print(src_imdir)

    # %%
    for i, class_name in enumerate(class_names):
        p = os.path.join(data_dir, 'labeled_resp', str(i+1)+'_'+class_name)
        if not os.path.exists(p):
            os.makedirs(p)

    for name in tqdm(image_dict.keys()):
        has_copy = dict(zip(class_names, [False]*len(class_names)))
        for l_bbox in image_dict[name]['l_bboxes']:
            cls = int(l_bbox[0])
            class_name = class_names[cls-1]
            if not has_copy[class_name]:
                has_copy[class_name] = True
                src = os.path.join(src_imdir, name)
                dst = os.path.join(data_dir, 'labeled_resp', str(cls)+'_'+class_name, name)
                shutil.copyfile(src, dst)


