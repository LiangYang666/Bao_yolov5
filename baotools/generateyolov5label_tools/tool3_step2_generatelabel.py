import os
import json
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import numpy as np
import math
import xml.etree.ElementTree as ET

import PIL.Image
from concurrent.futures import ThreadPoolExecutor
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 解决image file is truncated (XX bytes not processed)
from tool2_step1_statistics import get_boxes_from_xml
from tool1_copydata_transname import check_create_path

all_statistics = {'Prada':
                      {'categories_name': ['button', 'embossing', 'front', 'label', 'logo', 'sign', 'tag', 'zipper'],
                       'categories_name_dic': {'button': 264, 'embossing': 320, 'front': 2795, 'label': 1185,
                                               'logo': 985, 'sign': 3131, 'tag': 1275, 'zipper': 1289}},
                  'Chanel':
                      {'categories_name': ['coding', 'front', 'hasp', 'sign', 'tag'],
                       'categories_name_dic': {'coding': 1741, 'front': 5285, 'hasp': 994, 'sign': 3821,
                                               'tag': 3701}},
                  'Burberry':
                      {'categories_name': ['embossing', 'front', 'sign', 'tag'],
                       'categories_name_dic': {'embossing': 901, 'front': 4004, 'sign': 4147, 'tag': 3750}},
                  'Gucci':
                      {'categories_name': ['bag', 'button', 'coding', 'front', 'sign', 'tag'],
                       'categories_name_dic': {'bag': 1744, 'button': 740, 'coding': 1875, 'front': 4504, 'sign': 3302,
                                               'tag': 2604}},
                  'LV':
                      {'categories_name': ['button', 'coding', 'front', 'lock', 'sign', 'zipper'],
                       'categories_name_dic': {'button': 9932, 'coding': 4459, 'front': 6829, 'lock': 6628,
                                               'sign': 5265, 'zipper': 7428}}
                  }
from tool2_step1_statistics import brand

# brand = 'Chanel'
categories_name = all_statistics[brand]['categories_name']
all_brand = list(all_statistics.keys())

if __name__ == '__main__':
    print('Question:')
    print(
        f'\tIs the brand \033[1;31m{brand}\033[0m ? if yes please type the Enter key to skip, otherwise type brand\'s name.')
    print(f'\tThe available brand are ' + ' \033[1;32m' + ', '.join(all_brand) + '\033[0m.')
    print('\t\033[5mInput:\033[0m', end='')
    get = input('')
    if get == '':
        print('The brand is ', brand)
        pass
    else:
        brand = get
        print(f'Your choice is {get}')

    imgs_path = f'../../../data/{brand}/Detection/data/images'
    xmls_path = f'../../../data/{brand}/Detection/data/xmllabels'
    labels_path = f'../../../data/{brand}/Detection/data/labels'
    Detection_data_path = f'../../../data/{brand}/Detection/data'
    all_yolov5_imgs_txts_path = os.path.join(Detection_data_path, 'all_yolov5_imgs_txts.txt')

    check_create_path(labels_path)

    img_names = os.listdir(imgs_path)
    xml_names = os.listdir(xmls_path)
    # img_names.sort()
    # xml_names.sort()
    img_xml_datas = []

    print(f"Brand is {brand}")
    print("1.Getting the imgs\' list and xmls\' list")


    # for xml_name in xml_names:

    def get_imgs_xmls_list(xml_name):
        pre_name = xml_name.split('.')[0]
        for img_name in img_names:
            if img_name.split('.')[0] == pre_name:
                return (xml_name, img_name)
        return None


    executor = ThreadPoolExecutor(max_workers=4)
    for i in executor.map(get_imgs_xmls_list, tqdm(xml_names)):
        if i:
            img_xml_datas.append(i)
    # img_xml_datas = [i for i in img_xml_datas if i]
    print(f'\tThere is {len(img_xml_datas)} images with label is available!')
    print("2.Generating the yolo labels to " + xmls_path)


    def generate_yolotxt_1by1_real(xml_name, img_name):
        try:
            img = PIL.Image.open(
                os.path.join(imgs_path,
                             img_name))  # the windows labelimg software does not use the exif imfo, and PIL either
        except Exception:
            print('The image was damaged ', os.path.join(xmls_path, xml_name))
            return None
        try:
            info = get_boxes_from_xml(os.path.join(xmls_path, xml_name))
        except Exception:
            print('error happened when read the', os.path.join(xmls_path, xml_name))
            return None
        if len(info) == 0:
            return None
        w, h = img.size
        yolo_labels = []
        for each in info:
            catogory, xmin, ymin, xmax, ymax = each
            assert catogory in categories_name
            x_trans = (xmin + xmax) / 2 / w
            y_trans = (ymin + ymax) / 2 / h
            w_trans = (xmax - xmin) / w
            h_trans = (ymax - ymin) / h
            yolo_label = [categories_name.index(catogory), x_trans, y_trans, w_trans, h_trans]
            yolo_labels.append(yolo_label)
        txt_name = img_name.split('.')[0] + '.txt'
        with open(os.path.join(labels_path, txt_name), 'w') as f:
            for x in yolo_labels:
                f.write(f'{x[0]} {x[1]} {x[2]} {x[3]} {x[4]}\n')
        txt_name = os.path.join(os.path.basename(labels_path), txt_name)
        img_name = os.path.join(os.path.basename(imgs_path), img_name)
        return txt_name, img_name


    def generate_yolotxt_1by1(z):
        return generate_yolotxt_1by1_real(z[0], z[1])


    executor = ThreadPoolExecutor(max_workers=10)
    all_imgs_txts = []
    for i in executor.map(generate_yolotxt_1by1, tqdm(img_xml_datas)):
        if i:
            all_imgs_txts.append(i)

    print(f'3.Writing all yolov5 images and txt files\' name to {all_yolov5_imgs_txts_path}.')
    all_imgs_txts.sort()
    with open(all_yolov5_imgs_txts_path, 'w') as f:
        for i in tqdm(all_imgs_txts):
            f.write(' '.join(i) + '\n')
