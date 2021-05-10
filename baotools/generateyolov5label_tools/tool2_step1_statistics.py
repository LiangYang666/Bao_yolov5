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

"""
    Make a statistics pon given data
    In this step, you need to watch the run outputs to determine if the data needs to be modified
"""

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 解决image file is truncated (XX bytes not processed)

l = ['Chanel', 'Prada', 'Burberry', 'Gucci', 'LV']
brand = 'LV'
src_path = f'../../../data/{brand}/OriginalData'
xml_save_path = os.path.join(src_path, '../Detection/data/xmllabels')
assert os.path.exists(src_path), 'wrong source path'
# need_del = []



def get_boxes_from_xml(file):
    tree = ET.parse(file)  # 获取xml文件
    root = tree.getroot()
    filename = root.find('filename').text
    # object = root.find('object')
    info = []
    for object in root.findall('object'):
        name = object.find('name').text
        if name == 'loc':
            print('Change to lock', file)
            name = 'lock'
        bandbox = object.find('bndbox')
        xmin = int(bandbox.find('xmin').text)
        ymin = int(bandbox.find('ymin').text)
        xmax = int(bandbox.find('xmax').text)
        ymax = int(bandbox.find('ymax').text)
        each = [name, xmin, ymin, xmax, ymax]
        info.append(each)
    return info


if __name__ == "__main__":
    save_path = os.path.join(src_path, '../Detection/data')
    # statistic_path = os.path.join(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('Create ', save_path)
    categories_name = []
    categories_name_dic = {}

    #
    print('Step 1 : make a statistics of all category')
    xml_files = os.listdir(xml_save_path)
    for xml in xml_files:
        # print(os.path.join(root, xml))
        try:
            info = get_boxes_from_xml(os.path.join(xml_save_path, xml))
        except Exception:
            print('error happened when read the', os.path.join(xml_save_path, xml))
            continue
        for each in info:
            name = each[0]
            if name not in categories_name_dic.keys():
                categories_name_dic[name] = 0
            categories_name_dic[name] += 1
    categories_name = list(categories_name_dic.keys())
    categories_name.sort()
    categories_name_dic = dict(sorted(categories_name_dic.items(), key=lambda x: x[0]))
    # print('#', src_path.split('/')[-2])
    # print('categories_name =', categories_name)
    # print('categories_name_dic =', categories_name_dic)
    statistics = {src_path.split('/')[-2]: {
                  'categories_name': categories_name,
                  'categories_name_dic': categories_name_dic}}
    print(statistics)
    # print('{')
    # for key in statistics.keys():
    #     print(f'\t\"{key}\"', ':', statistics[key])
    # print('}')
    with open(save_path + '/class_statistics.json', 'w') as f:
        json.dump(statistics, f, indent=2)
    print(json.dumps(statistics, indent=2))
