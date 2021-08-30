"""
1. Copy the necessary img files and xml files to /Detection/data
2. Change the name xml files and img files with the path

"""

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
from shutil import copyfile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 解决image file is truncated (XX bytes not processed)
l = ['Chanel', 'Prada', 'Burberry', 'Gucci', 'LV']
brand = 'LV'
src_path = f'../../../data/{brand}/OriginalData'
assert os.path.exists(src_path), f'wrong source path {src_path}'


def check_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Create ', path)

if __name__ == '__main__':
    xml_save_path = os.path.join(src_path, '../Detection/data/xmllabels')
    imgs_save_path = os.path.join(src_path, '../Detection/data/images')
    check_create_path(xml_save_path)
    check_create_path(imgs_save_path)
    img_suffixs = ['jpg', 'jpeg', 'png', 'tif']
    sub_root_dirs = os.listdir(src_path)
    ssub_root_dirs = []
    for sub_root_dir in sub_root_dirs:
        all = os.listdir(os.path.join(src_path, sub_root_dir))
        for i in all:
            ssub_root_dirs.append(os.path.join(src_path, sub_root_dir, i))


    pbar = tqdm(total=len(ssub_root_dirs), ncols=80)
    def copy_trans_onedir(ssub_root_dir):
        """
        Copy the img files and xml files in a given director to a new path
        Change the img and xml files' name
        """
        pbar.update(1)
        # print(ssub_root_dir)
        files = os.listdir(ssub_root_dir)
        root = ssub_root_dir
        if len(files) > 0:  # has the xml and imgs
            xml_files = [xml for xml in files if xml.endswith('.xml')]
            img_files = [img for img in files if img.split('.')[-1].lower() in img_suffixs]
            if len(xml_files) > 0:
                for xml_file in xml_files:
                    pre_name = xml_file.split('.')[0]
                    img_file = None
                    for i in img_files:
                        if i.split('.')[0] == pre_name:
                            img_file = i
                            break

                    if not img_file:
                        break
                    img_file = os.path.join(img_file)
                    xmlname = ssub_root_dir.split(src_path)[-1][1:].replace('/', '_') + '_' + xml_file
                    imgname = ssub_root_dir.split(src_path)[-1][1:].replace('/', '_') + '_' + img_file
                    copyfile(os.path.join(ssub_root_dir, img_file), os.path.join(imgs_save_path, imgname))
                    copyfile(os.path.join(ssub_root_dir, xml_file), os.path.join(xml_save_path, xmlname))
    executor = ThreadPoolExecutor(max_workers=10)
    for i in executor.map(copy_trans_onedir, ssub_root_dirs):
        pass
    pbar.close()


