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
from tool3_step2_generatelabel import all_statistics, all_brand, brand
# all_brand = list(all_statistics.keys())
random.seed(1)


if __name__ == '__main__':
    print('Question:')
    print(f'\tIs the brand \033[1;31m{brand}\033[0m ? if yes please type the Enter key to skip, otherwise type brand\'s name.')
    print(f'\tThe available brand are '+' \033[1;32m'+', '.join(all_brand)+'\033[0m.')
    print('\t\033[5mInput:\033[0m', end='')
    get = input('')
    if get == '':
        print('The brand is ', brand)
        pass
    else:
        brand = get
        print(f'Your choice is {get}')

    categories_name = all_statistics[brand]['categories_name']
    categories_n = len(categories_name)

    Detection_data_path = f'../../../data/{brand}/Detection/data'
    labels_path = f'../../../data/{brand}/Detection/data/labels'
    # data_path = f'../../../data/{brand}/Detection/data'
    all_yolov5_imgs_txts_path = os.path.join(Detection_data_path, 'all_yolov5_imgs_txts.txt')
    train_txt = os.path.join(Detection_data_path, 'train.txt')
    test_txt = os.path.join(Detection_data_path, 'test.txt')

    with open(all_yolov5_imgs_txts_path, 'r') as f:
        all_yolov5_imgs_txts = f.readlines()

    random.shuffle(all_yolov5_imgs_txts)
    total = len(all_yolov5_imgs_txts)
    ratio = 0.7

    one_category_yolov5_imgs_txts_lines = []        # The lines point to the images which have only one category object
    one_more_categories_yolov5_imgs_txts_lines = [] # The lines point to the images which have one more category object

    imgs_txts_of_categories = {}
    for name in categories_name:
        imgs_txts_of_categories[name] = []
    # print(imgs_txts_of_categories)
    # print('\t********************')
    print(f'1.Analyze categories information to split the train and test data from each line of the file {all_yolov5_imgs_txts_path}.')
    for l in all_yolov5_imgs_txts:
        l = l.strip()
        with open(os.path.join(Detection_data_path, l.split()[0]), 'r') as f:
            infos = f.readlines()
        if len(infos) == 1:
            one_category_yolov5_imgs_txts_lines.append(l)
            category_index = int(infos[0].split()[0])
            # assert category_index < categories_n
            imgs_txts_of_categories[categories_name[category_index]].append(l)
        elif len(infos) > 1:
            one_more_categories_yolov5_imgs_txts_lines.append(l)

    train_yolov5_imgs_txts = []
    test_yolov5_imgs_txts = []

    # print('\t********************')
    print('2. Split the train and test data.')
    temp = [imgs_txts_of_categories[key] for key in imgs_txts_of_categories.keys()]+[one_more_categories_yolov5_imgs_txts_lines]
    for lines in temp:
        l = len(lines)
        t = int(l*ratio)
        random.shuffle(lines)
        train_yolov5_imgs_txts += lines[:t]
        test_yolov5_imgs_txts += lines[t:]


    # n = int(partial*total)
    # train_yolov5_imgs_txts = all_yolov5_imgs_txts[:n]
    # test_yolov5_imgs_txts = all_yolov5_imgs_txts[n:]
    #

    print('\tAll available files total', len(all_yolov5_imgs_txts))
    print('\tTrain files total', len(train_yolov5_imgs_txts))
    print('\tTest files total', len(test_yolov5_imgs_txts))

    print(f'3.Writing the txts to {train_txt} and {test_txt}.')
    with open(train_txt, 'w') as f:
        for s in train_yolov5_imgs_txts:
            f.write(s+'\n')
    with open(test_txt, 'w') as f:
        for s in test_yolov5_imgs_txts:
            f.write(s+'\n')
    print('\tDone!')
