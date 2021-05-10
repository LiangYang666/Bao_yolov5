import os
import json
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import numpy as np
from pathlib import Path

jilu = False
# def MyThreshold(th):
#     binary = cv2.threshold(img_gray, th, 255, cv2.THRESH_BINARY)[1]
#     cv2.imshow('image', binary)

data_dir = '../../data/CS1'
# src_label_imdir = os.path.join(data_dir, 'labeld_trainimg')

# src_imgs_dir = os.path.join(data_dir, 'tile_round1_train_20201231/train_imgs')
src_imgs_dir = os.path.join(data_dir, 'tile_round1_testA_20201231/testA_imgs')
img_4points_info_save_json_file = str(Path(src_imgs_dir).parent / 'ims_4points.json')
# img_4points_info_save_json_file = os.path.join(data_dir, 'tile_round1_train_20201231/train_4points.json')
src_label_json_file = os.path.join(data_dir, 'tile_round1_train_20201231/train_annos.json')
mini_save_dir = src_imgs_dir + '_minisave'
if not os.path.exists(mini_save_dir): os.makedirs(mini_save_dir)


def bos2lefttop(box):
    '''将box改为左上角开始顺时针旋转'''
    box_return = np.zeros(box.shape, dtype=np.int)
    # index_l = np.zeros(4, dtype=np.int)
    sum = box.sum(axis=1)
    max_index = np.where(sum == sum.max())[0]  # 取出最大的对应索引arr
    assert max_index.shape[0] == 1, 'should be only one max'
    max_index = max_index[0]
    min_index = np.where(sum == sum.min())[0]
    assert min_index.shape[0] == 1, 'should be only one max'
    min_index = min_index[0]

    box_return[0] = box[min_index]
    box_return[2] = box[max_index]
    box = np.delete(box, [min_index, max_index], axis=0)
    right_index = np.where(box[:, 0] == np.max(box, axis=0)[0])[0]
    assert right_index.shape[0] == 1, 'should be only one right'
    right_index = right_index[0]
    box_return[1] = box[right_index]
    box_return[3] = box[1 - right_index]
    return box_return

def image_preprocess(image_file, display=False):
    src_orgin = cv2.imread(image_file, cv2.IMREAD_COLOR)
    width = src_orgin.shape[1]//10
    height = src_orgin.shape[0]//10
    src = cv2.resize(src_orgin, (width, height))
    # print('resize.shape: ', src.shape)
    img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    k_s = width//120
    line_width = width//1200
    if k_s<3: k_s=3
    if line_width < 2: line_width=2

    # MORPH_CROSS MORPH_RECT
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_s, k_s))

    eroded = cv2.erode(img_gray, kernel, iterations=1)  # 腐蚀图像
    dilated = cv2.dilate(eroded, kernel, iterations=1)  # 膨胀图像
    # eroded = cv2.erode(dilated, kernel)    # 腐蚀图像
    # dilated = cv2.dilate(eroded, kernel)    # 膨胀图像
    k_s1 = k_s if k_s%2==1 else k_s+1
    blur = cv2.GaussianBlur(dilated, (k_s1, k_s1), 0)
    # blur = cv2.blur(dilated, (13, 13))
    # blur = cv2.medianBlur(binary, 5)  # 中值滤波
    binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # OTSU而之花
    # cannyPic = cv2.Canny(binary, 20, 100)  # 寻找边缘
    # 找出轮廓
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print(contours)
    maxArea = 0
    # 挨个检查看那个轮廓面积最大
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > cv2.contourArea(contours[maxArea]):
            maxArea = i
        # print(contours[i].shape, cv2.contourArea(contours[i]))
    imgarea = src.shape[0] * src.shape[0]
    max_area = cv2.contourArea(contours[maxArea])
    percentage = round(max_area / imgarea, 2)
    if percentage < 0.5:
        print(f'{name}')
        if jilu:
            f.write(name+'\n')
        print(f'Image\'s area is {imgarea} Max area:{max_area} percention:{percentage}')
    # print(f'Image\'s area is {imgarea} Max area:{max_area} percention:{percentage}')
    # assert percentage>0.5, f'{name}'
    cv2.drawContours(src, contours, -1, (0, 0, 255), line_width)
    # hull = cv2.convexHull(contours[maxArea])
    # cv2.drawContours(src, hull.reshape((1,64,1,2)), -1, (0, 255, 0), 3)
    # print(hull)
    # cv2.drawContours(src, contours, maxArea, (0, 0, 255), 4)
    # cv2.drawContours(src, [hull], -1, (0, 255, 0), 4)
    # rect = cv2.minAreaRect(hull)    # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    rect = cv2.minAreaRect(contours[maxArea])  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    box = cv2.boxPoints(rect)  # 获取四个顶点坐标
    box = np.int32(box)
    box = bos2lefttop(box)
    box_contour = box[:, np.newaxis, :]
    cv2.drawContours(src, [box_contour], -1, (255, 0, 0), line_width)
    # print(contours[maxArea])

    # lowThreshold = 0
    # max_lowThreshold = 100
    # cv2.createTrackbar('threshold', 'image', lowThreshold, max_lowThreshold, MyThreshold)
    if display:
        cv2.namedWindow("src", cv2.WINDOW_NORMAL)
        cv2.imshow('src', src)
        # cv2.namedWindow("img_gray", cv2.WINDOW_NORMAL)
        # cv2.imshow('img_gray', img_gray)
        # cv2.namedWindow("eroded", cv2.WINDOW_NORMAL)
        # cv2.imshow('eroded', eroded)
        # cv2.namedWindow("dilated", cv2.WINDOW_NORMAL)
        # cv2.imshow('dilated', dilated)
        # cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
        # cv2.imshow('binary', binary)

        # cv2.namedWindow("cannyPic", cv2.WINDOW_NORMAL)
        # cv2.imshow('cannyPic', cannyPic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return box, percentage, src, src_orgin



if __name__ == "__main__":
    ims_l = os.listdir(src_imgs_dir)
    # ims_l = ['253_194_t20201130134118539_CAM1.jpg', '241_31_t20201128123645884_CAM2.jpg',
    #          '253_192_t20201130134041685_CAM1.jpg', '220_51_t20201124133018331_CAM2.jpg',
    #          '253_234_t20201130144207493_CAM1.jpg']
    # with open('jilu.txt', 'r') as f:
    #     get = f.readlines()
    #     get = [x.strip() for x in get]
    # ims_l = get
    image_4points_list = []
    if jilu:
        f = open('jilu.txt', 'a')
        f.write('*****************************************************************************\n')
        f.write(f'test dir {src_imgs_dir}\n')
    save_mini_draw = False
    for name in tqdm(ims_l):
        image_file = os.path.join(src_imgs_dir, name)
        box, percentage, drawsrc, src_orgin = image_preprocess(image_file, display=False)   # 返回的box是降采样十倍的 需要*10
        box = box*10    # 4点坐标
        dic_ = {'name': name, 'percentage': percentage, 'box': [int(box.reshape((-1))[i]) for i in range(8)], 'h, w, c':src_orgin.shape}
        image_4points_list.append(dic_)
        if save_mini_draw:
            cv2.imwrite(os.path.join(mini_save_dir, name), drawsrc)
    # print(image_4points_list)
    with open(img_4points_info_save_json_file, 'w') as f:
        json.dump(image_4points_list, f, indent=2)



        # print(box.shape)
        # print(box)
        # print(name)





        # image_get = img_by_cut(src_orgin, name, box, bycut_imgs_save_dir)


        # cv2.namedWindow("image_get", cv2.WINDOW_NORMAL)
        # cv2.imshow('image_get', image_get)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # box = box * 10
        # box_contour = box[:, np.newaxis, :]
        # cv2.drawContours(src_orgin, [box_contour], -1, (255, 0, 0), 10)
        # cv2.namedWindow("src_orgin", cv2.WINDOW_NORMAL)
        # cv2.imshow('src_orgin', src_orgin)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    if jilu:
        f.close()

# %%


