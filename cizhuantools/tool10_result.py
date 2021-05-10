from tool2_generate4points import data_dir
import os
import json
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import numpy as np
import math
from pathlib import Path

src_imgs_dir = os.path.join(data_dir, 'tile_round1_testA_20201231/testA_imgs')
img_4points_info_save_json_file = str(Path(src_imgs_dir).parent / 'ims_4points.json')
by_json_file = os.path.join(data_dir, 'myresult/bydect1.json')
zx_json_file = os.path.join(data_dir, 'myresult/zxdect_CamJoinyolov5s_3.json')
all_json_file = os.path.join(data_dir, 'myresult/alldect9.json')

w = 50
def check_jiao_right(cut_4p, jiao_box_xyxy):
    '''
    检测角是否的标注框是否有相交 角的点可能有偏差 偏差设为w=10
    Args:
        cut_4p:
        jiao_box_xyxy:

    Returns:

    '''
    flag = False
    x1, y1, x2, y2 = jiao_box_xyxy
    # w = 20
    for i in range(4):
        [px, py] = cut_4p[i * 2:i * 2 + 2]
        px1, py1, px2, py2 = [px-w, py-w, px+w, py+w]
        ix1, iy1, ix2, iy2 = max(x1, px1), max(y1, py1), min(x2, px2), min(y2, py2)
        iw = ix2 - ix1 + 1
        ih = iy2 - iy1 + 1
        if iw>0 and ih>0:   # 只要有一个相交就返回正常
            return True
    return False


def check_bian_right(cut_4p, bian_box_xyxy):
    '''
    判断边缘框是属于边缘的步骤 边可能有偏差 偏差设为w=10
    1. 四点瓷砖框向外扩大10， 判断是否有标注框的任意顶点在内
    2. 四点瓷砖框内陷10 ， 判断是否有标注框的任意顶点在外
    上面两个条件同时满足 则保留 返回True
    Args:
        cut_4p: 定位的瓷砖四点
        bian_box_xyxy:  标注框

    Returns:
    '''
    # w = 20
    def scale_4p(p4, scale=w): # scale为正则扩大 返回的坐标可能无序 返回的为numpy数组
        p4 = np.array(p4).reshape(4, 1, 2)
        rect = cv2.minAreaRect(p4)
        rect = (rect[0], (rect[1][0]+scale, rect[1][1]+scale), rect[2])
        box = cv2.boxPoints(rect)  # 获取四个顶点坐标
        box = np.int32(box)
        return box
    def check_point_in_contour(p4, box_xyxy):
        x1, y1, x2, y2 = box_xyxy
        box_4p = [x1, y1, x2, y1,
                  x1, y2, x2, y2]
        for i in range(4):
            [x, y] = box_4p[i * 2:i * 2 + 2]
            test_point = (x, y)
            if cv2.pointPolygonTest(p4, test_point, False)>=0: #只要有一个点在轮廓内就返回真
                return True
        return False

    def check_point_out_contour(p4, box_xyxy):
        x1, y1, x2, y2 = box_xyxy
        box_4p = [x1, y1, x2, y1,
                  x1, y2, x2, y2]
        for i in range(4):
            [x, y] = box_4p[i * 2:i * 2 + 2]
            test_point = (x, y)
            if cv2.pointPolygonTest(p4, test_point, False)<=0: #只要有一个点在轮廓外就返回真
                return True
        return False
    # 扩10查看是否包含任意标注框点
    new_np_p4 = scale_4p(cut_4p, scale=w)
    flag = check_point_in_contour(new_np_p4, bian_box_xyxy)
    if not flag:
        return False
    new_np_p4 = scale_4p(cut_4p, scale=-w)
    flag = check_point_out_contour(new_np_p4, bian_box_xyxy)
    return flag

def check_zx_right(cut_4p, box_xyxy):
    '''
    扩大后判断是不是点均在内
    Args:
        cut_4p:
        box_xyxy:

    Returns:

    '''
    # w = 20
    def scale_4p(p4, scale=w): # scale为正则扩大 返回的坐标可能无序 返回的为numpy数组
        p4 = np.array(p4).reshape(4, 1, 2)
        rect = cv2.minAreaRect(p4)
        rect = (rect[0], (rect[1][0]+scale, rect[1][1]+scale), rect[2])
        box = cv2.boxPoints(rect)  # 获取四个顶点坐标
        box = np.int32(box)
        return box

    def check_point_allin_contour(p4, box_xyxy):
        x1, y1, x2, y2 = box_xyxy
        box_4p = [x1, y1, x2, y1,
                  x1, y2, x2, y2]
        for i in range(4):
            [x, y] = box_4p[i * 2:i * 2 + 2]
            test_point = (x, y)
            if cv2.pointPolygonTest(p4, test_point, False)<0: #只要有一个点在轮廓外就返回假
                return False
        return True

    new_np_p4 = scale_4p(cut_4p, scale=w)
    return check_point_allin_contour(new_np_p4, box_xyxy)



if __name__ == '__main__':

    with open(by_json_file, 'r') as f:
        by_json_list = json.load(f)

    with open(zx_json_file, 'r') as f:
        zx_json_list = json.load(f)
    image_4p_dic = {}
    with open(img_4points_info_save_json_file, 'r') as f:
        load_4p = json.load(f)
    for i in load_4p:
        name = i['name']
        _4p = i['box']
        image_4p_dic[name] = _4p

    all_json_list = []
    for a in by_json_list:
        cut_name = a['cut_name']
        name = cut_name.split('_by_')[0] + '.jpg'
        info = cut_name.split('_by_')[1].split('.')[0].split('_')
        x_ = int(info[3])
        y_ = int(info[4])
        x0, y0, x1, y1 = a['bbox']
        x0 += x_
        x1 += x_
        y0 += y_
        y1 += y_
        bbox = [x0, y0, x1, y1]
        # print(bbox)
        score = a['confidence']
        category = a['cut_category'] + 1

        if category == 2:  # 如果是角异常
            flag = check_jiao_right(image_4p_dic[name], bbox)
            if not flag:
                continue
        elif category == 1:  # 如果是边缘异常
            flag = check_bian_right(image_4p_dic[name], bbox)
            if not flag:
                continue
        dict = {'name': name, 'category': category, 'bbox': bbox, 'score': score}

        all_json_list.append(dict)

    for a in zx_json_list:
        cut_name = a['cut_name']
        name = cut_name.split('_zx_')[0] + '.jpg'
        info = cut_name.split('_zx_')[1].split('.')[0].split('_')
        x_ = int(info[3])
        y_ = int(info[4])
        x0, y0, x1, y1 = a['bbox']
        x0 += x_
        x1 += x_
        y0 += y_
        y1 += y_
        bbox = [x0, y0, x1, y1]
        # print(bbox)
        score = a['confidence']
        category = a['cut_category'] + 3

        if not check_zx_right(image_4p_dic[name], bbox):
            continue

        dict = {'name': name, 'category': category, 'bbox': bbox, 'score': score}

        all_json_list.append(dict)
        # cv2.pointPolygonTest()

    with open(all_json_file, 'w') as f:
        json.dump(all_json_list, f, indent=2)
