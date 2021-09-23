#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool5_get_intersect_C_imgxml.py
# @Time      :2021/9/2 下午3:53
# @Author    :Yangliang

# 获取正确检测到字母框 且xml标签含有对对应字母（如C非正圆）的标记框的图片及xml 将其复制到指定文件夹内做进一步处理 并裁剪有问题的字母图到新文件夹
import json
import os
import re
import shutil
import warnings
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt


def calculate_iou(bbox1, bbox2):
    bb = bbox1
    bbgt = bbox2
    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1
    ov = -1
    if iw > 0 and ih > 0:
        # compute overlap (IoU) = area of intersection / area of union
        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
        ov = iw * ih / ua
    return ov


def check_if_intersect(xml_path, xyxy):
    xmlDoc = ET.parse(xml_path)
    root = xmlDoc.getroot()
    for object in root.findall('object'):
        name = object.find('name').text
        bandbox = object.find('bndbox')
        x1 = int(bandbox.find('xmin').text)
        y1 = int(bandbox.find('ymin').text)
        x2 = int(bandbox.find('xmax').text)
        y2 = int(bandbox.find('ymax').text)
        if calculate_iou(xyxy, [x1, y1, x2, y2]) > 0.2:
            print(xml_path, calculate_iou(xyxy, [x1, y1, x2, y2]))
            return name
    return False


def get_detected_xyxy_of_category(detected_info, img_name, category_id):
    # 获取检测到的json文件中指定的category对应的xyxy框
    if detected_info[img_name].__contains__(category_id):
        bboxs = detected_info[img_name][category_id]
    else:
        bboxs = []
    return bboxs


def get_json_info(detected_img_dir):
    # detected_img_dir 为检测的输出文件夹， 文件夹内是已绘制检测框的图片以及一个json文件
    # 将检测得到的json文件的信息解析为字典并返回
    # 返回字典形如 detected_info[img_name][category_id]=[bbox1, bbox2...] 其中bbox为4元列表，存储x1,y1,x2,y2

    detected_info = {}
    files = os.listdir(detected_img_dir)
    json_files = [x for x in files if x.rsplit('.')[-1].lower() == 'json']
    assert len(json_files)==1
    json_file = json_files[0]
    with open(os.path.join(detected_img_dir, json_file), 'r') as f:
        dics = json.load(f)
    for dic in dics:
        img_name = dic['cut_name']
        category_id = dic['cut_category']
        bbox = dic['bbox']
        if img_name in files:
            if img_name not in detected_info.keys():
                detected_info[img_name] = {category_id:[bbox]}
            else:
                if category_id in detected_info[img_name].keys():
                    detected_info[img_name][category_id].append(bbox)
                else:
                    detected_info[img_name][category_id] = [bbox]
    return detected_info


def copy_intersect_img_xml_cutimg(detected_info, detected_img_name, category_id, src_img_dir, src_xml_dir, dst_img_dir, dst_xml_dir, dst_cut_img_dir):
    # detected_img_name = 'Chanel-2-doing_710803_金属件_sign.jpg'
    xml_name = detected_img_name.rsplit('.', 1)[0]+'.xml'
    src_xml_path = os.path.join(src_xml_dir, xml_name)
    get_total = 0
    if os.path.exists(src_xml_path):
        detected_xyxys = get_detected_xyxy_of_category(detected_info, detected_img_name, category_id)
        for i, detected_xyxy in enumerate(detected_xyxys):
            rs = check_if_intersect(src_xml_path, detected_xyxy)
            if isinstance(rs, str):
                # 存储完整图片
                src = os.path.join(src_img_dir, detected_img_name)
                dst = os.path.join(dst_img_dir, detected_img_name)
                shutil.copy(src, dst)
                img = cv2.imread(src)
                src = os.path.join(src_xml_dir, xml_name)
                dst = os.path.join(dst_xml_dir, xml_name)
                shutil.copy(src, dst)
                # 存储裁剪图片
                x1, y1, x2, y2 = detected_xyxy
                img_cut = img[y1:y2 + 1, x1:x2 + 1]
                img_save_path = os.path.join(dst_cut_img_dir, img_name)
                temp = img_save_path.rsplit('.', 1)
                rstr = r"[\/\\\:\*\?\"\<\>\|\.\s]"  # '/ \ : * ? " < > | .和空格'
                rs = re.sub(rstr, "", rs)
                img_save_path = f'{temp[0]}_{rs}_{i}.{temp[1]}'
                cv2.imwrite(img_save_path, img_cut)

                get_total += 1
    return get_total

def get_category_names(categories_file):
    with open(categories_file, 'r') as f:
        lines = f.readlines()
        categories = [line.strip() for line in lines if len(line.strip()) >= 1]
    return categories


def check_and_build(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)


def remove_and_rebuild(*dirs):
    for dir in dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)



from tool1_gennrate_yolov5label import brand, part
# brand = 'Chanel'
# part = 'sign'
if __name__ == "__main__":
    detected_img_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/{part}_2_handle/1detect_select_ok'
    categories_file = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/categories.txt'
    src_img_dir = src_xml_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/{part}_2_need_已鉴定标记_包含标签旋转后'

    detected_info = get_json_info(detected_img_dir)
    img_names = sorted(list(detected_info.keys()))
    categories = get_category_names(categories_file)

    for img_name in img_names:
        if len(detected_info[img_name]) != len(categories):
            warnings.warn(f'{img_name} 可能未准确识别到')
    print('all_categories: ', categories)
    dst_p_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/{part}_2_handle/2与1重复出现且对某字母打了标签的'
    dst_cut_p_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/{part}_2_handle/3裁剪后鉴定有问题的字母图片'
    check_and_build(dst_p_dir, dst_cut_p_dir)

    for category in categories:
        dst_img_dir = dst_xml_dir = os.path.join(dst_p_dir, category)
        dst_cut_img_dir = os.path.join(dst_cut_p_dir, category)
        remove_and_rebuild(dst_img_dir, dst_cut_img_dir)
        category_id = categories.index(category)
        total = 0
        for img_name in img_names:
            total += copy_intersect_img_xml_cutimg(detected_info, img_name, category_id, src_img_dir, src_xml_dir, dst_img_dir, dst_xml_dir, dst_cut_img_dir)
        copied_img_names = os.listdir(dst_img_dir)

        detected_selected_info = {}
        for img_name in detected_info.keys():
            if img_name in copied_img_names:
                detected_selected_info[img_name] = detected_info[img_name]
        print(f'{category}({category_id}): total {total}')
        with open(os.path.join(dst_img_dir, f'{category}_intersect_dicts.json'), 'w') as f:
            json.dump(detected_selected_info, f, indent=2)

        # for name in detected_info.keys():
        #     if '740100' in name:
        #         img_orgin = cv2.imread(os.path.join(src_img_dir, name))
        #         for bbox in detected_info[name]['3']:
        #             x1, y1, x2, y2 = bbox
        #             cv2.rectangle(img_orgin, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        #         save_path = os.path.join(os.path.dirname(dst_p_dir), name)
        #         cv2.imwrite(save_path, img_orgin)







