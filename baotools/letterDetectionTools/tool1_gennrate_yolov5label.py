#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool1_gennrate_yolov5label.py
# @Time      :2021/8/29 下午5:38
# @Author    :Yangliang
import os
import xml.etree.ElementTree as ET

import PIL.Image
from concurrent.futures import ThreadPoolExecutor
from PIL import ImageFile
from tqdm import tqdm


def get_boxes_from_xml(file):
    tree = ET.parse(file)  # 获取xml文件
    root = tree.getroot()
    filename = root.find('filename').text
    # object = root.find('object')
    info = []
    for object in root.findall('object'):
        name = object.find('name').text
        bandbox = object.find('bndbox')
        xmin = int(bandbox.find('xmin').text)
        ymin = int(bandbox.find('ymin').text)
        xmax = int(bandbox.find('xmax').text)
        ymax = int(bandbox.find('ymax').text)
        each = [name, xmin, ymin, xmax, ymax]
        info.append(each)
    return info


def get_image_name(xml, imgs):
    pre = xml.rsplit('.', 1)[-2]
    for img in imgs:
        if img.rsplit('.', 1)[-2] == pre:
            return img
    return None


def get_category_names(xmls_path):
    xmls = os.listdir(xmls_path)
    category_names = []
    xmls = [x for x in xmls if x.rsplit('.', 1)[-1].lower()=='xml']
    for xml in xmls:
        info = get_boxes_from_xml(os.path.join(xmls_path, xml))
        for each in info:
            name = each[0]
            if name not in category_names:
                category_names.append(name)
    return category_names



def write_all_yolo_labels_and_txts(img_xml_datas: list, imgs_path: str, xmls_path: str, categories_name: list,
                                   labels_path, txt_path, max_workers=10):
    # 将得到的所有图像xml标签对，转换成yolo v5的标签labels文件夹和all.txt文件

    # 参数 img_xml_datas 图像标签对列表，列表元素为二元列表 二元列表第一个元素为图片名称，第二个元素为xml文件名称
    # 参数 imgs_path 图片文件夹所处的路径
    # 参数 xmls_path xml文件的父级目录
    # 参数 categories_name 类别名列表，yolo存储的序号将按照所给的标签顺序标号
    # 参数 labels_path 要输出的yolo版标签txt文件位置 一般为 与图片所处的同级路径/labels
    # 参数 txt_path 要输出的yolo版txt文件位置 一般为 与图片所处的同级路径/all.txt

    def generate_yolotxt_1by1(z):  # 多线程封装使用
        return generate_yolotxt_1by1_real(z[0], z[1], imgs_path, xmls_path, labels_path, categories_name)

    def generate_yolotxt_1by1_real(img_name, xml_name, imgs_path, xmls_path, labels_path, categories_name):
        # 根据所给的xml文件名和图片名 写出labels文件夹下的yolo版标签

        # 返回值 对应的yolo 标签名称和图片名称
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

    executor = ThreadPoolExecutor(max_workers=max_workers)
    all_imgs_txts = []
    for i in executor.map(generate_yolotxt_1by1, tqdm(img_xml_datas)):
        if i:
            all_imgs_txts.append(i)
    print(f'3.Writing all yolov5 images and txt files\' name to {txt_path}.')
    all_imgs_txts = sorted(all_imgs_txts)
    with open(txt_path, 'w') as f:
        for i in tqdm(all_imgs_txts):
            f.write(' '.join(i) + '\n')


def get_imgs_xmls_datas(xml_names: list, img_names: list,
                        max_workers=4):  # 通过多线程对xml列表和图片列表进行对应，如果是成对的xml名和图片名，则将其存储，最终返回n个两元素列表
    # 输入参数xml_names 所有xml文件的名称
    # 输入参数img_names 所有图像文件的名称
    img_xml_datas = []

    def get_imgs_xmls_list(xml_name):
        pre_name = xml_name.split('.')[0]
        for img_name in img_names:
            if img_name.split('.')[0] == pre_name:
                return (img_name, xml_name)
        return None

    executor = ThreadPoolExecutor(max_workers=max_workers)
    for i in executor.map(get_imgs_xmls_list, tqdm(xml_names)):
        if i:
            img_xml_datas.append(i)

    return img_xml_datas

def xml_trans_to_yolo_labels(imgs_path, xmls_path, labels_path, txt_path):
    # 将xml版本的标签转换为yolo v5版本的标签

    # 参数 imgs_path 图片文件的父级目录
    # 参数 xmls_path xml文件的父级目录
    # 参数 labels_path 要输出的yolo版标签txt文件位置 一般为 与图片所处的同级路径/labels
    # 参数 txt_path 要输出的yolo版txt文件位置 一般为 与图片所处的同级路径/all.txt
    categories_name = get_category_names(xmls_path)
    print(categories_name)

    if not os.path.exists(labels_path):
        os.mkdir(labels_path)
        print(f"create {labels_path}")

    xml_names = os.listdir(xmls_path)
    xml_names = [x for x in xml_names if x.rsplit('.', 1)[-1].lower()=='xml']
    img_suffixs = ['jpg', 'jpeg', 'png', 'tif']
    img_names = os.listdir(imgs_path)
    img_names = [x for x in img_names if x.rsplit('.', 1)[-1].lower() in img_suffixs]
    img_xml_datas = get_imgs_xmls_datas(xml_names, img_names)
    write_all_yolo_labels_and_txts(img_xml_datas, imgs_path, xmls_path, categories_name, labels_path, txt_path)


if __name__ == "__main__":
    brand = 'Chanel'
    part = 'sign'
    path = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}'
    labels_path = os.path.join(path, 'labels')
    imgxmls_path = os.path.join(path, 'imgxmls')
    all_txts_path = os.path.join(path, 'all.txt')

    xml_trans_to_yolo_labels(imgxmls_path, imgxmls_path, labels_path, all_txts_path)















