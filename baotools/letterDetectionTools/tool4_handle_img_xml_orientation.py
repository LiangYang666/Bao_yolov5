#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool4_handle_img_xml_orientation.py
# @Time      :2021/9/2 下午12:19
# @Author    :Yangliang

# 实现对win版的labelimg软件打的标签和图片进行更正  均根据exif信息进行更正 图片存储为普通旋转后的图片
# win版本的labelimg不会根据exif信息对图片进行旋转 而cv2读取的图片会根据exif信息进行旋转
import os

import PIL
from PIL import Image, ImageOps
import xml.etree.ElementTree as ET

from tqdm import tqdm

'''
It's EXIF tag 274
1: rotate 0 degrees 
6: rotate 90 degrees
3: rotate 180 degrees
8: rotate 270 degrees
'''


def trans2after_rotate(w, h, x1, y1, x2, y2, degrees):
    degrees_list = [0, 90, 180, 270]
    assert degrees in degrees_list
    if degrees == 0:
        return x1, y1, x2, y2
    elif degrees==90:
        return h+1-y2, x1, h+1-y1, x2
    elif degrees==180:
        return w+1-x2, h+1-y2, w+1-x1, h+1-y1
    elif degrees==270:
        return y1, w+1-x2, y2, w+1-x1

def change_xml(xml_src, xml_dst, degrees):
    xmlDoc = ET.parse(xml_src)
    root = xmlDoc.getroot()
    size = root.find('size')
    origin_w = int(size.find('width').text.strip())
    origin_h = int(size.find('height').text.strip())
    if degrees == 90 or degrees == 270:
        size.find('width').text = str(origin_h)
        size.find('height').text = str(origin_w)

    for object in root.findall('object'):
        name = object.find('name').text
        bandbox = object.find('bndbox')
        origin_x1 = int(bandbox.find('xmin').text)
        origin_y1 = int(bandbox.find('ymin').text)
        origin_x2 = int(bandbox.find('xmax').text)
        origin_y2 = int(bandbox.find('ymax').text)
        x1, y1, x2, y2 = trans2after_rotate(origin_w, origin_h, origin_x1, origin_y1, origin_x2, origin_y2, degrees)
        bandbox.find('xmin').text = str(x1)
        bandbox.find('ymin').text = str(y1)
        bandbox.find('xmax').text = str(x2)
        bandbox.find('ymax').text = str(y2)
    xmlDoc.write(xml_dst, 'utf-8', True)



def trans_img_xml_2_new_dir(src_img_dir, src_xml_dir, dst_img_dir, dst_xml_dir):
    rotate_dic = {1: 0, 6: 90, 3: 180, 8: 270}
    img_suffixs = ['jpg', 'jpeg', 'png', 'tif']
    img_names = os.listdir(src_img_dir)
    img_names = sorted([x for x in img_names if x.rsplit('.', 1)[-1].lower() in img_suffixs])
    xml_names = os.listdir(src_xml_dir)
    xml_names = sorted([x for x in xml_names if x.rsplit('.', 1)[-1].lower() == 'xml'])
    for img_name in tqdm(img_names):
        img_path = os.path.join(src_img_dir, img_name)
        img = Image.open(img_path)
        exif_data = img._getexif()
        degrees = 0
        if exif_data != None:
            if 274 in exif_data:
                degrees = rotate_dic[exif_data[274]]
        xml_name = img_name.rsplit('.', 1)[0]+'.xml'
        if xml_name in xml_names:
            xml_names.remove(xml_name)
            xml_src = os.path.join(src_xml_dir, xml_name)
            xml_dst = os.path.join(dst_xml_dir, xml_name)
            change_xml(xml_src, xml_dst, degrees)
        dst_img_path = os.path.join(dst_img_dir, img_name)
        img = ImageOps.exif_transpose(img)
        # img.show()
        img.save(dst_img_path)





if __name__ == "__main__":
    # path = '/media/D_4TB/YL_4TB/BaoDetection/data/Chanel/LetterDetection/data/sign/sign_2_need_已鉴定标记/Chanel-2-doing_726493_logo图_sign.jpeg'
    # path2 = '/media/D_4TB/YL_4TB/BaoDetection/data/Chanel/LetterDetection/data/sign/sign_2_need_已鉴定标记/Chanel-2-doing_714360_logo图_sign.jpeg'
    # img = Image.open(path2)
    # exif_data = img._getexif()
    #
    # exit()


    src_img_dir = '/media/D_4TB/YL_4TB/BaoDetection/data/Chanel/LetterDetection/data/sign/sign_2_need_已鉴定标记'
    src_xml_dir = '/media/D_4TB/YL_4TB/BaoDetection/data/Chanel/LetterDetection/data/sign/sign_2_need_已鉴定标记'
    dst_img_dir = '/media/D_4TB/YL_4TB/BaoDetection/data/Chanel/LetterDetection/data/sign/sign_2_need_已鉴定标记_包含标签旋转后'
    dst_xml_dir = '/media/D_4TB/YL_4TB/BaoDetection/data/Chanel/LetterDetection/data/sign/sign_2_need_已鉴定标记_包含标签旋转后'
    trans_img_xml_2_new_dir(src_img_dir, src_xml_dir, dst_img_dir, dst_xml_dir)




