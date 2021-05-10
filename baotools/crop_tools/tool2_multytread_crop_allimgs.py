import os
import cv2
from PIL import Image
from shutil import copyfile
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 解决image file is truncated (XX bytes not processed)


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
def crop_one_file(ssub_root_dir):
    # for ssub_root_dir in ssub_root_dirs:
    #     pbar.update(1)
    # print(ssub_root_dir)
    files = os.listdir(ssub_root_dir)
    root = ssub_root_dir
    if len(files) > 0:  # has the xml and imgs
        xml_files = [xml for xml in files if xml.endswith('.xml')]
        img_files = [img for img in files if img.split('.')[-1].lower() in img_suffixs]
        if len(xml_files)>0:
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
                img = PIL.Image.open(os.path.join(root, img_file))  # the windows labelimg software does not use the exif imfo, and PIL either

                try:
                    info = get_boxes_from_xml(os.path.join(root, xml_file))
                except Exception:
                    print('error happened when read the', os.path.join(root, xml_file))
                    continue
                for each in info:
                    catogory = each[0]
                    assert catogory in classes_name
                    cropped = img.crop((each[1], each[2], each[3], each[4]))  # PIL读取这样裁剪 (left, upper, right, lower)
                    name = ssub_root_dir.split(src_path)[-1][1:].replace('/', '_')+'_'+img_file.split('.')[0]+'_'+catogory+'.'+img_file.split('.')[-1]
                    # name = os.path.join(name)
                    fpath = os.path.join(save_path, catogory, name)
                    # print(os.listdir())
                    # print(fpath)
                    w, h = cropped.size
                    if h > 10 and w > 10:
                        # cv2.imwrite(fpath, cut_img)   # cv保存图片方法
                        cropped.save(fpath)  # PIL 保存图片方法
    return ssub_root_dir

if __name__ == '__main__':

    src_path = '../../../data/Gucci/OriginalData'
    assert os.path.exists(src_path), 'wrong source path'
    save_path = os.path.join(src_path, './Classification/data')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('Create ', save_path)
    classes_name = []
    classes_name_dic = {}

    #
    print('Step 1 : make a statistics of all category')
    for root, dirs, files in os.walk(src_path, topdown=False):
        if len(files) > 0:  # has the xml and imgs
            # print(root)
            xml_files = [xml for xml in files if xml.endswith('.xml')]
            for xml in xml_files:
                # print(os.path.join(root, xml))
                try:
                    info = get_boxes_from_xml(os.path.join(root, xml))
                except Exception:
                    print('error happened when read the', os.path.join(root, xml))
                    continue
                for each in info:
                    name = each[0]
                    if name not in classes_name_dic.keys():
                        classes_name_dic[name] = 0
                    classes_name_dic[name] += 1
    classes_name = list(classes_name_dic.keys())
    classes_name.sort()
    classes_name_dic = dict(sorted(classes_name_dic.items(), key=lambda x:x[0]))
    print(classes_name)
    print(classes_name_dic)
    for n in classes_name:
        if not os.path.exists(os.path.join(save_path, n)):
            os.makedirs(os.path.join(save_path, n))
    print('Create the classes name dirs')
    print('Step 2 : Begin to crop the image')
    img_suffixs = ['jpg', 'jpeg', 'png', 'tif']
    sub_root_dirs = os.listdir(src_path)
    ssub_root_dirs = []
    for sub_root_dir in sub_root_dirs:
        all = os.listdir(os.path.join(src_path, sub_root_dir))
        for i in all:
            ssub_root_dirs.append(os.path.join(src_path, sub_root_dir, i))

    pbar = tqdm(total=len(ssub_root_dirs), desc=f'crop imgs', unit='dir', ncols=80)


    def crop_one_file(ssub_root_dir):
        # for ssub_root_dir in ssub_root_dirs:
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
                    img = PIL.Image.open(os.path.join(root,
                                                      img_file))  # the windows labelimg software does not use the exif imfo, and PIL either

                    try:
                        info = get_boxes_from_xml(os.path.join(root, xml_file))
                    except Exception:
                        print('error happened when read the', os.path.join(root, xml_file))
                        continue
                    for each in info:
                        catogory = each[0]
                        assert catogory in classes_name
                        cropped = img.crop(
                            (each[1], each[2], each[3], each[4]))  # PIL读取这样裁剪 (left, upper, right, lower)
                        name = ssub_root_dir.split(src_path)[-1][1:].replace('/', '_') + '_' + img_file.split('.')[
                            0] + '_' + catogory + '.' + img_file.split('.')[-1]
                        # name = os.path.join(name)
                        fpath = os.path.join(save_path, catogory, name)
                        # print(os.listdir())
                        # print(fpath)
                        h, w = cropped.size
                        if h > 10 and w > 10:
                            # cv2.imwrite(fpath, cut_img)   # cv保存图片方法
                            cropped.save(fpath)  # PIL 保存图片方法
        return ssub_root_dir

    executor = ThreadPoolExecutor(max_workers=10)
    for i in executor.map(crop_one_file, ssub_root_dirs):
        # print(i)
        pass
    pbar.close()












