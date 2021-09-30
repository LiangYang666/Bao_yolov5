#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool1_split_train_test.py
# @Time      :2021/9/6 下午4:54
# @Author    :Yangliang
import os
import random
import shutil

from baotools.letterDetectionTools.tool5_cut_intersect_letter_imgxml import get_category_names

brand = 'LV'
part = 'sign'

if __name__ == "__main__":

    true_imgs_P_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/{part}_1_handle/2裁剪后正确真类字母'
    fake_imgs_P_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/{part}_2_handle/3裁剪后鉴定有问题的字母图片'
    categories_file = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/categories.txt'
    dst_img_P_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{brand}/LetterDetection/data/{part}/classification_data'
    ratio = 0.7
    img_suffixs = ['jpg', 'jpeg', 'png', 'tif']
    categories = get_category_names(categories_file)
    for category in categories:
        print("类别:", category)
        # 生成目录
        print("1----生成目录")
        def delete_and_rebuild(dir):
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.mkdir(dir)
        category_save_dir = os.path.join(dst_img_P_dir, category)
        if not os.path.exists(dst_img_P_dir):
            os.mkdir(dst_img_P_dir)
        if not os.path.exists(category_save_dir):
            os.mkdir(category_save_dir)
        train_imgs_dir = os.path.join(category_save_dir, 'train')
        test_imgs_dir = os.path.join(category_save_dir, 'test')
        if not os.path.exists(train_imgs_dir):
            os.mkdir(train_imgs_dir)
        if not os.path.exists(test_imgs_dir):
            os.mkdir(test_imgs_dir)
        train_true = os.path.join(train_imgs_dir, 'true')
        train_fake = os.path.join(train_imgs_dir, 'fake')
        test_true = os.path.join(test_imgs_dir, 'true')
        test_fake = os.path.join(test_imgs_dir, 'fake')
        delete_and_rebuild(train_true)
        delete_and_rebuild(train_fake)
        delete_and_rebuild(test_true)
        delete_and_rebuild(test_fake)
        print("2---抽取图片")
        category_true_files = os.listdir(os.path.join(true_imgs_P_dir, category))
        category_true_imgs = [x for x in category_true_files if x.rsplit('.', 1)[-1].lower() in img_suffixs]
        category_fake_files = os.listdir(os.path.join(fake_imgs_P_dir, category))
        category_fake_imgs = [x for x in category_fake_files if x.rsplit('.', 1)[-1].lower() in img_suffixs]
        min_img_n = min(len(category_true_imgs), len(category_fake_imgs))
        random.seed(1)      # 指定随机种子
        random.shuffle(category_true_imgs)
        random.shuffle(category_fake_imgs)
        print("3---复制图片")
        for i in range(min_img_n):
            if i<min_img_n*ratio:
                shutil.copy(os.path.join(true_imgs_P_dir, category, category_true_imgs[i]), os.path.join(train_true, category_true_imgs[i]))
                shutil.copy(os.path.join(fake_imgs_P_dir, category, category_fake_imgs[i]), os.path.join(train_fake, category_fake_imgs[i]))
            else:
                shutil.copy(os.path.join(true_imgs_P_dir, category, category_true_imgs[i]), os.path.join(test_true, category_true_imgs[i]))
                shutil.copy(os.path.join(fake_imgs_P_dir, category, category_fake_imgs[i]), os.path.join(test_fake, category_fake_imgs[i]))
        print("4---完成")








