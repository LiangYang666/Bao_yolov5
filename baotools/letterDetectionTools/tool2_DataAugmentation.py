#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tool2DataAugmentation.py
# @Time      :2021/8/3 下午5:16
# @Author    :Yangliang
import os
from tkinter import Tk

import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from matplotlib import pyplot as plt
from tqdm import tqdm

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
imgaug_seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        # iaa.Fliplr(0.3),  # horizontally flip 50% of all images
        # iaa.Flipud(0.2),  # vertically flip 20% of all images
        # crop images by -2.5% to 5% of their height/width
        # sometimes(iaa.CropAndPad(
        #     percent=(-0.03, 0.08),
        #     pad_mode=ia.ALL,
        #     pad_cval=(0, 255)
        # )),
        # sometimes(iaa.Affine(
        #     scale={"x": (0.8, 1.1), "y": (0.8, 1.1)},
        #     # scale images to 80-120% of their size, individually per axis
        #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
        #     rotate=(-20, 20),  # rotate by -45 to +45 degrees
        #     shear=(-16, 16),  # shear by -16 to +16 degrees
        #     order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
        #     cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
        #     # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        # )),
        iaa.Sometimes(0.8, iaa.AddToHueAndSaturation((-50, 50))),  # change hue and saturation
        # either change the brightness of the whole image (sometimes
        # per channel) or change the brightness of subareas

        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((3, 6),
                   [
                       # sometimes(iaa.Superpixels(p_replace=(0, 0.2), n_segments=100)),    # 去除临近填充
                       # convert images into their superpixel representation
                       # iaa.OneOf([
                       #     iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                       #     iaa.AverageBlur(k=(2, 7)),
                       #     # blur image using local means with kernel sizes between 2 and 7
                       #     iaa.MedianBlur(k=(3, 11)),
                       #     # blur image using local medians with kernel sizes between 2 and 7
                       # ]),

                       iaa.GaussianBlur((0, 2.0)),  # blur images with a sigma between 0 and 3.0
                       iaa.Sharpen(alpha=(0, 0.2), lightness=(0.85, 1.2)),  # sharpen images
                       iaa.Emboss(alpha=(0, 0.1), strength=(0, 1.5)),  # emboss images
                       # search either for all edges or for directed edges,
                       # blend the result with the original image using a blobby mask
                       iaa.SimplexNoiseAlpha(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0.1, 0.16)),
                           iaa.DirectedEdgeDetect(alpha=(0.1, 0.16), direction=(0.0, 1.0)),
                       ])),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.015 * 255), per_channel=0.5),
                       # add gaussian noise to images

                       # iaa.Sometimes(0.3, iaa.Cartoon()),
                       # iaa.OneOf([
                       #     iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                       #     iaa.CoarseDropout((0.03, 0.05), size_percent=(0.01, 0.03), per_channel=0.2),
                       # ]),
                       iaa.Invert(0.05, per_channel=True),  # invert color channels
                       iaa.Add((-10, 10), per_channel=0.5),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.Pepper(0.03),  # 添加椒盐噪声

                       iaa.OneOf([
                           iaa.Multiply((0.9, 1.1), per_channel=0.5),
                           iaa.FrequencyNoiseAlpha(
                               exponent=(-4, 0),
                               first=iaa.Multiply((0.8, 1.2), per_channel=True),
                               second=iaa.LinearContrast((0.5, 2.0))
                           )
                       ]),
                       iaa.LinearContrast((0.7, 1.2), per_channel=0.5),  # improve or worsen the contrast
                       # iaa.Grayscale(alpha=(0.0, 1.0)),
                       # sometimes(iaa.ElasticTransformation(alpha=(0.5, 2.5), sigma=0.25)),
                       # move pixels locally around (with random strengths)
                       # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.02))),
                       # sometimes move parts of the image around
                       # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.05))),

                       iaa.ChangeColorTemperature((1100, 10000))
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)

def aug_one_image(image, aug_seq):
    return aug_seq(images=[image])[0]


def load_batch(batch_idx):
    # dummy function, implement this
    # Return a numpy array of shape (N, height, width, #channels)
    # or a list of (height, width, #channels) arrays (may have different image
    # sizes).
    # Images should be in RGB for colorspace augmentations.
    # (cv2.imread() returns BGR!)
    # Images should usually be in uint8 with values from 0-255.

    src_images_dir = "/media/D_4TB/YL_4TB/BaoDetection/data/Chanel/LetterDetection/data/sign/imgxmls/"
    files = sorted(os.listdir(src_images_dir))
    img_suffixs = ['jpg', 'jpeg', 'png', 'tif']
    files = [x for x in files if x.rsplit('.', 1)[-1].lower() in img_suffixs]

    images = []
    for file in files:
        file_name = os.path.join(src_images_dir, file)
        image = plt.imread(file_name)
        images.append(image)
    return images


def train_on_images(images):
    # dummy function, implement this
    pass

seq_test = iaa.Sequential([
    # iaa.Sometimes(0.8, iaa.AddToHueAndSaturation((-50, 50))),  # change hue and saturation

    # iaa.Superpixels(p_replace=(0, 0.2), n_segments=100),


    # # convert images into their superpixel representation
    # iaa.OneOf([
    #     iaa.GaussianBlur(2),  # blur images with a sigma between 0 and 3.0
    #     iaa.AverageBlur(k=7),
    #     # blur image using local means with kernel sizes between 2 and 7
    #     iaa.MedianBlur(k=(3, 11)),
    #     # blur image using local medians with kernel sizes between 2 and 7
    # ]),

    iaa.Sharpen(alpha=0.2, lightness=(0.85, 1.15)),  # sharpen images

    iaa.Emboss(alpha=0.1, strength=(0, 2.0)),  # emboss images

    # # search either for all edges or for directed edges,
    # # blend the result with the original image using a blobby mask
    iaa.SimplexNoiseAlpha(iaa.OneOf([
        iaa.EdgeDetect(alpha=0.15),
        iaa.DirectedEdgeDetect(alpha=0.15, direction=(0.0, 1.0)),
    ])),

    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    # # add gaussian noise to images
    #
    # iaa.Sometimes(0.3, iaa.Cartoon()),
    # # iaa.OneOf([
    # #     iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
    # #     iaa.CoarseDropout((0.03, 0.05), size_percent=(0.01, 0.03), per_channel=0.2),
    # # ]),
    # iaa.Invert(0.05, per_channel=True),  # invert color channels

    # iaa.Add((-10, 10), per_channel=0.5),

    # # change brightness of images (by -10 to 10 of original value)
    # iaa.Pepper(0.02),  # 添加椒盐噪声
    #
    # iaa.OneOf([
    #     iaa.Multiply((0.8, 1.2), per_channel=0.5),
    #     iaa.FrequencyNoiseAlpha(
    #         exponent=(-4, 0),
    #         first=iaa.Multiply((0.8, 1.2), per_channel=True),
    #         second=iaa.LinearContrast((0.5, 2.0))
    #     )
    # ]),
    # iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
    # # iaa.Grayscale(alpha=(0.0, 1.0)),
    # sometimes(iaa.ElasticTransformation(alpha=(0.5, 2.5), sigma=0.25)),
    # # move pixels locally around (with random strengths)
    # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.02))),
    # # sometimes move parts of the image around
    # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.05))),
    #
    # iaa.ChangeColorTemperature((1100, 10000))
])

def aug_save_img(aug_times = 100):
    src_images_dir = "/media/D_4TB/YL_4TB/BaoDetection/data/Chanel/LetterDetection/data/sign/imgxmls"
    aug_img_dir = '/media/D_4TB/YL_4TB/BaoDetection/data/Chanel/LetterDetection/data/sign/augmented_imgs'
    files = sorted(os.listdir(src_images_dir))
    img_suffixs = ['jpg', 'jpeg', 'png', 'tif']
    files = [x for x in files if x.rsplit('.', 1)[-1].lower() in img_suffixs]
    for file in tqdm(files):
        # src_img = plt.imread(os.path.join(src_images_dir, file))
        src_img = cv2.imread(os.path.join(src_images_dir, file), cv2.IMREAD_COLOR+cv2.IMREAD_IGNORE_ORIENTATION)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        suffix = file.rsplit('.', 1)[-1]
        file_path = os.path.join(aug_img_dir, file.rsplit('.', 1)[0]+'_000.'+suffix)
        plt.imsave(file_path, src_img)
        for t in range(aug_times):
            index = str(t + 1).zfill(3)
            file_path = os.path.join(aug_img_dir, file.rsplit('.', 1)[0] + f'_{index}.' + suffix)
            aug_img = aug_one_image(src_img, imgaug_seq)
            plt.imsave(file_path, aug_img)




def main():
    images = load_batch(0)
    # images_aug = seq(images=images)
    # images_aug = seq_test(images=images)

    images_aug = []
    for image in images:
        # images_aug.append(aug_one_image(image, imgaug_seq))
        images_aug.append(aug_one_image(image, seq_test))
    id = 0

    while id < len(images_aug):
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.imshow(images[id])
        plt.subplot(1, 2, 2)
        plt.imshow(images_aug[id])
        plt.pause(0.01)
        key_press = 0
        img = images_aug[id]
        while True:
            pos = plt.ginput(n=1, timeout=1000)  # n is times »
            # print(pos)
            if len(pos) > 0:
                if pos[0][0] > img.shape[1] / 2:
                    id += 1
                    break
                if pos[0][0] < img.shape[1] / 2:
                    id -= 1
                    break

        plt.cla()


if __name__ == "__main__":
    aug_save_img(aug_times=100)

    exit(0)


    main()

    images = load_batch(0)



    # images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
    seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.GaussianBlur((0, 3.0))])

    # Show an image with 8*8 augmented versions of image 0 and 8*8 augmented
    # versions of image 1. Identical augmentations will be applied to
    # image 0 and 1.
    # seq.show_grid([images[0], images[1]], cols=2, rows=2)

    # win = Tk()

    # for batch_idx in range(100):
    #     images = load_batch(batch_idx)
    #     images_aug = seq(images=images)  # done by the library
    #     train_on_images(images_aug)
