import os
import json
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import numpy as np
from pathlib import Path

from tool2_generate4points import data_dir
RaceCutImageDir = os.path.join(data_dir, 'tile_round1_testA_20201231/testA_imgs_zxcutsave')
Cam = ['CAM1', 'CAM2', 'CAM3']
# 生成三种相机的图片txt
if __name__ == "__main__":
    imgFs = os.listdir(RaceCutImageDir)
    CutCamImgFDic = dict(zip(Cam, [[], [], []]))
    for name in imgFs:
        for c in Cam:
            if c in name:
                p = os.path.join('tile_round1_testA_20201231/testA_imgs_zxcutsave', name)
                CutCamImgFDic[c].append(p+'\n')
                break
    for c in Cam:
        with open(os.path.join(data_dir, f'RaceZxTest{c}.txt'), 'w') as f:
            f.writelines(CutCamImgFDic[c])