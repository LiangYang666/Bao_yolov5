import os
import json
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import numpy as np
from pathlib import Path

from tool2_generate4points import data_dir
prezx_txt = os.path.join(data_dir, 'testzx0.txt')

CAM = ['CAM1', 'CAM2', 'CAM3']

if __name__ == "__main__":
    with open(prezx_txt, 'r') as f:
        lines = f.readlines()
    # lines = [x.strip() for x in lines]
    # label_C_dic = dict(zip(CAM, [[]]*len(CAM)))   # 该种方法会是的列表复制 内存共享
    label_C_dic = dict(zip(CAM, [[], [], []]))
    # print(label_C_dic)

    for l in lines:
        for c in CAM:
            if c in l:
                # print(c)
                # print(l)
                label_C_dic[c].append(l)
                # print(label_C_dic[CAM[0]])
                break

    for c in CAM:
        p = Path(prezx_txt)
        with open(os.path.join(str(p.parent), c+str(p.name)), 'w') as f:
            f.writelines(label_C_dic[c])
    # print(label_C_dic[CAM[0]])
