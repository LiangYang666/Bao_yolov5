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

from tool2_generate4points import data_dir
zx_json_file = os.path.join(data_dir, 'myresult/zxdect_CamJoinyolov5s_3.json')

Cam = ['CAM1', 'CAM2', 'CAM3']
zx_json = []
if __name__ == '__main__':
    for c in Cam:
        with open(os.path.join(data_dir, f'myresult/dectckp_{c}allzx0_4c_epoch300.json'), 'r') as f:
            j = json.load(f)
        assert isinstance(j, list)
        zx_json += j

    with open(zx_json_file, 'w') as f:
        json.dump(zx_json, f, indent=2)
