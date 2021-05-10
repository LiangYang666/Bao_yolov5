import os
import json
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm

data_dir = '../data/CS1/tile_round1_train_20201231'
json_file = os.path.join(data_dir, 'train_annos.json')

def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in plt.rcParams['axes.prop_cycle'].by_key()['color']]

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == '__main__':
    with open(json_file, 'r') as load_f:
      load_list = json.load(load_f)

    image_dict = {}    # 每张图片一个字典key 其value为字典 字典中为高宽及框的标签列表  标签列表中含n个框 每个框为一个列表 列表5个值 类别及两点坐标
    for label in load_list[:]:
        name = label['name']
        image_height = label['image_height']
        image_width = label['image_width']
        l_bboxes = [label['category']]+label['bbox']
        if name not in image_dict.keys():
            image_dict[name] = {'image_height': image_height, 'image_width': image_width, 'l_bboxes': []}
        image_dict[name]['l_bboxes'].append(l_bboxes)

    print(image_dict)


    colors = color_list()  # list of colors
    print(colors)


    #
    img_save_dir = os.path.join('../data/CS1', 'labeld_trainimg')
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    # 2497/5388
    # %%
    image_file = os.path.join(data_dir, 'train_imgs', '245_38_t20201128141050945_CAM1.jpg')
    src = cv2.imread(image_file, cv2.IMREAD_COLOR)

    # 245_38_t20201128141050945_CAM1.jpg # 图片不完整 损坏

    # %%
    for name in tqdm(image_dict.keys()):
        image_file = os.path.join(data_dir, 'train_imgs', name)
        src = cv2.imread(image_file, cv2.IMREAD_COLOR)
        assert src.shape[0]==image_dict[name]['image_height'] and src.shape[1]==image_dict[name]['image_width'], f'picture {name}\'s size is not right'
        for l_bbox in image_dict[name]['l_bboxes']:
            bbox = l_bbox[1:]
            bbox[0] = int(bbox[0])
            bbox[1] = int(bbox[1])
            bbox[2] = round(bbox[2]+0.5)
            bbox[3] = round(bbox[3]+0.5)

            cls = int(l_bbox[0])
            color = colors[cls % len(colors)]
            label = str(cls)
            plot_one_box(bbox, src, label=label, color=color, line_thickness=3)
        save_path = os.path.join(img_save_dir, str(os.path.basename(image_file)))
        cv2.imwrite(save_path, src)

