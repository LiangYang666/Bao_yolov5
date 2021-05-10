import argparse
import time
from pathlib import Path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, MyLoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from torch.utils.data import DataLoader

from utils.plots import plot_images, output_to_target
from tqdm import tqdm
from zizhuantools.tool3_labelresp import class_names


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = False
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #     ('rtsp://', 'rtmp://', 'http://'))

    source = os.path.join(opt.data_dir, source)

    # Directories
    save_name_add = os.path.basename(str(weights)).split('.')[0]  # 名称中要增加保存的字符
    # print(Path(opt.data_dir) / 'runs' / 'test' /(opt.name+f'({save_name_add})'))
    save_dir = Path(increment_path(Path(opt.data_dir) / 'runs' / 'dect' / (opt.name + f'_{save_name_add}_'),
                                   exist_ok=opt.exist_ok))  # increment run

    # save_dir = Path(increment_path(Path(opt.data_dir) / Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    weights = os.path.join(opt.data_dir, weights)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        # dataset = LoadImages(source, img_size=imgsz)
        dataset = MyLoadImages(source, img_size=imgsz)
        dataset = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False,
                                    num_workers=16,
                                    pin_memory=True)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    dectjson_list = []
    add_name = os.path.basename(weights).split('.')[0]
    save_json_path = str(save_dir / Path(f'dect{add_name}.json'))
    print('save_dir:', save_dir)

    for batch_i, (path, img, orgin_hw) in enumerate(tqdm(dataset)):
        # if batch_i < 32: continue
        # print('**************type', type(orgin_hw[0, :]))
        # print(len(orgin_hw))
        # print(orgin_hw)
        # img = torch.from_numpy(img).to(device)
        # print('**************img shape', img.shape) # torch.Size([8, 3, 640, 640])

        img = img.to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, model, img, orgin_hwshape)

        # Process detections
        # print('pred len', len(pred))
        # print(type(pred))

        for i, det in enumerate(pred):  # detections per image
            det = det.clone()  # 防止将pred改变
            # if webcam:  # batch_size >= 1
            #     p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
            # else:
            p, s, hwshape = Path(path[i]), '', orgin_hw[i]
            save_path = str(save_dir / p.name)
            # txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(hwshape)[[1, 0, 1, 0]]  # normalization gain whwh
            gn = hwshape[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # assert orgin_hwshape.ndim 
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], hwshape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                save_each_cut = False
                if save_each_cut:
                    img_orgin = cv2.imread(str(p))

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # conf : tensor(0.52002, device='cuda:0')
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_json_path:
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                        box = torch.tensor(xyxy).view(-1).tolist()
                        import numpy as np
                        x0, y0, x1, y1 = [int(x) for x in box[:2]] + [int(np.ceil(x)) for x in box[2:]]
                        category = int(cls.cpu().view(-1).tolist()[0])
                        confidence = round(conf.cpu().view(-1).tolist()[0], 2)
                        # print('conf', confidence)
                        # print(type(confidence))
                        dict = {'cut_name': p.name, 'cut_category': category,
                                'confidence': confidence, 'bbox': [x0, y0, x1, y1]}
                        dectjson_list.append(dict)
                        if save_each_cut:
                            cv2.rectangle(img_orgin, (x0, y0), (x1, y1), (255, 0, 0), thickness=1)
                            if 'by' in weights:
                                dis_name = class_names[category]
                            elif 'zx' in weights:
                                dis_name = class_names[category+2]
                            cv2.putText(img_orgin, f'{dis_name}  {confidence}', (x0, y0-2), cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=0.7, color=(0, 255, 255),
                                        thickness=2)
                if save_each_cut:
                    cv2.imwrite(save_path, img_orgin)
        import json
        if batch_i % 5 == 0:
            with open(save_json_path, 'w') as f:
                json.dump(dectjson_list, f, indent=2)

                # if save_img or view_img:  # Add bbox to image
                #     label = '%s %.2f' % (names[int(cls)], conf)
                #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        # Plot images
        plots = False
        if plots:
            # f = save_dir / f'test_batch{batch_i}_labels.jpg'  # filename
            # plot_images(img, targets, paths, f, names)  # labels
            f = save_dir / f'test_batch{batch_i}_.jpg'
            image = plot_images(img, output_to_target(pred), path, f, names)  # predictions
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("image", 1000, 1000)
            cv2.imshow('image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('-w', '--weights', type=str, default='runs/train/exp(allzx0)(OK 600 yolov5s)/weights/ckp_allzx0_4c_epoch600.pt', help='model.pt path')
    # parser.add_argument('--source', type=str, default='tile_round1_testA_20201231/testA_imgs_zxcutsave',
    #                     help='source')  # file/folder, 0 for webcam
    parser.add_argument('-s', '--source', type=str, default='tile_round1_testA_20201231/testA_imgs',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0,1,2,3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('-dir', '--data-dir', dest='data_dir', type=str, default='../data/CS1', help='dataset dir')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='size of each image batch')
    # parser.add_argument('-test', '-test_txtl_file', dest='test_txt',
    #                     type=str, default='testby0.txt', help="test txt file")
    opt = parser.parse_args()
    print(opt)
    # python detect_multi.py -b 128 --device 0,1,2,3 -w runs/train/exp(allzx0)(OK 600 yolov5s)/weights/ckp_allzx0_4c_epoch600.pt
    # python detect_multi.py -b 128 --device 0,1,2,3 -s RaceZxTestCAM1.txt -w runs/train/exp_CAM1allzx0_/weights/ckp_CAM1allzx0_4c_epoch600.pt

    with torch.no_grad():
        # if opt.update:  # update all models (to fix SourceChangeWarning)
        #     for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
        #         detect()
        #         strip_optimizer(opt.weights)
        # else:

        detect()
