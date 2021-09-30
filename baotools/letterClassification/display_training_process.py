#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :display_training_process.py
# @Time      :2021/9/24 上午10:57
# @Author    :Yangliang

import argparse
import os
import re
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
# from visualdl import LogWriter

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Letter Training')
    parser.add_argument('--brand', type=str, default='LV')
    parser.add_argument('--part', type=str, default='sign')
    parser.add_argument('--letter', type=str, default='A')
    args = parser.parse_args()
    log_txt_dir = f'/media/D_4TB/YL_4TB/BaoDetection/data/{args.brand}/LetterDetection/data/{args.part}/classification_rundata/{args.letter}/log'

    log_txt_files = sorted(os.listdir(log_txt_dir))
    log_txt_files = [x for x in log_txt_files if x.endswith('.txt')]

    for path in tqdm(log_txt_files):
        log_txt_path = os.path.join(log_txt_dir, path)
        with open(log_txt_path, 'r') as f:
            lines = f.readlines()
        model = None
        train_record = []
        for line in lines:
            line = line.strip()
            if model is None:
                model = path
                # if "'model': " in line:
                #     model = line.split("'model': ")[1].strip().strip("'")
                #     print(model)
            else:
                if "INFO: Epoch:" in line:
                    # searchObj = re.search(r'(.*) are (.*?) .*', line, re.M | re.I)
                    # line 为  "2021-09-07 15:51:42,301 letter_train.py[line:229] INFO: Epoch:0 Train accuracy: 48.68%(37/76)	true:100.00%(37/37)	fake:0.00%(0/39)	Test accuracy: 45.16%(14/31)	true:100.00%(14/14)	fake:0.00%(0/17)"
                    pattern = re.compile(r'.*INFO: Epoch:(\d*)\s*Train accuracy:\s*([\d\\.]*)%.*true:\s*([\d\\.]*)%.*fake:\s*([\d\\.]*)%.*Test accuracy:\s*([\d\\.]*)%.*true:\s*([\d\\.]*)%.*fake:\s*([\d\\.]*)%.*')
                    groups = pattern.match(line).groups()
                    train_record.append(groups)

        train_log_dir = log_txt_path.rsplit('.', 1)[0]

        with SummaryWriter(log_dir=train_log_dir) as writer:
            for record in train_record:
                step = int(record[0])
                rate = [float(x)/100 for x in record[1:]]
                writer.add_scalar(tag="train/acc",  global_step=step, scalar_value=rate[0])
                writer.add_scalar(tag="train/true", global_step=step, scalar_value=rate[1])
                writer.add_scalar(tag="train/fake", global_step=step, scalar_value=rate[2])
                writer.add_scalar(tag="test/acc",   global_step=step, scalar_value=rate[3])
                writer.add_scalar(tag="test/true",  global_step=step, scalar_value=rate[4])
                writer.add_scalar(tag="test/fake",  global_step=step, scalar_value=rate[5])

