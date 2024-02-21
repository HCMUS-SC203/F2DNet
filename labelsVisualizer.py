import argparse

import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
# import time
# import cv2
# import torch
# import glob
# import json
# import mmcv
# import numpy as np
# import clip
# import subprocess
# from PIL import Image

# from mmdet.apis import inference_detector, init_detector, show_result

def parse_args():
    parser = argparse.ArgumentParser(description='Label Visualizer')
    parser.add_argument('input_img_dir', type=str, help='the dir of input images')
    parser.add_argument('output_dir', type=str, help='the dir for result images')
    parser.add_argument('gt_dir', type=str, help='gt path')
    parser.add_argument('gt_suffix', type=str, help='suffix of the gt file')
    args = parser.parse_args()
    return args

def visualize_labels():
    args = parse_args()
    input_img_dir = args.input_img_dir
    output_dir = args.output_dir
    gt_dir = args.gt_dir
    gt_suffix = args.gt_suffix
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    img_list = os.listdir(input_img_dir)
    for img in img_list:
        img_path = osp.join(input_img_dir, img)
        img_name = img.split('.')[0]
        # split the img_name using '_' and get everything except the last element, then join them together using '_' again
        if gt_suffix == '':
            img_name_comp = img_name.split('_')
        else:
            img_name_comp = img_name.split('_')[:-1]
            img_name_comp.append(gt_suffix)
        img_name = '_'.join(img_name_comp)
        gt_path = osp.join(gt_dir, img_name + '.json')
        print(gt_path)
        # with open(gt_path, 'r') as f:
        #     gt = json.load(f)
        # img = cv2.imread(img_path)
        # for box in gt:
        #     x1, y1, x2, y2 = box['bbox']
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(img, box['category'], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.imwrite(osp.join(output_dir, img), img)

if __name__ == '__main__':
    args = parse_args()