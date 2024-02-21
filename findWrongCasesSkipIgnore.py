import argparse

import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
import time
import cv2
import torch
import glob
import json
import mmcv
import numpy as np
import clip
import subprocess
from PIL import Image

from mmdet.apis import inference_detector, init_detector, show_result


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_img_dir', type=str, help='the dir of input images')
    parser.add_argument('output_dir', type=str, help='the dir for result images')
    parser.add_argument('gt_path', type=str, help='gt path')
    parser.add_argument('threshold_IoU', type=float, help='threshold of IoU')
    parser.add_argument('clip_model', type=str, help='clip model name')
    parser.add_argument('filter_threshold', type=float, help='threshold of clip model')
    parser.add_argument('clip_padding', type=int, help='padding of clip model')
    parser.add_argument('show_tp_gt', type=bool, help='show true positive ground truth')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mean_teacher', action='store_true', help='test the mean teacher pth')
    args = parser.parse_args()
    return args

def get_gt_bboxes(gt_path, image_name, is_ignore=False):
    gt_boxes = []
    with open(gt_path) as f:
        gt_data = json.load(f)
        anno_size = len(gt_data["annotations"])
        if anno_size == 0:
            return []
        for image_info in gt_data["images"]:
            if image_info["im_name"] == image_name:
                image_id = image_info["id"]
                l = 0
                r = anno_size - 1
                while l < r:
                    mid = (l + r) // 2
                    if gt_data["annotations"][mid]["image_id"] >= image_id:
                        r = mid
                    else:
                        l = mid+1
                assert 0 <= l and l < anno_size
                if gt_data["annotations"][l]["image_id"] != image_id:
                    return []
                while l < anno_size and gt_data["annotations"][l]["image_id"] == image_id:
                    if gt_data["annotations"][l]["ignore"] == is_ignore:
                        gt_boxes.append(gt_data["annotations"][l]["bbox"])
                    # gt_boxes.append(gt_data["annotations"][l]["bbox"])
                    l += 1
                break
    return gt_boxes

def draw_gt_bboxes(gt_path, image_name, image_path, output_dir):
    gt_boxes = get_gt_bboxes(gt_path, image_name)
    gt_ignore_boxes = get_gt_bboxes(gt_path, image_name, True)
    image = cv2.imread(image_path)
    for gt_box in gt_boxes:
        x, y, w, h = gt_box
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 1)
    for gt_ignore_box in gt_ignore_boxes:
        x, y, w, h = gt_ignore_box
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 1)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, image_name), image)

def get_detector_bboxes(model, image_path, score_thr=0.3):
    image = cv2.imread(image_path)
    results = inference_detector(model, image)
    if isinstance(results, tuple):
        bbox_result = results[0]
    else:
        bbox_result = results
    bboxes_with_scores = np.vstack(bbox_result)
    bboxes = []
    for bbox_with_score in bboxes_with_scores:
        if bbox_with_score[4] > score_thr:
            bboxes.append(bbox_with_score[:4])
    for i in range(len(bboxes)):
        bboxes[i][2] -= bboxes[i][0]
        bboxes[i][3] -= bboxes[i][1]
    return bboxes

def filter_gt_bboxes(model_name, image_path, bboxes, padding = 5, threshold=0.5):
    if (model_name == "None"):
        return bboxes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)

    filtered_bboxes = []
    probs_list = []
    crop_image_list = []
    # get width and height of the image
    width, height = Image.open(image_path).size
    labels = [[1, 'a picture of people walking'], 
              [0, 'a picture of people riding a bike'], 
              [0, 'a picture of people on a vehicle']]
    for bbox in bboxes:
        x, y, w, h = bbox
        x = max(0, x-padding)
        y = max(0, y-padding)
        w = min(w+2*padding, width-x)
        h = min(h+2*padding, height-y)
        crop_image = Image.open(image_path).crop((x, y, x+w, y+h))
        crop_image_list.append(crop_image)
        image = preprocess(Image.open(image_path).crop((x, y, x+w, y+h))).unsqueeze(0).to(device)
        labels_text = []
        for label in labels:
            labels_text.append(label[1])
        text = clip.tokenize(labels_text).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # find max prob
        max_prob_id = np.argmax(probs[0])
        if probs[0][max_prob_id] >= threshold:
            if labels[max_prob_id][0] == 0:
                pass
            else:
                filtered_bboxes.append(bbox)
        else:
            filtered_bboxes.append(bbox)
        probs_list.append(probs)
    return filtered_bboxes, crop_image_list, probs_list

def show_vis_ratio_list(gt_path):
    gt_data = json.load(open(gt_path))
    vis_ratio_list = set()
    for anno in gt_data["annotations"]:
        vis_ratio_list.add(anno["vis_ratio"])
    vis_ratio_list = list(vis_ratio_list)
    vis_ratio_list.sort()
    print(vis_ratio_list)

def create_base_dir(dest):
    basedir = os.path.dirname(dest)
    if not os.path.exists(basedir):
        os.makedirs(basedir)

def get_IoU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x = max(x1, x2)
    y = max(y1, y2)
    w = min(x1+w1, x2+w2) - x
    h = min(y1+h1, y2+h2) - y
    if w <= 0 or h <= 0:
        return 0
    return w*h/(w1*h1+w2*h2-w*h)

def run_detector_on_dataset():
    args = parse_args()
    input_dir = args.input_img_dir
    output_dir = args.output_dir
    gt_path = args.gt_path
    threshold_IoU = args.threshold_IoU
    clip_model = args.clip_model
    filter_threshold = args.filter_threshold
    clip_padding = args.clip_padding
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(input_dir)
    eval_imgs = glob.glob(os.path.join(input_dir, '*.png'))

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))
    
    print("Start processing...")
    print("Total number of images: ", len(eval_imgs))

    # prog_bar = mmcv.ProgressBar(len(eval_imgs))
    cnt = 0
    for im in eval_imgs:
        cnt += 1
        print("------------------")
        print(f"Processing ({cnt}/{len(eval_imgs)}) {im} :")
        detection_bbox = get_detector_bboxes(model, im)
        print("Detected bbox: ", len(detection_bbox))
        print("Filtering...")
        old_detection_bbox = detection_bbox
        detection_bbox, crop_image_list, probs_list = filter_gt_bboxes(clip_model, im, detection_bbox, clip_padding, filter_threshold)
        print("Filtered bbox: ", len(detection_bbox))
        gt_bboxes = get_gt_bboxes(gt_path, os.path.basename(im), False)
        # gt_ignore_boxes = get_gt_bboxes(gt_path, os.path.basename(im), True)
        print(f"Detected: {len(detection_bbox)}")
        print(f"Ground truth: {len(gt_bboxes)}")
        # print(f"Ignore: {len(gt_ignore_boxes)}")
        edge_list = []
        for i in range(len(detection_bbox)):
            for j in range(len(gt_bboxes)):
                # x1, y1, w1, h1 = detection_bbox[i]
                # x2, y2, w2, h2 = gt_bboxes[j]
                edge_list.append((i, j, get_IoU(detection_bbox[i], gt_bboxes[j])))
        used_detection = set()
        used_gt = set()
        used_edge_id = set()
        edge_list.sort(key=lambda x: x[2], reverse=True)
        for i in range(len(edge_list)):
            edge = edge_list[i]
            u, v, IoU = edge
            if u in used_detection or v in used_gt or IoU < threshold_IoU:
                continue
            used_detection.add(u)
            used_gt.add(v)
            used_edge_id.add(i)
        is_correct_case = True
        for i in range(len(gt_bboxes)):
            if i not in used_gt:
                is_correct_case = False
                break
        for i in range(len(detection_bbox)):
            if i not in used_detection:
                is_correct_case = False
                break
        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        BLUE = (255, 0, 0)
        YELLOW = (0, 255, 255)
        if not is_correct_case:
            print("Wrong case!")
            image = cv2.imread(im)
            # Draw the ground truth
            for i in range(len(gt_bboxes)):
                x, y, w, h = gt_bboxes[i]
                if i not in used_gt: # False negative
                    cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), YELLOW, 1)
                else: # True positive
                    if args.show_tp_gt:
                        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), BLUE, 1)
            # Draw the detection
            for i in range(len(detection_bbox)):
                x, y, w, h = detection_bbox[i]
                if i in used_detection: # True positive
                    cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), GREEN, 1)
                else: # False positive
                    cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), RED, 1)
            # Output the image
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(os.path.join(output_dir, "images")):
                os.makedirs(os.path.join(output_dir, "images"))
            cv2.imwrite(os.path.join(output_dir, "images", os.path.basename(im)), image)
            # Output log file
            ## Create log folder
            if not os.path.exists(os.path.join(output_dir, "log")):
                os.makedirs(os.path.join(output_dir, "log"))
            ## Output log file
            with open(os.path.join(output_dir, "log", os.path.basename(im).split('.')[0]+".txt"), "w") as f:
                f.write(f"Detected: {len(detection_bbox)}\n")
                for i in range(len(detection_bbox)):
                    x, y, w, h = detection_bbox[i]
                    f.write(f"{i} {x} {y} {w} {h}\n")
                f.write("\n")
                f.write(f"Ground truth: {len(gt_bboxes)}\n")
                for i in range(len(gt_bboxes)):
                    x, y, w, h = gt_bboxes[i]
                    f.write(f"{i} {x} {y} {w} {h}\n")
                f.write("\n")
                f.write(f"Egde list: {len(edge_list)}\n")
                for i in range(len(edge_list)):
                    u, v, IoU = edge_list[i]
                    f.write(f"{u} {v} {IoU}\n")
                f.write("\n")
                f.write(f"True positive: {len(used_edge_id)}\n")
                for i in used_edge_id:
                    u, v, IoU = edge_list[i]
                    f.write(f"{detection_bbox[u]} {gt_bboxes[v]} {IoU}\n")
                f.write(f"False positive: {len(detection_bbox) - len(used_detection)}\n")
                for i in range(len(detection_bbox)):
                    if i not in used_detection:
                        f.write(f"{i} {detection_bbox[i]}\n")
                f.write(f"False negative: {len(gt_bboxes) - len(used_gt)}\n")
                for i in range(len(gt_bboxes)):
                    if i not in used_gt:
                        f.write(f"{i} {gt_bboxes[i]}\n")
                f.close()
            assert len(old_detection_bbox) == len(probs_list)
            assert len(old_detection_bbox) == len(crop_image_list) 
            ## Output log_probs file
            if not os.path.exists(os.path.join(output_dir, "log_probs")):
                os.makedirs(os.path.join(output_dir, "log_probs"))
            with open(os.path.join(output_dir, "log_probs", os.path.basename(im).split('.')[0]+"_probs.txt"), "w") as f:
                for i in range(len(old_detection_bbox)):
                    x, y, w, h = old_detection_bbox[i]
                    f.write(f"{i} {x} {y} {w} {h}: {probs_list[i]}\n")
                f.close()
            ## Output crop images
            if not os.path.exists(os.path.join(output_dir, "crop_images")):
                os.makedirs(os.path.join(output_dir, "crop_images"))
            if not os.path.exists(os.path.join(output_dir, "crop_images", os.path.basename(im).split('.')[0])):
                os.makedirs(os.path.join(output_dir, "crop_images", os.path.basename(im).split('.')[0]))
            for i in range(len(crop_image_list)):
                crop_image_list[i].save(os.path.join(output_dir, "crop_images", os.path.basename(im).split('.')[0], f"{i}.png"))
        else:
            print("Correct case!")
        # prog_bar.update()

if __name__ == '__main__':
    run_detector_on_dataset()