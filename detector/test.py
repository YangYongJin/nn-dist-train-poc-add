from collections import OrderedDict
import numpy as np
from pathlib import Path
from time import time
from typing import Tuple
import math 
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import Tensor
from torchvision import datasets
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,fasterrcnn_mobilenet_v3_large_fpn
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torchvision.datasets import VOCDetection
import torchvision.models as models
from sklearn.metrics import average_precision_score
from collections import defaultdict

VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def compute_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute area of intersection
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute area of both boxes
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute IoU
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou


def transform_voc_annotation(annotation):
    boxes = []
    labels = []
    for obj in annotation['annotation']['object']:
        bbox = obj['bndbox']
        boxes.append([int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])])
        labels.append(VOC_CLASSES.index(obj['name']))
    return {'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': torch.tensor(labels, dtype=torch.int64)}

device = torch.device("mps")
model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
num_classes = 21  # 20 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model = model.to(device)
transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
voc_test = VOCDetection(root='./data', year='2007', image_set='train', download=True, transform=transform)

testloader = torch.utils.data.DataLoader(voc_test, batch_size=32, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
print(len(voc_test))

# model.train()
# predictions = []
# with torch.no_grad():
#     for images, targets in testloader:
#         images = [img.to(device) for img in images]
#         targets = [transform_voc_annotation(anno) for anno in targets]
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         loss_dict = model(images, targets)
#         print(loss_dict)
#         losses = sum(loss for loss in loss_dict.values())
#         print(losses)


all_predictions = []
all_gts = []

# Get model predictions and ground truths
model.eval()
with torch.no_grad():
    for i, (images, annotations) in enumerate(testloader):
        print(i)
        images = [img.to(device) for img in images]
        predictions = model(images)

        # Append to placeholder lists
        all_predictions.extend(predictions)
        all_gts.extend(annotations)

        if i > 3:
            break
        

# Calculate mAP
aps = []

for class_idx, class_name in enumerate(VOC_CLASSES[1:]):  # Starting from 1 to skip background class
    print(class_name)
    y_true = []
    y_scores = []
    for i in range(len(all_gts)):
        gt_boxes = [box['bndbox'] for box in all_gts[i]['annotation']['object'] if VOC_CLASSES.index(box['name']) == class_idx]
        pred_boxes = all_predictions[i]['boxes'][all_predictions[i]['labels'] == class_idx].tolist()
        scores = all_predictions[i]['scores'][all_predictions[i]['labels'] == class_idx].tolist()

        for pb, score in zip(pred_boxes, scores):
            matched = False
            for gb in gt_boxes:
                iou = compute_iou(pb, [int(gb['xmin']), int(gb['ymin']), int(gb['xmax']), int(gb['ymax'])])
                if iou > 0.2:
                    matched = True
                    break
            y_true.append(int(matched))
            y_scores.append(score)

    if y_scores:
        ap = average_precision_score(y_true, y_scores)
        aps.append(ap)

mAP = sum(aps) / len(aps)
print(mAP)        
    

