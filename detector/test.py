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
from collections import defaultdict

VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def transform_voc_annotation(annotation):
    boxes = []
    labels = []
    for obj in annotation['annotation']['object']:
        bbox = obj['bndbox']
        boxes.append([int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])])
        labels.append(VOC_CLASSES.index(obj['name']))
    return {'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': torch.tensor(labels, dtype=torch.int64)}

model = fasterrcnn_mobilenet_v3_large_fpn()
device='cpu'
num_classes = 21  # 20 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)),
        ]
    )
voc_test = VOCDetection(root='./data', year='2007', image_set='train', download=True, transform=transform)

testloader = torch.utils.data.DataLoader(voc_test, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
print(len(voc_test))

model.train()
predictions = []
with torch.no_grad():
    for images, targets in testloader:
        images = [img.to(device) for img in images]
        targets = [transform_voc_annotation(anno) for anno in targets]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        print(losses)
        1/0
        
    

