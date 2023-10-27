# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================



from collections import OrderedDict
import numpy as np
from pathlib import Path
from time import time
from typing import Tuple
import math 
import flwr as fl
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import Tensor
from torchvision import datasets
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from torchvision.datasets import VOCDetection
import torchvision.models as models
from collections import defaultdict
from sklearn.metrics import average_precision_score
import json
import os
from PIL import Image


# Set the root directory of PASCAL VOC dataset
voc_root = "./data/VOCdevkit/VOC2007"  # Adjust to your path
farm_root = "./data/SmartFarm"  # Adjust to your path

DATA_ROOT = Path("./data")


VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
CLASSES = ["normal", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def transform_voc_annotation(annotation):
    boxes = []
    labels = []
    for obj in annotation['annotation']['object']:
        bbox = obj['bndbox']
        boxes.append([int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])])
        labels.append(VOC_CLASSES.index(obj['name']))
    return {'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': torch.tensor(labels, dtype=torch.int64)}

# Modify this function to extract bounding boxes and classes from the custom JSON format - custom annotation
def transform_smartfarm_annotation(annotation):
    boxes = []
    labels = []

    for obj in annotation['annotations']['object']:
        label = obj['class']

        # Check if label is outside the valid range
        if label > 20 or label < 0:
            continue  # Skip this object

        for points in obj['points']:
            boxes.append([points['xtl'], points['ytl'], points['xbr'], points['ybr']])
            labels.append(label)

    if not boxes or not labels:
        return None

    return {'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': torch.tensor(labels, dtype=torch.int64)}

# Dummy CustomDataset to load data from your JSON format
# You might need to further modify it to suit your directory structure and file naming
class SmartFarmDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder, transform=None):
        
        self.images_folder = os.path.join(root_folder, 'images')
        self.labels_folder = os.path.join(root_folder, 'labels')
        self.transform = transform

        # List all JSON files in the labels directory
        self.label_files = [f for f in os.listdir(self.labels_folder) if f.endswith('.json')]
    
    def __len__(self):
        return len(self.label_files)
    
    def __getitem__(self, idx):
        # Load the JSON file for this index
        with open(os.path.join(self.labels_folder, self.label_files[idx]), 'r') as f:
            item = json.load(f)
        
        # Fetch the corresponding image from the images directory
        image_path = os.path.join(self.images_folder, item['description']['image'])

        # If image_path doesn't exist, try with different extensions
        if not os.path.exists(image_path):
            base_path, _ = os.path.splitext(image_path)
            found = False
            for ext in ['jpg', 'jpeg', 'JPG']:
                new_path = f"{base_path}.{ext}"
                if os.path.exists(new_path):
                    image_path = new_path
                    found = True
                    break
            if not found:
                raise ValueError(f"Image not found for {image_path} even after trying different extensions.")
        
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        target = item # transform_smartfarm_annotation(item)
        return image, target



def load_model(model_name: str) -> nn.Module:
    if model_name == "mobilenet":
        model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        num_classes = 21  # 20 classes + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        return model
    elif model_name == "resnet":
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        num_classes = 21  # 20 classes + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        return model
    else:
        raise NotImplementedError(f"model {model_name} is not implemented")


# pylint: disable=unused-argument
def load_pascal(download=False) -> Tuple[datasets.CocoDetection, datasets.CocoDetection]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)),
        ]
    )
    voc_root = "./data/VOC2007/"
    # Training dataset
    trainset = VOCDetection(root=voc_root, year='2007', image_set='train', download=download, transform=transform)

    # Testing/validation dataset
    testset = VOCDetection(root=voc_root, year='2007', image_set='test', download=download, transform=transform)
    return trainset, testset

# Modify loading function to use CustomDataset
def load_smartfarm_data():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)),
        ]
    )
    dataset = SmartFarmDataset(farm_root, transform=transform)
    return dataset  # For simplicity, returning just one dataset, split into train/test as needed


def train(
    net: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    lr: float,
    epochs: int,
    optimizer_n: str,
    algorithm: str,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    """Train the network."""

    net.train()

    # Define loss and optimizer
    if optimizer_n == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    elif optimizer_n == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif optimizer_n == "RMSprop":
        optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)
    elif optimizer_n == "SGDM":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        raise NotImplementedError(f"optimizer {optimizer_n} is not implemented")

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    t = time()
    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (images, targets) in enumerate(trainloader, 0):
            targets_transformed = [transform_smartfarm_annotation(anno) for anno in targets]
        
            # Filter out None and the corresponding images
            valid_indices = [i for i, t in enumerate(targets_transformed) if t is not None]
            images = [images[i] for i in valid_indices]
            targets_transformed = [t for t in targets_transformed if t is not None]

            # Skip this batch if there are no valid samples left
            if not images or not targets_transformed:
                continue

            images = [img.to(device) for img in images]
            targets_transformed = [{k: v.to(device) for k, v in t.items()} for t in targets_transformed]



            loss_dict = net(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            losses.backward()
            optimizer.step()

            # print statistics
            running_loss += losses.item()
            if i % 50 == 49:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
                
    print(f"Epoch took: {time() - t:.2f} seconds")


def test(
    net: nn.Module,
    testset,
    testloader: torch.utils.data.DataLoader,
    device: torch.device
) -> float:
    """Validate the network on the entire test set."""
    # Placeholder for predictions and ground truths
    all_predictions = []
    all_gts = []

    # Get model predictions and ground truths
    net.eval()
    with torch.no_grad():
        for images, annotations in testloader:
            images = [img.to(device) for img in images]
            predictions = net(images)

            # Append to placeholder lists
            all_predictions.extend(predictions)
            all_gts.extend(annotations)

    # Calculate mAP
    aps = []

    # Assuming you have some predefined CLASSES constant
    for class_idx, class_name in enumerate(CLASSES):
        y_true = []
        y_scores = []
        for i in range(len(all_gts)):
            gt_boxes = [box for obj in all_gts[i]['annotations']['object'] if obj['class'] == class_idx for box in obj['points']]
            pred_boxes = all_predictions[i]['boxes'][all_predictions[i]['labels'] == class_idx].tolist()
            scores = all_predictions[i]['scores'][all_predictions[i]['labels'] == class_idx].tolist()

            for pb, score in zip(pred_boxes, scores):
                matched = False
                for gb in gt_boxes:
                    iou = compute_iou(pb, [gb['xtl'], gb['ytl'], gb['xbr'], gb['ybr']])
                    if iou > 0.5:
                        matched = True
                        break
                y_true.append(int(matched))
                y_scores.append(score)

        if y_scores:
            ap = average_precision_score(y_true, y_scores)
            aps.append(ap)

    mAP = sum(aps) / len(aps) if aps else 0  # Handling cases where aps might be empty
    return mAP

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


    
    
