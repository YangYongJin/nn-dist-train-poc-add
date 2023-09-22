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


# Set the root directory of PASCAL VOC dataset
voc_root = "./data/VOCdevkit/VOC2007"  # Adjust to your path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
DATA_ROOT = Path("./data")

__all__ = ['resnet']



def load_model(model_name: str) -> nn.Module:
    if model_name == "mobile":
        return fasterrcnn_mobilenet_v3_large_fpn()
    elif model_name == "resnet":
        model = fasterrcnn_resnet50_fpn_v2()
        num_classes = 21  # 20 classes + background
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        return fasterrcnn_resnet50_fpn_v2()
    else:
        raise NotImplementedError(f"model {model_name} is not implemented")


# pylint: disable=unused-argument
def load_pascal(download=False) -> Tuple[datasets.CocoDetection, datasets.CocoDetection]:
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)),
        ]
    )
    voc_root = "./data/VOC2007/"
    # Training dataset
    trainset = VOCDetection(root=voc_root, year='2007', image_set='train', transform=transform)

    # Testing/validation dataset
    testset = VOCDetection(root=voc_root, year='2007', image_set='test', transform=transform)
    return trainset, testset


def train(
    net: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    lr: float,
    epochs: int,
    optimizer_n: str,
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
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

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
    testset: datasets,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    net.eval()
    predictions = []
    with torch.no_grad():
        for images, targets in testloader:
            images = [img.to(device) for img in images]
            output = net(images)
            predictions.extend(output)
    
    
    return compute_mAP(testset, predictions)


def compute_mAP(gt, preds):
    APs = []
    for class_id in range(len(gt.classes) - 1):  # Exclude the background class
        detections = []
        ground_truths = []

        # Gather all detections and ground truths for this class
        for i, prediction in enumerate(preds):
            if len(prediction["labels"]) > 0:
                for p, label in zip(prediction["scores"], prediction["labels"]):
                    if label == class_id:
                        detections.append((i, p))
                
            gt_boxes = gt[i]['annotation']['bbox']
            for box in gt_boxes:
                if box['label'] == class_id:
                    ground_truths.append(i)

        # Sort detections by score
        detections.sort(key=lambda x: x[1], reverse=True)
        
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_gt = len(ground_truths)

        for i, detection in enumerate(detections):
            img_id, _ = detection
            if img_id in ground_truths:
                TP[i] = 1
                ground_truths.remove(img_id)
            else:
                FP[i] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_gt + 1e-10)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-10)
        recalls = torch.cat((torch.tensor([0]), recalls))
        precisions = torch.cat((torch.tensor([1]), precisions))
        
        # Compute average precision using the trapz rule
        AP = -torch.trapz(precisions, recalls)
        APs.append(AP)

    return torch.mean(torch.stack(APs))

    
    
