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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
DATA_ROOT = Path("./data")

__all__ = ['resnet']



def load_model(model_name: str) -> nn.Module:
    if model_name == "mobile":
        return fasterrcnn_mobilenet_v3_large_fpn()
    elif model_name == "resnet":
        return fasterrcnn_resnet50_fpn_v2()
    else:
        raise NotImplementedError(f"model {model_name} is not implemented")


# pylint: disable=unused-argument
def load_coco(download=False) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)),
        ]
    )
    # Training dataset
    trainset = CocoDetection(root="./data/COCO/train2017",
                            annFile="./data/COCO/annotations/instances_train2017.json",
                            transform=transform)

    # Testing/validation dataset
    testset = CocoDetection(root="./data/COCO/val2017",
                            annFile="./data/COCO/annotations/instances_val2017.json",
                            transform=transform)
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
    results = []

    with torch.no_grad():  # Disables gradient computation
        for images, _ in testloader:  # We don't need ground truth for prediction
            images = list(img.to(device) for img in images)
            predictions = net(images)
            results.extend(predictions)
    
    coco_results = []
    for image_id, prediction in enumerate(results):
        image_info = testset.coco.imgs[testset.ids[image_id]]
        for bbox, score, label in zip(prediction["boxes"], prediction["scores"], prediction["labels"]):
            x, y, width, height = bbox
            result = {
                "image_id": image_info["id"],
                "category_id": label.item(),
                "bbox": [x.item(), y.item(), (width - x).item(), (height - y).item()],
                "score": score.item()
            }
            coco_results.append(result)


    coco_gt = COCO("./data/COCO/annotations/instances_val2017.json")
    coco_dt = coco_gt.loadRes(coco_results)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    # This is the line that extracts mAP@0.5 from the COCOeval object
    mAP_at_05 = coco_eval.stats[1]
    
    return mAP_at_05

    
    
