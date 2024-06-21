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
"""PyTorch CIFAR-10 image classification.

The code is generally adapted from 'PyTorch: A 60 Minute Blitz'. Further
explanations are given in the official PyTorch tutorial:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""


# mypy: ignore-errors
# pylint: disable=W0223
from torch import optim
import numpy as np
from collections import OrderedDict
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
from torchvision import models
import copy
import os
import time
#from utils import get_optimizer, get_model
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import init
import sys
import torch.optim as optim
from resnet import ResNet
DATA_ROOT = Path("./veri1")

__all__ = ['resnet']

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
        #print(self.classifier)
    def forward(self, x):
        x = self.add_block(x)
        
        if self.return_f:
            y = self.classifier(x)
            return x, y
        else:
            x = self.classifier(x)
            return x
          
# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net, self).__init__()
       
        model_ft = models.resnet18(pretrained=False)
        # model_ft=torch.load('saved_res50.pkl')
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(512, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x 




# pylint: disable=unsubscriptable-object
class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # pylint: disable=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)


def ResNet8():
    model = ResNet(8, num_classes=5)
    return model

def ResNet18():
    model = ft_net(class_num=5)
    return model

def ResNet50():
    model = ft_net(class_num=100)
    return model


def load_model(model_name: str) -> nn.Module:
    if model_name == "Net":
        return Net()
    elif model_name == "ResNet18":
        return ResNet18()
    elif model_name == "ResNet8":
        return ResNet8()
    elif model_name == "ResNet50":
        return ResNet50()
    else:
        raise NotImplementedError(f"model {model_name} is not implemented")

        
def load_cifar(download=False) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = datasets.CIFAR10(
        root=DATA_ROOT / "cifar-10", train=True, download=download, transform=transform
    )
    testset = datasets.CIFAR10(
        root=DATA_ROOT / "cifar-10", train=False, download=download, transform=transform
    )
    return trainset, testset


def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to('cuda:0')
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)


    logits = torch.gather(logits, 1, nt_positions)

    return logits

class NTD_Loss(nn.Module):
    def __init__(self, num_classes=10, tau=3, beta=1):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta

    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)
        ntd_loss = self._ntd_loss(logits, dg_logits, targets)

        loss = ce_loss + self.beta * ntd_loss

        return loss

    def _ntd_loss(self, logits, dg_logits, targets):
        logits = refine_as_not_true(logits, targets, self.num_classes)
        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)

        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed dg_model prediction
        with torch.no_grad():
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = self.KLDiv(pred_probs, dg_probs)

        return loss
def fliplr(img):
    """flip horizontal
    """
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        ff = torch.FloatTensor(n, 5).zero_().cuda()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            #for scale in ms:
                #if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    #input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
            outputs = model(input_img)
            ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features



def get_optimizer1(model, lr):
    ignored_params = list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.0*lr},
            {'params': model.classifier.parameters(), 'lr': lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    return optimizer_ft

# for ntd, pav
def get_optimizer2(model, lr):
    ignored_params = list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': lr},
            {'params': model.classifier.parameters(), 'lr': lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    return optimizer_ft

def train(
    model: torch.nn.Module,
    prev_model: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    dataset_sizes,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    sys.setrecursionlimit(10000)
    y_err = []
    y_loss = []
    #self.model.load_state_dict(federated_model.state_dict())
    pro1_model = copy.deepcopy(model)
    global_model = copy.deepcopy(model)
    #model.classifier.classifier = model.classifier
    #old_classifier = copy.deepcopy(model.classifier)
    #print(model)
    
    #optimizer = get_optimizer(model, lr=0.01)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()

    # MOON hyperparameters
    cos = nn.CosineSimilarity(dim=-1)
    mu = 5              # 0 / 0.1 / 1 / 5 / 10
    temperature = 0.5   # 0.1 / 0.5 / 1.0
    lr=0.01
    batch_size=10
    #train_criterion = NTD_Loss(5, 3, 1)
    #global_model=copy.deepcopy(model)
    #since = time.time()
    print( 'start training')
    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    #t = time()
    for epoch in range(epochs):
        if epoch % 3 == 0:
            optimizer = get_optimizer1(model, lr)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        else:
            optimizer = get_optimizer2(model, lr)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        scheduler.step()
        #model.train(True)
        running_loss = 0.0
        running_corrects = 0.0
        for data in trainloader:
            inputs, labels = data
            b, c, h, w = inputs.shape
            if b < batch_size:
                continue
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda().detach())
                labels = Variable(labels.cuda().detach())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
                
            optimizer.zero_grad()

            """
                - pro1: representation from local model [b, c]
                - pro2: representation from global model [b, c]
                - pro3: representation from previous local model [b, c]

                - loss1: cross-entropy loss
                - loss2: contrastive loss
            """
            pro1 = pro1_model(inputs)
            p_logits = model(inputs)
                #pro2, _ = global_model(inputs)
            pro2 = global_model(inputs)
            _, preds = torch.max(p_logits.data, 1)

            posi = cos(pro1, pro2)          # positive pairs [b]
                #logits = posi.reshape(-1, 1)    # [b, 1]

            pro3 = prev_model(inputs)
            nega = cos(pro1, pro3)
            b_logits = torch.cat((posi.reshape(-1,1), nega.reshape(-1,1)), dim=1)

                # prev_model.to('cpu')
            b_logits /= temperature
            b_labels = torch.zeros(inputs.size(0)).cuda().long()

            b_loss = mu * criterion(b_logits, b_labels)


            ce_loss=criterion(p_logits, labels)
            loss = ce_loss + b_loss





            #pro1, outputs = model(inputs)
            #pro2, _ = global_model(inputs)
            #_, preds = torch.max(outputs.data, 1)

            #posi = cos(pro1, pro2)          # positive pairs [b]
            #logits = posi.reshape(-1, 1)    # [b, 1]

            #pro3, _ = prev_model(inputs)
            #nega = cos(pro1, pro3)
            #logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

            # prev_model.to('cpu')

            #logits /= temperature
            #labels_new = torch.zeros(inputs.size(0)).cuda().long()

            #loss1 = criterion(outputs, labels)
            #loss2 = mu * criterion(logits, labels_new)
            #loss = loss1 + loss2


            #outputs = self.model(inputs)
            # outputs, t_logits = model(inputs), dg_model(inputs)
            # _, preds = torch.max(outputs.data, 1)
            # loss = criterion(outputs, labels)
            #loss = train_criterion(outputs,labels,t_logits)
            loss.backward()

            optimizer.step()

            running_loss += loss.item() * b
            running_corrects += float(torch.sum(preds == labels.data))

        used_data_sizes = (dataset_sizes - dataset_sizes % batch_size)
        epoch_loss = running_loss / used_data_sizes
        epoch_acc = running_corrects / used_data_sizes

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

        y_loss.append(epoch_loss)
        y_err.append(1.0-epoch_acc)
        
    #print(f"Epoch took: {time() - t:.2f} seconds")
    '''
        time_elapsed = time.time() - since
        print('Client', self.cid, ' Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    time_elapsed = time.time() - since
    print('Client', self.cid, 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    '''
        # save_network(self.model, self.cid, 'last', self.project_dir, self.model_name, gpu_ids)
        
    #self.classifier = self.model.classifier.classifier
    #self.distance = self.optimization.cdw_feature_distance(federated_model, self.old_classifier, self.model)
    #self.model.classifier.classifier = nn.Sequential()

def fliplr(img):
    """flip horizontal
    """
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # good index
    gl= np.array(gl)
    ql= np.dtype('int64').type(ql)
    query_index = np.argwhere(gl==ql)
    gc=np.array(gc)
    qc=np.dtype('int64').type(qc)
    camera_index = np.argwhere(gc==qc)
    #good_index = index
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)

    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc
    

def test(
    model: torch.nn.Module,
    testloader,
    gallery_meta,
    query_meta,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    
    print("="*10)
    print("Start Testing!")
    print("="*10)
    #print('We use the scale: %s'%self.multiple_scale)

    with torch.no_grad():
        gallery_feature = extract_feature(model, testloader['gallery'])
        query_feature = extract_feature(model, testloader['query'])

    result = {'gallery_f': gallery_feature.numpy(),
              'gallery_label': gallery_meta['labels'],
              'gallery_cam': gallery_meta['cameras'],
              'query_f': query_feature.numpy(),
              'query_label': query_meta['labels'],
              'query_cam': query_meta['cameras']}
    print(type(result))
    #scipy.io.savemat(os.path.join('.','ResNet18','pytorch_result.mat'),result)
    #os.system('python evaluate.py --result_dir {}'.format(os.path.join('.', 'ResNet18'))) 
    query_feature = torch.FloatTensor(result['query_f'])
    query_cam = result['query_cam']
    query_label = result['query_label']
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_cam = result['gallery_cam']
    gallery_label = result['gallery_label']

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    #print(query_feature.shape)
    #print(gallery_label)
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    #print('Rank@286:%f mAP:%f'%(CMC[286], ap/len(query_label)))
    print(' Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[1], CMC[5], CMC[10], ap/len(query_label)))
    print('-'*15)
    print()

    #print(self.model_name)
    #print(dataset):

        
