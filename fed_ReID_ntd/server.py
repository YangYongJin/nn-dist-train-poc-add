import os
import math
import json
import matplotlib.pyplot as plt
from utils import get_model, extract_feature
import torch.nn as nn
import torch
import scipy.io
import copy
from data_utils import ImageDataset
import random
import torch.optim as optim
from torchvision import datasets
import sys
# def add_model(dst_model, src_model, dst_no_data, src_no_data): #다른 디바이스에 있는 것을 하는것은 jetson nano에서 어려울듯 싶어 fedml로 대체!!
#     if dst_model is None:
#         result = copy.deepcopy(src_model)
#         return result
#     params1 = src_model.named_parameters()
#     params2 = dst_model.named_parameters()
#     dict_params2 = dict(params2)
#     with torch.no_grad():
#         for name1, param1 in params1:
#             if name1 in dict_params2:
#                 dict_params2[name1].set_(param1.data*src_no_data + dict_params2[name1].data*dst_no_data)
#     return dst_model

# def scale_model(model, scale):
#     params = model.named_parameters()
#     dict_params = dict(params)
#     with torch.no_grad():
#         for name, param in dict_params.items():
#             dict_params[name].set_(dict_params[name].data * scale)
#     return model

def aggregate_models(models, weights):
    """aggregate models based on weights
    params:
        models: model updates from clients
        weights: weights for each model, e.g. by data sizes or cosine distance of features
    """
    training_num=sum(weights)
    averaged_params=models[0]
    for k in averaged_params.keys():
        for i in range(0, len(models)):
            w=weights[i]/training_num
            if i==0:
                averaged_params[k]=models[i][k]*w
            else:
                averaged_params[k]+=models[i][k]*w
    return averaged_params


class Server():
    def __init__(self, clients, data, device, project_dir, model_name, num_of_clients, lr, drop_rate, stride, multiple_scale):
        self.project_dir = project_dir
        self.data = data
        self.device = device
        self.model_name = model_name
        self.clients = clients
        self.client_list = self.data.client_list
        self.num_of_clients = num_of_clients
        self.lr = lr
        self.multiple_scale = multiple_scale
        self.drop_rate = drop_rate
        self.stride = stride

        self.multiple_scale = []
        for s in multiple_scale.split(','):
            self.multiple_scale.append(math.sqrt(float(s)))

        self.full_model = get_model(750, drop_rate, stride) #750 들어가도 상관없음, 이것은 class 갯수인데 server에서는 도려낼거라 의미없음!!               
        self.full_model.classifier.classifier = nn.Sequential()
        self.federated_model_0=self.full_model.to(self.device) #feature extractor만 남긴다!!
        self.federated_model=copy.deepcopy(self.federated_model_0).state_dict()
        self.federated_model_0.eval()
        self.train_loss = [] #매 round마다 중앙 서버 모델 performance 기재


    def train(self, epoch, cdw, use_cuda):
        if (epoch+1)%5==0:
            self.lr*=0.99
        models = []
        loss = []
#         cos_distance_weights = [] #pure fedpav 기준 필요 없을 것 같음!! 
        data_sizes = []
        current_client_list = random.sample(self.client_list, self.num_of_clients) #cross-device도 가능은 함!!
        for i in current_client_list:
            self.clients[i].train(self.federated_model, use_cuda, self.lr)
#             cos_distance_weights.append(self.clients[i].get_cos_distance_weight()) #pure fedpav 기준 필요 없을 것 같음!! 
            loss.append(self.clients[i].get_train_loss())
            models.append(self.clients[i].get_model())
            data_sizes.append(self.clients[i].get_data_sizes())

#         if epoch==0:
#             self.L0 = torch.Tensor(loss) #L0은 언급되지 않아서 주석처리 해도 될듯!!

        avg_loss = sum(loss) / self.num_of_clients

        print("==============================")
        print("number of clients used:", len(models))
        print('Train Epoch: {}, AVG Train Loss among clients of lost epoch: {:.6f}'.format(epoch, avg_loss))
        
        self.train_loss.append(avg_loss)
        
        weights = data_sizes
        
#         if cdw: #pure fedpav 기준 필요 없을 것 같음!! 
#             print("cos distance weights:", cos_distance_weights)
#             weights = cos_distance_weights

        self.federated_model = aggregate_models(models, weights)

    def draw_curve(self):
        plt.figure()
        x_epoch = list(range(len(self.train_loss)))
        plt.plot(x_epoch, self.train_loss, 'bo-', label='train')
        plt.legend()
        dir_name = os.path.join(self.project_dir, 'model', self.model_name)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        plt.savefig(os.path.join(dir_name, 'train.png'))
        plt.close('all')
        
    def test(self):
        print("="*10)
        print("Start Tesing!")
        print("="*10)
        print('We use the scale: %s'%self.multiple_scale)#multiple scale 의미 없는 것 같음
        
        for dataset in self.data.datasets:
            
#             self.federated_model = self.federated_model.eval()
            self.federated_model_0.load_state_dict(self.federated_model)
#             if use_cuda:
# #                 self.federated_model = self.federated_model.cuda()
#                self.federated_model_0=self.federated_model_0.to(self.device)
            
            with torch.no_grad():
                gallery_feature = extract_feature(self.federated_model_0, self.data.test_loaders[dataset]['gallery'], self.multiple_scale, self.device) #각 gallery 데이타의 normalized된 feature vector를 concatenate한다.
                query_feature = extract_feature(self.federated_model_0, self.data.test_loaders[dataset]['query'], self.multiple_scale, self.device) #각 query 데이타의 normalized된 feature vector를 concatenate한다.

            result = {
                'gallery_f': gallery_feature.numpy(),
                'gallery_label': self.data.gallery_meta[dataset]['labels'],
                'gallery_cam': self.data.gallery_meta[dataset]['cameras'],
                'query_f': query_feature.numpy(),
                'query_label': self.data.query_meta[dataset]['labels'],
                'query_cam': self.data.query_meta[dataset]['cameras']}

            scipy.io.savemat(os.path.join(self.project_dir,
                        'model',
                        self.model_name,
                        'pytorch_result.mat'),
                        result)
                        
            print(self.model_name)
            print(dataset)

            os.system('python evaluate.py --result_dir {} --dataset {}'.format(os.path.join(self.project_dir, 'model', self.model_name), dataset))

#     def knowledge_distillation(self, regularization):#pure fedpav 기준 필요 없을 것 같음!! 
#         MSEloss = nn.MSELoss().to(self.device)
#         optimizer = optim.SGD(self.federated_model.parameters(), lr=self.lr*0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)
#         self.federated_model.train()

#         for _, (x, target) in enumerate(self.data.kd_loader): 
#             x, target = x.to(self.device), target.to(self.device)
#             # target=target.long()
#             optimizer.zero_grad()
#             soft_target = torch.Tensor([[0]*512]*len(x)).to(self.device)
        
#             for i in self.client_list:
#                 i_label = (self.clients[i].generate_soft_label(x, regularization))
#                 soft_target += i_label
#             soft_target /= len(self.client_list)
        
#             output = self.federated_model(x)
            
#             loss = MSEloss(output, soft_target)
#             loss.backward()
#             optimizer.step()
#             print("train_loss_fine_tuning", loss.data)`