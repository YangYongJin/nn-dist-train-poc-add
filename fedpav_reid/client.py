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
"""Flower client example using PyTorch for CIFAR-10 image classification."""


import argparse
import timeit
from collections import OrderedDict
from importlib import import_module
import os
import flwr as fl
import numpy as np
import torch
import torchvision
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights
from data_utils import Data
import utils

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member


def get_weights(model: torch.nn.ModuleList) -> fl.common.Weights:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


class CifarClient(fl.client.Client):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        cid: str,
        model: torch.nn.Module,
        data,
        
    ) -> None:
        self.cid = cid
        self.model = model
        self.data = data
        


    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = get_weights(self.model)
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def _instantiate_model(self, model_str: str):

        # will load utils.model_str
        m = getattr(import_module("utils"), model_str)
        # instantiate model
        self.model = m()
        

    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit")

        weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()

        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        pin_memory = bool(config["pin_memory"])
        num_workers = int(config["num_workers"])

        # Set model parameters
        set_weights(self.model, weights)

        if torch.cuda.is_available():
            kwargs = {
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "drop_last": True,
            }
        else:
            kwargs = {"drop_last": True}

        # Train model
        '''
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True, **kwargs
        )
        '''
        
        
        utils.train(self.model, self.data.trainloader, self.data.dataset_sizes, epochs=epochs, device=DEVICE)
        #utils.test(self.model, self.data.testloader, self.data.gallery_meta, self.data.query_meta,  device=DEVICE)

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = get_weights(self.model)
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = self.data.dataset_sizes
        metrics = {"duration": timeit.default_timer() - fit_begin}
        return FitRes(
            parameters=params_prime, num_examples=num_examples_train, metrics=metrics
        )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")

        weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config
        
        # Use provided weights to update the local model
        set_weights(self.model, weights)
        num_examples_test = self.data.dataset_sizes
        cmc, ap, loss, acc = utils.test(self.model, self.data.testloader, self.data.gallery_meta, self.data.query_meta, device=DEVICE)
        loss_value = float(loss.cpu().detach().numpy())
        metrics = {"rank10": float(cmc[10].cpu().detach().numpy())}
        # Evaluate the updated model on the local dataset

        '''
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=32, shuffle=False
        )
        '''
        #utils.test(self.model, self.data.testloader, self.data.gallery_meta, self.data.query_meta, device=DEVICE)
        '''
        print("="*10)
        print("Start Testing!")
        print("="*10)
        #print('We use the scale: %s'%self.multiple_scale)
        for dataset in self.data.datasets:
            self.net = self.net.eval()
            if use_cuda:
                self.net = self.net.cuda()
            
            with torch.no_grad():
                gallery_feature = extract_feature(self.net, self.data.test_loaders[dataset]['gallery'], self.multiple_scale)
                query_feature = extract_feature(self.net, self.data.test_loaders[dataset]['query'], self.multiple_scale)

            result = {
                    'gallery_f': gallery_feature.numpy(),
                    'gallery_label': self.data.gallery_meta[dataset]['labels'],
                    'gallery_cam': self.data.gallery_meta[dataset]['cameras'],
                    'query_f': query_feature.numpy(),
                    'query_label': self.data.query_meta[dataset]['labels'],
                    'query_cam': self.data.query_meta[dataset]['cameras']}
        
            scipy.io.savemat(os.path.join('.','model',args.model,'pytorch_result.mat'),result)
            os.system('python evaluate.py --result_dir {} --dataset {}'.format(os.path.join('.', 'model', args.model), dataset))                
            #print(self.model_name)
            #print(dataset)
        

        # Return the number of evaluation examples and the evaluation result (loss)
        #metrics = {"accuracy": float(accuracy)}
        '''
        return EvaluateRes(loss=loss_value, num_examples=num_examples_test, metrics=metrics
            
        )
        

def main() -> None:
    """Load data, create and start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        required=True,
        help=f"gRPC server address",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where the dataset lives",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ResNet18",
        choices=["Net", "ResNet18","ResNet8","ResNet50"],
        help="model to train",
    )
    parser.add_argument('--batch_size', type=int)
    #parser.add_argument('--data_dir',default='data',type=str, help='training dir path')
    parser.add_argument('--train_all', action='store_true', help='use all training data')
    parser.add_argument('--stride', default=2, type=int, help='stride')
    #parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    #parser.add_argument('--drop_rate', default=0.5, type=float, help='drop rate')

# arguments for federated setting
    #parser.add_argument('--local_epoch', default=1, type=int, help='number of local epochs')
    #parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    #parser.add_argument('--num_of_clients', default=9, type=int, help='number of clients')

# arguments for data transformation
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )

# arguments for testing federated model
    parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--multi', action='store_true', help='use multiple query' )
    parser.add_argument('--multiple_scale',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
    parser.add_argument('--test_dir',default='all',type=str, help='./test_data')


    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)
    

    # model
    model = utils.load_model(args.model)
    model.to(DEVICE)
    # load (local, on-device) dataset
    
    data = Data(args.batch_size, args.erasing_p, args.color_jitter, args.train_all)
    data.preprocess()
    #refined_data = data.transform()
    #trainloader, testloader, gallery_meta, query_meta = data.preprocess()
    #trainset, testset = utils.load_cifar()

    # Start client
    client = CifarClient(args.cid, model, data)
    fl.client.start_client(args.server_address, client)


if __name__ == "__main__":
    main()
