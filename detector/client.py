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

import flwr as fl
import numpy as np
import torch
import torchvision
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights
import copy

from torch.utils.data import Dataset
import random

import utils



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

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

    def __init__(
        self,
        cid: str,
        model: torch.nn.Module,
        trainset: torchvision.datasets,
        testset: torchvision.datasets,
        classes: list,
    ) -> None:
        self.cid = cid
        self.model = model
        self.trainset = trainset
        self.testset = testset
        # self.classes = classes

        # make random lists for selection of classes
        # select 80% for classes in self.classes and 20% for other classes
        # self.id_idxs = []
        # for i, (_, label) in enumerate(self.trainset):
        #     if label in self.classes:
        #         if random.random() < 0.8:
        #             self.id_idxs.append(i)
        #     else:
        #         if random.random() < 0.2:
        #             self.id_idxs.append(i)


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
        lr = float(config["lr"])
        batch_size = int(config["batch_size"])
        optimizer_name = config["optimizer"]
        algorithm = config["algorithm"]
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
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True,  collate_fn=lambda x: tuple(zip(*x)), **kwargs
        )
        utils.train(self.model, trainloader, lr=lr, epochs=epochs, optimizer_n=optimizer_name, algorithm=algorithm ,device=DEVICE)

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = get_weights(self.model)
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = len(self.trainset)
        metrics = {"duration": timeit.default_timer() - fit_begin}
        return FitRes(
            parameters=params_prime, num_examples=num_examples_train, metrics=metrics
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")

        weights = fl.common.parameters_to_weights(ins.parameters)

        # Use provided weights to update the local model
        set_weights(self.model, weights)

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x))
        )
        loss, accuracy = 0.0, 0.0 #utils.test(self.model, self.testset, testloader, device=DEVICE)

        # Return the number of evaluation examples and the evaluation result (loss)
        metrics = {"accuracy": float(accuracy)}
        return EvaluateRes(
            loss=float(loss), num_examples=len(self.testset), metrics=metrics
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
        default="resnet",
        choices=["resnet", "mobilenet"],
        help="model to train",
    )
    # arguments for self.classes (default: [0,1]) list
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=[0,1],
        help="list of classes to train on",
    )
    
    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # model
    model = utils.load_model(args.model)
    model.to(DEVICE)
    # load (local, on-device) dataset
    trainset = utils.load_smartfarm_data()
    testset = trainset

    # Start client
    client = CifarClient(args.cid, model, trainset, testset, args.classes)
    fl.client.start_client(args.server_address, client)


if __name__ == "__main__":
    main()
