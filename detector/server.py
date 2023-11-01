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
"""Minimal example on how to start a simple Flower server."""


import argparse
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple
from logging import INFO, DEBUG
import flwr as fl
import numpy as np
import torch
import torchvision
import random
import utils
import time
import os
import wandb
from torch.utils.data import DataLoader, Subset

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False


# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

parser = argparse.ArgumentParser(description="Flower")

# set seed
parser.add_argument(
    "--random_seed",
    type=int,
    default=1,
    help="random seed (default: 1)",
)

parser.add_argument(
    "--server_address",
    type=str,
    required=True,
    help=f"gRPC server address",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=1,
    help="Number of rounds of federated learning (default: 1)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_sample_size",
    type=int,
    default=2,
    help="Minimum number of clients used for fit/evaluate (default: 2)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)
parser.add_argument(
    "--log_host",
    default="./log",
    type=str,
    help="Log directory (no default)",
)
parser.add_argument(
    "--model",
    type=str,
    default="resnet",
    choices=["resnet", "mobilenet"],
    help="model to train",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="training batch size",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=1,
    help="number of epochs to train",
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.1,
    help="learning rate",
)

parser.add_argument(
    "--optimizer",
    type=str,
    default="SGD",
    choices=["SGD", "Adam", "RMSprop", "SGDM"],
    help="optimizer to use",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="fedavg",
    choices=["fedavg", "fedprox", "fedadam", "fedyogi"],
    help="optimizer to use",
)


parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="number of workers for dataset reading",
)
parser.add_argument("--pin_memory", action="store_true")
args = parser.parse_args()

BEST_ACCURACY = 0.0

def main() -> None:
    """init wandb"""
    wandb.init(project="feddetector")
    wandb.config.update(args)

    """Start server and train five rounds."""

    print(args)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)



    assert (
        args.min_sample_size <= args.min_num_clients
    ), f"Num_clients shouldn't be lower than min_sample_size"

    # Configure logger
    fl.common.logger.configure(identifier=f"Feddetector_{args.random_seed}", filename=os.path.join(args.log_host, f"seed_{args.random_seed}.log"))

    # Load evaluation data
    testset = utils.load_smartfarm_data()

    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()

    model = utils.load_model(args.model)
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = fl.common.weights_to_parameters(weights)
    parameters = fl.common.ParametersRes(parameters=parameters)

    if args.algorithm == "fedavg":
        strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(testset),
        on_fit_config_fn=fit_config,
    )
    elif args.algorithm == "fedprox":
        strategy = fl.server.strategy.FedProx(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(testset),
        on_fit_config_fn=fit_config,
    )
    elif args.algorithm == "fedadam":
        strategy = fl.server.strategy.FedAdam(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(testset),
        on_fit_config_fn=fit_config,
        initial_parameters=parameters,
        eta = 1e-2,
        eta_l = 1e-3,
        beta_1 = 0.9,
        beta_2 = 0.99,
    )
    elif args.algorithm == "fedyogi":
        strategy = fl.server.strategy.FedYogi(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(testset),
        on_fit_config_fn=fit_config,
        initial_parameters=parameters,
        eta = 1e-2,
        eta_l = 1e-3,
        beta_1 = 0.9,
        beta_2 = 0.99,
    )
    else: 
        raise NotImplementedError(f"algorithm {args.algorithm} is not implemented")
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)

    # Start timer
    start_time = time.time()

    # Run server
    fl.server.start_server(
        server_address=args.server_address,
        server=server,
        config={"num_rounds": args.rounds},
    )

    # Stop timer in minutes
    end_time = time.time()
    print(f"Total time for training {args.rounds}: {(end_time - start_time)/60} minutes")

    # Best Performance
    


def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(server_round),
        "epochs": str(args.epochs),
        "lr": str(args.lr),
        "algorithm": str(args.algorithm),
        "batch_size": str(args.batch_size),
        "num_workers": str(args.num_workers),
        "pin_memory": str(args.pin_memory),
        "optimizer": str(args.optimizer),
    }
    return config


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def get_eval_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        model = utils.load_model(args.model)
        set_weights(model, weights)
        model.to(DEVICE)

        # # Determine the number of samples for 10% of the dataset
        # num_samples = len(testset)
        # subset_size = int(0.01 * num_samples)

        # # Generate random indices for the samples
        # indices = np.random.choice(num_samples, subset_size, replace=False)

        # # Create the subset
        # subset = Subset(testset, indices)

        testloader = DataLoader(testset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
        mAP_at_05 = utils.test(model, testset, testloader, device=DEVICE)

        # log this accuracy
        if args.log_host:
            fl.common.logger.log(
                INFO, f"eval mAP_at_05 {mAP_at_05}"
            )
        wandb.log({"mAP_at_05": mAP_at_05})
        # update best accuracy 
        global BEST_ACCURACY
        if mAP_at_05 > BEST_ACCURACY:
            BEST_ACCURACY = mAP_at_05
            fl.common.logger.log(
                INFO, f"eval {mAP_at_05} BEST map"
            )
        wandb.log({"best_mAP_at_05": mAP_at_05})

        return (0.0, {"mAP_at_05": mAP_at_05})

    return evaluate


if __name__ == "__main__":
    main()
