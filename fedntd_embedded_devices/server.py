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
    default="ResNet18",
    choices=["Net", "ResNet18","ResNet8"],
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
    wandb.init(project="fedntd")
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
    fl.common.logger.configure(identifier=f"FedFN_{args.random_seed}", filename=os.path.join(args.log_host, f"seed_{args.random_seed}.log"))

    # Load evaluation data
    _, testset = utils.load_cifar(download=True)

    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(testset),
        on_fit_config_fn=fit_config,
    )
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
        "batch_size": str(args.batch_size),
        "num_workers": str(args.num_workers),
        "pin_memory": str(args.pin_memory),
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

        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        loss, accuracy = utils.test(model, testloader, device=DEVICE)

        # log this accuracy
        if args.log_host:
            fl.common.logger.log(
                INFO, f"eval/loss {loss} accuracy {accuracy}"
            )
        wandb.log({"accuracy": accuracy, "loss": loss})
        # update best accuracy 
        global BEST_ACCURACY
        if accuracy > BEST_ACCURACY:
            BEST_ACCURACY = accuracy
            fl.common.logger.log(
                INFO, f"eval/loss {loss} accuracy {accuracy} BEST ACCURACY"
            )
        wandb.log({"best_accuracy": BEST_ACCURACY})

        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    main()
