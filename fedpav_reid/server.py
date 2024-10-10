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

import wandb
import argparse
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple
from data_utils import Data
import flwr as fl
import numpy as np
import torch
import torchvision
import random
import utils
from data_utils import Data
import os
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(1)
random.seed(1)
# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

parser = argparse.ArgumentParser(description="Flower")
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
    type=str,
    help="Logserver address (no default)",
)
parser.add_argument(
    "--model",
    type=str,
    default="ResNet18",
    choices=["Net", "ResNet18","ResNet8", "ResNet50"],
    help="model to train",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="training batch size",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="number of workers for dataset reading",
)
parser.add_argument('--data_dir',default='data',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
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

parser.add_argument("--pin_memory", action="store_true")
args = parser.parse_args()
""""""

def main() -> None:
    """ Start server and train five rounds."""
    wandb.init(project="veri776", reinit=True)
    wandb.run.name = 'fedpav'
    wandb.run.save()
    print(args)

    assert (
        args.min_sample_size <= args.min_num_clients
    ), f"Num_clients shouldn't be lower than min_sample_size"

    # Configure logger
    fl.common.logger.configure("server", host=args.log_host)

    # Load evaluation data
    #_, testset = utils.load_cifar(download=True)
    data = Data(args.batch_size, args.erasing_p, args.color_jitter, args.train_all)
    data.preprocess()
    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        min_eval_clients=args.min_num_clients,
        eval_fn=get_eval_fn(data.testloader, data.gallery_meta, data.query_meta),
        on_fit_config_fn=fit_config,
    )
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)

    # Run server
    fl.server.start_server(
        server_address=args.server_address,
        server=server,
        config={"num_rounds": args.rounds},
    )


def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(server_round),
        "epochs": str(3),
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
    testloader: torch.utils.data.DataLoader,
    gallery_meta,
    query_meta,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        model = utils.load_model(args.model)
        set_weights(model, weights)
        model.to(DEVICE)

        #testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        cmc, ap, loss, acc = utils.test(model, testloader, gallery_meta, query_meta, device=DEVICE)
        #wandb.log({"CMC": acc.item() if isinstance(acc, torch.Tensor) else acc,
                   #"Average Precision": ap.item() if isinstance(ap, torch.Tensor) else ap,
                   #})
        #return  {"cmc": accuracy}

    return evaluate


if __name__ == "__main__":
    main()
