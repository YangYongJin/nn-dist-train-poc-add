# FedPAV, FedNTD, FedCON, and FedCON+ - Re-ID

## Prerequisites & Installation


```bash
conda env create -n reid --file etri.yaml
```

You can simply download environment for run re-id code of various models.

## Run Server and Client for Re-ID Experiment

First, navigate to the folder of the model you want to run the experiment for:

```bash
cd "model"
```
Then, execute the server and client code simultaneously.

* For Sever

```bash
python3 server.py --server_address 192.168.0.17:8080 --rounds 15 --min_num_clients "Number of client" --min_sample_size "Minimum number of participating clinets" --model ResNet50
```

* For Client

```bash
python3 client.py --server_address=137.68.194.166:8080 --cid=0 --model=ResNet50 --batch_size "batch size"
```

To customize the number of clients, you can change --min_num_clients and --min_sample_size to the number of clients you wish to have.

Ex. If you have 5 clients and want to participate 3 clients then, --min_num_clients = 5 and --min_sample_size = 3.


## For Vehicle Re-ID tasks

### Datasets

We conducted a Vehicle Re-ID task using the Veri-776 dataset.

You can download the datasets from [here](https://vehiclereid.github.io/VeRi/).

## For Person Re-ID tasks

### Datasets

We conducted a Re-ID task using the "한국인 재식별 이미지" dataset provided by AI HUB.

You can download the datasets from [here](https://aihub.or.kr/aidata/7977).

### Preprocessing 

For preprocessing of “Korean re-identification image”, run the code below.

'''bash
python3 preprocessing.py
'''

You can change the type of dataset you want to use as an option.

## Acknowledgments

In this repo, FedPAV refers to [FedReID](https://github.com/cap-ntu/FedReID), while the remaining models were implemented directly. Thanks!


