# nn-dist-train-poc

### Acknowledgement
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2021-0-00907, Development of Adaptive and Lightweight Edge-Collaborative Analysis Technology for Enabling Proactively Immediate Response and Rapid Learning).

We implemented various FL algorithms across different tasks in our framework:
### Image Classification
- **FedAVG** ([Paper Link](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf), presented at **AISTATS 2017**),
- **FedNTD (Ours)** ([Paper Link](https://openreview.net/pdf?id=qw3MZb1Juo), presented at **NeurIPS 2022**),
- **FedFN (Ours)** ([Paper Link](https://openreview.net/pdf?id=4apX9Kcxie), presented at **NeurIPS Workshop 2023**).
- **FedDr+ (Ours)** ([Paper Link](https://openreview.net/pdf?id=FOi26eD2lQ), presented at **FedKDD 2024**).
### Re-Identification
- **FedPAV** ([Paper Link](https://dl.acm.org/doi/pdf/10.1145/3531013), presented at **ACMMM 2020**),
- **FedDKD (Ours)** ([Paper Link](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003117176), presented at **KIISE 2022**),  
### Object Detction

# Directory explanation

1. Federated image classification: embedded_devices (fedavg), fedntd_embedded_devices (fedntd)
2. Federated vehicle Re-ID: fedpav_reid (fedpav), fedntd_reid(fedpav+fedntd)



# Federated Learning on Embedded Devices with Flower

This demo will show you how Flower makes it very easy to run Federated Learning workloads on edge devices. Here we'll be showing how to use NVIDIA Jetson devices and Raspberry Pi as Flower clients. This demo uses Flower with PyTorch. The source code used is mostly borrowed from the [example that Flower provides for CIFAR-10](https://github.com/adap/flower/tree/main/src/py/flwr_example/pytorch_cifar).

## Getting things ready

This is a list of components that you'll need: 

* For server: A machine running Linux/macOS.
* For clients: either a Rapsberry Pi 3 B+ (RPi 4 would work too) or a Jetson Xavier-NX (or any other recent NVIDIA-Jetson device).
* A 32GB uSD card and ideally UHS-1 or better. (not needed if you plan to use a Jetson TX2 instead)
* Software to flash the images to a uSD card (e.g. [Etcher](https://www.balena.io/etcher/))

What follows is a step-by-step guide on how to setup your client/s and the server. In order to minimize the amount of setup and potential issues that might arise due to the hardware/software heterogenity between clients we'll be running the clients inside a Docker. We provide two docker images: one built for Jetson devices and make use of their GPU; and the other for CPU-only training suitable for Raspberry Pi (but would also work on Jetson devices). 

## Clone this repo

Start with cloning the Flower repo and checking out the example. We have prepared a single line which you can copy into your shell:

```bash
$ git clone https://github.com/etri-edgeai/nn-dist-train-poc.git
```

## Setting up the server

The only requirement for the server is to have flower installed. You can do so by running `pip install flwr` inside your virtualenv or conda environment.

## Setting up a Jetson Xavier-NX

> These steps have been validated for a Jetson Xavier-NX Dev Kit. An identical setup is needed for a Jetson Nano and Jetson TX2 once you get ssh access to them (i.e. jumping straight to point `4` below). For instructions on how to setup these devices please refer to the "getting started guides" for [Jetson Nano](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro) and [Jetson TX2](https://developer.nvidia.com/embedded/dlc/l4t-28-2-jetson-developer-kit-user-guide-ga). 

1. Download the Ubuntu 18.04 image from [NVIDIA-embedded](https://developer.nvidia.com/embedded/downloads), note that you'll need a NVIDIA developer account. This image comes with Docker pre-installed as well as PyTorch+Torchvision compiled with GPU support.
2. Extract the imgae (~14GB) and flash it onto the uSD card using Etcher (or equivalent).
3. Follow [the instructions](https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit) to setup the device.
4. Installing Docker: Docker comes pre-installed with the Ubuntu image provided by NVIDIA. But for convinience we will create a new user group and add our user to it (with the idea of not having to use `sudo` for every command involving docker (e.g. `docker run`, `docker ps`, etc)). More details about what this entails can be found in the [Docker documentation](https://docs.docker.com/engine/install/linux-postinstall/). You can achieve this by doing:
    ``` bash
    $ sudo usermod -aG docker $USER
    # apply changes to current shell (or logout/reboot)
    $ newgrp docker
    ```
5. The minimal installation to run this example only requires an additional package, `git`, in order to clone this repo. Install `git` by:

    ```bash
    $ sudo apt-get update && sudo apt-get install git -y
    ```


## Setting up a Raspberry Pi (3B+ or 4B)

1. Install Ubuntu server 20.04 LTS 64-bit for Rapsberry Pi. You can do this by using one of the images provided [by Ubuntu](https://ubuntu.com/download/raspberry-pi) and then use Etcher. Alternativelly, astep-by-step installation guide, showing how to download and flash the image onto a uSD card and, go throught the first boot process, can be found [here](https://ubuntu.com/tutorials/how-to-install-ubuntu-on-your-raspberry-pi#1-overview). Please note that the first time you boot your RPi it will automatically update the system (which will lock `sudo` and prevent running the commands below for a few minutes)

2. Install docker (+ post-installation steps as in [Docker Docs](https://docs.docker.com/engine/install/linux-postinstall/)):
    ```bash
    # make sure your OS is up-to-date
    $ sudo apt-get update

    # get the installation script
    $ curl -fsSL https://get.docker.com -o get-docker.sh

    # install docker
    $ sudo sh get-docker.sh

    # add your user to the docker group
    $ sudo usermod -aG docker $USER

    # apply changes to current shell (or logout/reboot)
    $ newgrp docker
    ```


# Dataset

For this demo we'll be using cifar 10 and veri 776. cifar 10 (https://www.cs.toronto.edu/~kriz/cifar.html), a popular dataset for image classification comprised of 10 classes (e.g. car, bird, airplane) and a total of 60K `32x32` RGB images. The training set contains 50K images. 


<img src="./cifar10.png" width="1000"/>

veri776 (https://github.com/JDAI-CV/VeRidataset), is  a large-scale benchmark dateset for vehicle Re-Id in the real-world urban surveillance scenario. It contains over 50,000 images of 776 vehicles captured by 20 cameras covering an 1.0 km^2 area in 24 hours, which makes the dataset scalable enough for vehicle Re-Id and other related research. The images are captured in a real-world unconstrained surveillance scene and labeled with varied attributes, e.g. BBoxes, types, colors, and brands. So complicated models can be learnt and evaluated for vehicle Re-Id. Each vehicle is captured by 2 ∼ 18 cameras in different viewpoints, illuminations, resolutions, and occlusions, which provides high recurrence rate for vehicle Re-Id in practical surveillance environment. It is also labeled with sufficient license plates and spatiotemporal information, such as the BBoxes of plates, plate strings, the timestamps of vehicles, and the distances between neighbouring cameras. 

&ensp;&ensp;&ensp;&ensp;![Image](./VeRi_240.png)&ensp;&ensp;![Image](./VeRi2_240.png)

The server will automatically download the dataset should it not be found in `./data`. To keep the client side simple, the datasets will be downloaded when building the docker image. This will happen as the first stage in both `run_pi.sh` and `run_jetson.sh`.


## Server

Launch the server and define the model you'd like to train. The current code (see `utils.py`) provides two models for CIFAR-10: a small CNN (more suitable for Raspberry Pi) and, a ResNet18, which will run well on the gpu. Each model can be specified using the `--model` flag with options `Net` or `ResNet18`. Launch a FL training setup with one client and doing three rounds as:
```bash
# launch your server. It will be waiting until one client connects
$ python3 server.py --server_address <YOUR_SERVER_IP:PORT> --rounds 30 --min_num_clients 4 --min_sample_size 4 --model ResNet18
```

## Clients

Asuming you have cloned this repo onto the device/s, then execute the appropiate script to run the docker image, connect with the server and proceed with the training. Note that you can use both a Jetson and a RPi simultaneously, just make sure you modify the script above when launching the server so it waits until 2 clients are online. 

### For Jetson

```bash
$ ./run_jetson.sh --server_address=SERVER_ADDRESS:8080 --cid=0 --model=ResNet18 --batch_size 50
$ python3 client.py --server_address=SERVER_ADDRESS:8080 --cid=0 --model=ResNet18 --batch_size 50
```

### For Raspberry Pi

Depending on the model of RapsberryPi you have, running the smaller `Net` model might be the only option due to the higher RAM budget needed for ResNet18. It should be fine for a RaspberryPi 4 with 4GB of RAM to run a RestNet18 (with an appropiate batch size) but bear in mind that each batch might take several second to complete. The following would run the smaller `Net` model:

```bash
# note that pulling the base image, extracting the content might take a while (specially on a RPi 3) the first time you run this.
$ ./run_pi.sh --server_address=<SERVER_ADDRESS> --cid=0 --model=Net
```

### Baseline(fedavg) and proposed Algorithm(fedntd) for federated image classification 

### 1st year Algorithm for federated vehicle re-id (fedpav)

Code for ACMMM 2020 oral paper - **[Performance Optimization for Federated Person Re-identification via Benchmark Analysis](https://arxiv.org/abs/2008.11560)**

Personal re-identification is an important computer vision task, but its development is constrained by the increasing privacy concerns. Federated learning is a privacy-preserving machine learning technique that learns a shared model across decentralized clients. In this work, we implement federated learning to person re-identification (**FedReID**) and optimize its performance affected by **statistical heterogeneity** in the real-world scenario. 

Algorithm: Federated Partial Averaging (FedPav)

<img src="./fedpav-new.png" width="700">

### 2nd year Algorithm for federated vehicle re-id (feddkd) 

We apply not true distilation proposed by [**"Preservation of Global Knowledge by Not-True Distillation in Federated Learning (NeurIPS 2022)"**](https://arxiv.org/abs/2106.03097) in fedpav algorithm 

<img src="./fedntd.png" width="600"/>

Detailed algorithm structure are as follows: 

## Notation
- **Total number of clients:** \( N \)
- **Total number of rounds:** \( R \)
- **Data held by each client \( n \in \{1, \ldots, N\} \):** \( D_n \)
- **Number of clients selected in round \( r \):** \( K_r \)
- **Initial global model parameter:** \( \theta^0 \)

## Procedure

1. **Initialization:**
   \[
   \theta^0 \quad \text{(initialize global model parameters)}
   \]

2. **For round \( r = 1, 2, \ldots, R \), do:**

   - **Client selection:**
     - Randomly sample \( K_r \) clients from \( N \) clients, denoted as \( S_r \subset \{1, \ldots, N\} \).

   - **Local update:**
     - Each selected client \( k \in S_r \):
       - **Step 1:** Divide local data \( D_k \) into two parts with equal probability and construct teacher model \( c_{k}^{t,r} \) using one part.
       - **Step 2:** Initialize student model \( \theta_{k}^{s,r,0} = \theta^{r-1} \).
       - **Step 3:** Train the student model using knowledge distillation from the teacher model, achieving local student model \( \theta_{k}^{s,r} \).
       - **Step 4:** Send the difference between the global model and the trained student model \( \Delta \theta_{k}^{r} = \theta_{k}^{s,r} - \theta^{r-1} \) to the server.

   - **Aggregation:**
     - The server updates the global model using the aggregation of local updates:
       \[
       \theta^{r} = \theta^{r-1} + \frac{\sum_{k \in S_r} |D_k| \Delta \theta_{k}^{r}}{\sum_{k \in S_r} |D_k|}
       \]


### 3rd year Algorithm for federated vehicle re-id (fedcon)

We focus on aligning global feature and local clients feature by using contrastive loss as proposed by [**"Model-Contrastive Federated Learning (CVPR 2021)"**](https://arxiv.org/abs/2103.16257) 
The structure of the algorithm remains consistent with previous ones. Algorithm are as follows: 

<img src="./moon.png" width="600"/>

## Notation
- **Feature extractor:** \( \theta(f) \)
- **Model parameters:** \( \theta = (\theta(f), \theta(c)) \)
- **Global feature extractor:** \( \theta^{r-1}(f) \)
- **Client's local model:** \( \theta_{i}^{r-1} = (\theta_{i}^{r-1}(f), \theta_{i}^{r-1}(c)) \)
- **Loss function:** \( l_{r} = l_{CE} + l_{con} \)
- **Hyperparameters:** \( \tau = 0.5 \), \( \mu = 5 \)

## Procedure

1. **Initialization:**
   - **Central server:** Transmits the global feature extractor \( \theta^{r-1}(f) \) to \( K \) selected clients among \( N \) clients.

2. **For each round \( r \):**
   - **Client \( i \) Initialization:**
     - Client \( i \) initializes its local model as \( \theta_{i} = (\theta^{r-1}(f), \theta_{i}^{r-1}(c)) \).

   - **Local Model Training:**
     - Client \( i \) uses the stochastic gradient descent (SGD) optimizer to train the model with the loss function \( l_{r} \), which is the sum of cross-entropy loss \( l_{CE} \) and contrastive loss \( l_{con} \).
     - The loss function \( l_{r} \) has two hyperparameters \( (\tau, \mu) \) as specified in MOON[7].
     - The cosine similarity between two vectors is denoted as \( \text{sim}(\cdot, \cdot) \).
     - The feature representation at the penultimate layer (before the logit layer) is indicated as \( f(x; \theta(f)) \).

   - **Model Update:**
     - Client \( i \) optimizes the loss \( l_{r} \) to obtain the updated local model \( \theta_{i}^{r} \).
     - Client \( i \) transmits only the updated feature extractor \( \theta_{i}^{r}(f) \) to the central server.

   - **Aggregation:**
     - The central server aggregates the feature extractors \( \theta_{i}^{r}(f) \) from the \( K \) selected clients.
     - The global feature extractor \( \theta^{r}(f) \) is updated using a convex combination of the received feature extractors.


### 4th year Algorithm for federated vehicle re-id (fedcon+)

we propose fedcon+ which applies lp-ft (linear probing and fine tuning) on fedcon algorithm. The structure of the algorithm remains consistent with previous ones,  and below figure describes the differences compared to the earlier algorithm.

<img src="./lp-ft.jpg" width="600"/>


# Result of federated vehicle re-id (dataset: veri 776) - to be updated 

communication round:30, local iteration:1

CMC (Cumulative Matching Characteristics)
AP (Average Precision)

||CMC rank@1|CMC Rank@5|CMC Rank@10|AP|
|------|---|---|---|---|
|fedpav|40.00|72.50|85.00|0.19|
|fedntd|45.00|75.00|90.00|0.20|
