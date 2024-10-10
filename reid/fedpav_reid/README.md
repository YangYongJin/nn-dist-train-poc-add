## Getting things ready

This is a list of components that you'll need: 

* For server: A machine running Linux/macOS.
* For clients: either a Rapsberry Pi 3 B+ (RPi 4 would work too) or a Jetson Xavier-NX (or any other recent NVIDIA-Jetson device).
* A 32GB uSD card and ideally UHS-1 or better. (not needed if you plan to use a Jetson TX2 instead)
* Software to flash the images to a uSD card (e.g. [Etcher](https://www.balena.io/etcher/))

What follows is a step-by-step guide on how to setup your client/s and the server. 
In order to minimize the amount of setup and potential issues that might arise due to the hardware/software heterogenity between clients we'll be running the clients inside a Docker. 
We provide two docker images: one built for Jetson devices and make use of their GPU; and the other for CPU-only training suitable for Raspberry Pi (but would also work on Jetson devices). 

## Server

```bash
# launch your server. It will be waiting until one client connects
$ python3 server.py --server_address <YOUR_SERVER_IP:PORT> --rounds 3 --min_num_clients 1 --min_sample_size 1 --model ResNet18
```

## Clients

```bash
$ ./run_jetson.sh --server_address=SERVER_ADDRESS:8080 --cid=0 --model=ResNet18 --batch_size=10
```

![image](https://github.com/etri-edgeai/nn-dist-train-poc/assets/58346392/c30bf76c-3464-4e1d-aac8-f7a455175f9a)

