
ARG BASE_IMAGE_TYPE=gpu
# these images have been pushed to Dockerhub but you can find
# each Dockerfile used in the `base_images` directory 
FROM jetsonfederated_$BASE_IMAGE_TYPE:latest
# FROM jafermarq/jetsonfederated_$BASE_IMAGE_TYPE:latest
# - except jetson orin

RUN apt-get install wget -y

# Download and extract CIFAR-10
# To keep things simple, we keep this as part of the docker image.
# If the dataset is already in your system you can mount it instead.
ENV DATA_DIR=/app/data/cifar-10
RUN mkdir -p $DATA_DIR
WORKDIR $DATA_DIR
RUN wget https://www.cs.toronto.edu/\~kriz/cifar-10-python.tar.gz 
RUN tar -zxvf cifar-10-python.tar.gz

WORKDIR /app
# Scripts needed for Flower client
ADD client.py /app
ADD utils.py /app

# RUN apt update

# RUN apt install python3.8 -y

# update pip
RUN pip3 install --upgrade pip


# making sure the latest version of flower is installed
RUN pip3 install flwr==0.19.0
#RUN pip3 install flwr==0.19.0

ENTRYPOINT ["python3","-u","./client.py"]
