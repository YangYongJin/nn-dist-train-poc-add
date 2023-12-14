ARG BASE_IMAGE_TYPE=cpu
# these images have been pushed to Dockerhub but you can find
# each Dockerfile used in the `base_images` directory 
FROM jafermarq/jetsonfederated_$BASE_IMAGE_TYPE:latest

RUN apt-get install wget -y

# Download and extract CIFAR-10
# To keep things simple, we keep this as part of the docker image.
# If the dataset is already in your system you can mount it instead.


ENV DATA_DIR=/app/veri1
#RUN mkdir -p $DATA_DIR
#WORKDIR $DATA_DIR

WORKDIR /app
# Scripts needed for Flower client
ADD veri1 /app
ADD client.py /app
ADD utils.py /app
ADD data_utils.py /app
ADD evaluate.py /app
ADD random_erasing.py /app
ADD resnet.py /app
# update pip
RUN pip3 install --upgrade pip

# making sure the latest version of flower is installed
RUN pip3 install flwr==0.16.0

ENTRYPOINT ["python3","-u","./client.py"]
