FROM tensorflow/tensorflow:latest-gpu-py3
#RUN apt-get update -y
#RUN apt-get upgrade -y
RUN apt-get install -y nvidia-smi
