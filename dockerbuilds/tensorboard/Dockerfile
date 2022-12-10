#Use existing docker image as a base
# Docker  Dokerfile commands https://docs.docker.com/engine/reference/builder/
# Docker run commands https://docs.docker.com/engine/reference/commandline/run/
FROM python:3.11.0-slim

ARG TF2_VERSION=2.11.0
ARG TF2_PORT=6006
ARG LOGDIR_PATH="/ray_results"
LABEL maintainer="peterpirogtf@gmail.com"

RUN apt-get update
RUN apt-get dist-upgrade -y

RUN export python=python3
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools

RUN pip install -U tensorboard==$TF2_VERSION

RUN mkdir -p $LOGDIR_PATH
EXPOSE $TF2_PORT
VOLUME $LOGDIR_PATH
COPY Dockerfile /
WORKDIR $LOGDIR_PATH
CMD ["/bin/bash"]
#CMD ["tensorboard --logdir /ray_results --port=6006 --bind_all"]

# command to ru
# docker run -it -v C:\Users\Ila\ray_results:/ray_results -p6006:6006 peterpirogtf/tensorboard:2.11.0 tensorboard --logdir /ray_results --port=6006 --bind_all

# tensorboard --logdir /ray_results --port=6001 --bind_all
# tensorboard --logdir C:\Users\Ila\ray_results --port=6006 --bind_all

#command to build new image:
#sudo docker build -t peterpirogtf/tensorboard:2.11.0 .

#How to push docker image to hub

#login by:
# docker login
# docker push

