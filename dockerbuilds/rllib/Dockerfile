#Use existing docker image as a base
# Docker  Dokerfile commands https://docs.docker.com/engine/reference/builder/
# Docker run commands https://docs.docker.com/engine/reference/commandline/run/
FROM tensorflow/tensorflow:2.11.0-gpu
LABEL maintainer="peterpirogtf@gmail.com"
ARG GYM_VERSION=0.23.1
#RAY Variables
ARG RAY_VERSION=2.1.0
ARG RAY_RESULTS="/root/ray_results"
ARG RAY_PORT=6379
ARG RAY_CLIENT_SERVER_PORT=10001
ARG RAY_DASHBOARD_PORT=8265

#Download and install dependencies
#RUN export python=python3
RUN export python=python3

RUN apt-get update -y
RUN apt install xvfb -y

#Upgrade pip and setuptools
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools

# Add rllib and tune
RUN pip install "ray[rllib]"==$RAY_VERSION
RUN pip install "ray[tune]"==$RAY_VERSION

# INSTALL GYM ATARI
RUN pip install pygame
RUN pip install "gym[atari]"==$GYM_VERSION "gym[accept-rom-license]"==$GYM_VERSION atari_py

# INSTALL GYM BOX2D
RUN apt-get install swig -y # needed for box2d
RUN pip install "gym[box2d]"==$GYM_VERSION

# INSTALL GYM MUJOCO
#  https://github.com/openai/mujoco-py#install-mujoco
RUN apt install libosmesa6-dev libgl1-mesa-glx libglfw3 -y
RUN pip install lockfile
RUN pip install Cython
RUN pip install cffi  # needed for mujoco
RUN apt install wget
RUN apt-get install patchelf
RUN pip install glfw


RUN mkdir -p /root/.mujoco
WORKDIR /root/.mujoco
RUN wget https://www.roboti.us/download/mjpro150_linux.zip
RUN unzip mjpro150_linux.zip
RUN wget https://www.roboti.us/file/mjkey.txt
RUN rm mjpro150_linux.zip

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin"
ENV PATH="$PATH:/usr/local/bin/"

RUN pip install "gym[mujoco]"==$GYM_VERSION

WORKDIR /
RUN pip freeze >> requirements.txt
COPY Dockerfile ./
RUN apt-get autoremove -y


#EXPOSE PORTS
EXPOSE $RAY_PORT
EXPOSE $RAY_DASHBOARD_PORT
EXPOSE $RAY_CLIENT_SERVER_PORT

# /usr/local/bin/ray start --head --port 6379   --ray-client-server-port 10001

#
#RUN export python=xvfb-run -s "screen 0 1400x900x24" python3

# Container start command
#RUN chmod +x /usr/local/bin/ray

CMD ["/bin/bash"]
#CMD ["/usr/local/bin/ray start --head --port $RAY_PORT --dashboard-port $RAY_DASHBOARD_PORT --ray-client-server-port $RAY_CLIENT_SERVER_PORT"]
#CMD ["ray start --head"]
# ray start --head --dashboard-host 0.0.0.0 --include-dashboard True --dashboard-port 8265

#command to build new image:
#sudo docker build -t peterpirogtf/rllib210:gpu .

#How to push docker image to hub

#login by:
# docker login
# docker push

