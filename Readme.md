# Examples of using ray rllib 

Some examples how to use [Ray Rllib](https://docs.ray.io/en/latest/rllib/index.html#) excellent framework for Reinforcement Learning


## Ray dependencies to install
# Tested with python 3.8
pip install pip-autoremove <br />
pip install -U "ray[default]"<br />
pip install -U "ray[tune]"  # installs Ray + dependencies for Ray Tune<br />
pip install -U "ray[rllib]"  # installs Ray + dependencies for Ray Rllib<br />
pip install tensorflow<br />
pip install pygame

pip install gym<br />
pip install gym[classic_control]<br />
pip install "gym[atari]" "gym[accept-rom-license]" atari_py<br />


pip install tensorflow
# How to install  CUDA in WINDOWS 10 in super fast way
 CUDA https://www.youtube.com/watch?v=toJe8ZbFhEc
conda create -n tf_rllib python==3.8
conda activate tf_rllib
conda install cudatoolkit=11.2 cudnn=8.1 -c=conda-forge
pip install --upgrade tensorflow-gpu==2.10.1
pip install -U "ray[tune]"
pip install -U "ray[rllib]"
pip install pygame
pip install gym[classic_control]
pip install "gym[atari]" "gym[accept-rom-license]" atari_py

## Step 1  - test if gym environments are running
Test files below if its work correctly. If no, don't go to the next step.

[CartPole basic example](01a_test_cartpole_env.py)<br />
[Breakout basic example](01b_test_breakout_env.py)<br />


rllib train --env=PongDeterministic-v4 --run=A2C --config '{"num_workers": 4}'
# DOCKER
docker run -it rayproject/ray:latest-gpu rllib train --run=PPO --env=CartPole-v0

docker build -t rllib210 .
docker run -it -d rllib210 rllib train --run=PPO --env=CartPole-v0   