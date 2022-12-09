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
Command to make GPU Benchmark
https://www.docker.com/blog/wsl-2-gpu-support-for-docker-desktop-on-nvidia-gpus/
docker run -it --gpus=all --rm nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -benchmark

docker run -it rayproject/ray:latest-gpu rllib train --run=PPO --env=CartPole-v0
docker run -it --gpus=all peterpirogtf/ray_tf2:gpu rllib train --run=PPO --env=CartPole-v0 --config '{"num_workers": 4,"num_gpus": 0}'

docker run -it --gpus=all peterpirogtf/ray_tf2:gpu rllib train --run=PPO --env=CartPole-v0 --config '{"num_workers": 4,"num_gpus": 0}'
908a340a1fe6

docker build -t rllib210 .
docker run -it -d rllib210 rllib train --run=PPO --env=CartPole-v0   

rllib train --run=PPO --env=CartPole-v1 --config '{"num_workers": 4,"num_gpus": 0,"framework":"tf2"}'


pip install "ray[rllib]" tensorflow

Command to run if ray server is inactive:
rllib train --run DQN --env CartPole-v1 --framework tf2 --ray-num-cpus 8 --ray-num-gpus 0 --config '{"num_workers": 7}'

usage: rllib train [-h] [--run RUN] [--stop STOP] [--config CONFIG] [--resources-per-trial RESOURCES_PER_TRIAL] [--num-samples NUM_SAMPLES]
                   [--checkpoint-freq CHECKPOINT_FREQ] [--checkpoint-at-end] [--sync-on-checkpoint] [--keep-checkpoints-num KEEP_CHECKPOINTS_NUM]
                   [--checkpoint-score-attr CHECKPOINT_SCORE_ATTR] [--export-formats EXPORT_FORMATS] [--max-failures MAX_FAILURES] [--scheduler SCHEDULER]
                   [--scheduler-config SCHEDULER_CONFIG] [--restore RESTORE] [--ray-address RAY_ADDRESS] [--ray-ui] [--no-ray-ui] [--local-mode]
                   [--ray-num-cpus RAY_NUM_CPUS] [--ray-num-gpus RAY_NUM_GPUS] [--ray-num-nodes RAY_NUM_NODES] [--ray-object-store-memory RAY_OBJECT_STORE_MEMORY]
                   [--experiment-name EXPERIMENT_NAME] [--local-dir LOCAL_DIR] [--upload-dir UPLOAD_DIR] [--framework {tf,tf2,tfe,torch}] [-v] [-vv] [--resume] [--trace]
                   [--env ENV] [-f CONFIG_FILE] [--torch] [--eager]
