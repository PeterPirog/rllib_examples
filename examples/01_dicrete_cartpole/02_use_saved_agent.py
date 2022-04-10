import ray
import pyglet
from ray.rllib.agents.ppo.ppo import PPOTrainer
import gym
from gym.wrappers import RecordVideo
# pip install ffmpeg
# pip install imageio-ffmpeg
# sudo apt install python-opengl
# sudo apt install ffmpeg
# sudo apt install xvfb
# pip3 install pyvirtualdisplay

# configuration the same like during training
config={"env": "CartPole-v1",
                     "evaluation_interval": 2,
                     "evaluation_num_episodes": 20}

if __name__ == '__main__':
    agent=PPOTrainer(config=config)
    agent.restore(checkpoint_path="/home/ppirog/projects/rllib_examples/examples/01_dicrete_cartpole/cartpole_v1/PPO/PPO_CartPole-v1_6b7eb_00000_0_2022-04-10_08-44-27/checkpoint_001462/checkpoint-1462")
    #ray.init()

    #env=gym.make("CartPole-v1")
    env=RecordVideo(gym.make("CartPole-v1"),"ppo_video")
    obs=env.reset()
    while True:
        action=agent.compute_action(obs)
        obs,reward,done,_=env.step(action)
        #env.render()
        if done:
            break
    env.close()