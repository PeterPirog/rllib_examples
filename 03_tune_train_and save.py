# https://gym.openai.com/envs/LunarLanderContinuous-v2/
from ray import tune

if __name__ == '__main__':

    tune.run("PPO",
             config={"env": "LunarLanderContinuous-v2",
                     "evaluation_interval": 2,
                     "evaluation_num_episodes": 20},
             #resources_per_trial={"cpu": 8, "gpu": 0},
             local_dir="LunarLanderContinuous_v2",  # directory to save results
             checkpoint_freq=2,  # frequency between checkpoints
             keep_checkpoints_num=3, )

# tensorboard --bind_all --port=12301 --logdir /home/ppirog/projects/rllib_examples/examples/02_continous_lunarlander/LunarLanderContinuous_v2/PPO
# web browser address: http://212.109.128.252:12301/#scalars
