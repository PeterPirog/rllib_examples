import numpy as np
import gym
from gym.spaces import Box, Discrete, Dict
from ray.tune.registry import register_env
from ray import tune


class BanditEnv(gym.Env):
    def __init__(self, probabilities=[0.1, 0.5, 0.6], number_of_draws=50):
        self.probabilities = probabilities
        self.number_of_draws = number_of_draws
        self.max_avail_actions = len(self.probabilities)
        self.action_space = Discrete(self.max_avail_actions)
        self.observation_space = Box(0, 2, (self.max_avail_actions,), ) # using 2 not 1 cause of the noise value

        self.reset()

    def reset(self):
        self.current_draw = 0
        self.done = False
        self.observation = np.ones(self.max_avail_actions)

        return self.observation

    def step(self, action):
        val = np.random.uniform(low=0.0, high=1.0)
        if val <= self.probabilities[action]:
            reward = 1.0
        else:
            reward = 0.0

        info = {}
        self.current_draw += 1
        if self.current_draw == self.number_of_draws:
            self.done = True

        self.observation = np.ones(self.max_avail_actions)+0.01*np.random.randn(self.max_avail_actions)

        # print(self.current_draw, self.observation, self.done, info,f'reward: {reward}, action={action} val={val}')
        return self.observation, reward, self.done, info


if __name__ == "__main__":
    def env_creator(env_config={}):
        return BanditEnv(probabilities=[0.1, 0.5, 0.6], number_of_draws=50)  # return an env instance


    register_env("my_env", env_creator)

    tune.run("DQN",
             # algorithm specific configuration
             config={"env": "my_env",  #
                     "framework": "tf2",
                     "num_gpus": 0,
                     "num_workers": 63,
                     # "model": {"custom_model": "pa_model", },
                     # "evaluation_interval": 1,
                     # "evaluation_num_episodes": 5
                     },
             local_dir="cartpole_v1",  # directory to save results
             checkpoint_freq=2,  # frequency between checkpoints
             keep_checkpoints_num=6,
             )
    """
    env=BanditEnv()
    for _ in range(1000):
        action=env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(env.current_draw, observation, done, info, f'reward: {reward}, action={action}')

        if done:
            observation = env.reset()
            break

    env.close()

"""