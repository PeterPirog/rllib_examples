import random

import gym
import numpy as np
from gym.spaces import Box, Dict, Discrete
from ray import tune
from ray.tune.registry import register_env


class ParametricActionsCartPole(gym.Env):
    """Parametric action version of CartPole.
    In this env there are only ever two valid actions, but we pretend there are
    actually up to `max_avail_actions` actions that can be taken, and the two
    valid actions are randomly hidden among this set.
    At each step, we emit a dict of:
        - the actual cart observation
        - a mask of valid actions (e.g., [0, 0, 1, 0, 0, 1] for 6 max avail)
        - the list of action embeddings (w/ zeroes for invalid actions) (e.g.,
            [[0, 0],
             [0, 0],
             [-0.2322, -0.2569],
             [0, 0],
             [0, 0],
             [0.7878, 1.2297]] for max_avail_actions=6)
    In a real environment, the actions embeddings would be larger than two
    units of course, and also there would be a variable number of valid actions
    per step instead of always [LEFT, RIGHT].
    """

    def __init__(self, max_avail_actions):
        # Use simple random 2-unit action embeddings for [LEFT, RIGHT]
        self.left_action_embed = np.random.randn(2)
        self.right_action_embed = np.random.randn(2)
        self.action_space = Discrete(max_avail_actions)
        self.wrapped = gym.make("CartPole-v0")
        self.observation_space = Dict(
            {
                "action_mask": Box(0, 1, shape=(max_avail_actions,), dtype=np.float32),
                "avail_actions": Box(-10, 10, shape=(max_avail_actions, 2)),
                "cart": self.wrapped.observation_space,
            }
        )
        self._skip_env_checking = True

    def update_avail_actions(self):
        self.action_assignments = np.array(
            [[0.0, 0.0]] * self.action_space.n, dtype=np.float32
        )
        #print(f'self.action_assignments={self.action_assignments}')
        self.action_mask = np.array([0.0] * self.action_space.n, dtype=np.float32)
        self.left_idx, self.right_idx = random.sample(range(self.action_space.n), 2)
        self.action_assignments[self.left_idx] = self.left_action_embed
        self.action_assignments[self.right_idx] = self.right_action_embed
        self.action_mask[self.left_idx] = 1
        self.action_mask[self.right_idx] = 1

    def reset(self):
        self.update_avail_actions()
        return {
            "action_mask": self.action_mask,
            "avail_actions": self.action_assignments,
            "cart": self.wrapped.reset(),
        }

    def step(self, action):
        if action == self.left_idx:
            actual_action = 0
        elif action == self.right_idx:
            actual_action = 1
        else:
            raise ValueError(
                "Chosen action was not one of the non-zero action embeddings",
                action,
                self.action_assignments,
                self.action_mask,
                self.left_idx,
                self.right_idx,
            )
        orig_obs, rew, done, info = self.wrapped.step(actual_action)
        self.update_avail_actions()
        self.action_mask = self.action_mask.astype(np.float32)
        obs = {
            "action_mask": self.action_mask,
            "avail_actions": self.action_assignments,
            "cart": orig_obs,
        }
        return obs, rew, done, info


class ParametricActionsCartPoleNoEmbeddings(gym.Env):
    """Same as the above ParametricActionsCartPole.
    However, action embeddings are not published inside observations,
    but will be learnt by the model.
    At each step, we emit a dict of:
        - the actual cart observation
        - a mask of valid actions (e.g., [0, 0, 1, 0, 0, 1] for 6 max avail)
        - action embeddings (w/ "dummy embedding" for invalid actions) are
          outsourced in the model and will be learned.
    """

    def __init__(self, max_avail_actions):
        # Randomly set which two actions are valid and available.
        self.left_idx, self.right_idx = random.sample(range(max_avail_actions), 2)
        self.valid_avail_actions_mask = np.array(
            [0.0] * max_avail_actions, dtype=np.float32
        )
        self.valid_avail_actions_mask[self.left_idx] = 1
        self.valid_avail_actions_mask[self.right_idx] = 1
        self.action_space = Discrete(max_avail_actions)
        self.wrapped = gym.make("CartPole-v0")
        self.observation_space = Dict(
            {
                "valid_avail_actions_mask": Box(0, 1, shape=(max_avail_actions,)),
                "cart": self.wrapped.observation_space,
            }
        )
        self._skip_env_checking = True

    def reset(self):
        return {
            "valid_avail_actions_mask": self.valid_avail_actions_mask,
            "cart": self.wrapped.reset(),
        }

    def step(self, action):
        if action == self.left_idx:
            actual_action = 0
        elif action == self.right_idx:
            actual_action = 1
        else:
            raise ValueError(
                "Chosen action was not one of the non-zero action embeddings",
                action,
                self.valid_avail_actions_mask,
                self.left_idx,
                self.right_idx,
            )
        orig_obs, rew, done, info = self.wrapped.step(actual_action)
        obs = {
            "valid_avail_actions_mask": self.valid_avail_actions_mask,
            "cart": orig_obs,
        }
        return obs, rew, done, info


if __name__ == "__main__":
    """
    env = ParametricActionsCartPoleNoEmbeddings(max_avail_actions=6)
    state, reward = env.reset()
    # state, reward, done, _ = env.step(0) 
    print(state, reward)

    print(np.array([[0.0, 0.0]] * 6, dtype=np.float32))
    """




    def env_creator(env_config={}):
        return ParametricActionsCartPoleNoEmbeddings(max_avail_actions=6)  # return an env instance


    register_env("my_env", env_creator)
    #trainer = ppo.PPOTrainer(env="my_env")

    #register_env("my_env", lambda config: YourExternalEnv(config))



    tune.run("PPO",
             # algorithm specific configuration
             config={"env": "my_env",
                     "num_gpus": 0,
                     "num_workers": 7,
                     "evaluation_interval": 2,
                     "evaluation_num_episodes": 20},
             local_dir="cartpole_v1",  # directory to save results
             checkpoint_freq=2,  # frequency between checkpoints
             keep_checkpoints_num=6, )
