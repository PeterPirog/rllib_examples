import random

import gym
import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete, Dict
from ray import tune
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

tf1, tf, tfv = try_import_tf()


class ParametricActionsCartPole(gym.Env):
    def __init__(self, max_avail_actions):
        # Randomly set which two actions are valid and available.
        self.left_idx, self.right_idx = random.sample(range(max_avail_actions), 2)
        print(f'self.left_idx={self.left_idx}, self.right_idx={self.right_idx}')
        self.valid_avail_actions_mask = np.array(
            [0.0] * max_avail_actions, dtype=np.float32
        )
        self.valid_avail_actions_mask[self.left_idx] = 1
        self.valid_avail_actions_mask[self.right_idx] = 1
        self.action_space = Discrete(max_avail_actions)
        self.wrapped = gym.make("CartPole-v0")
        self.observation_space = Dict(
            {
                "action_mask": Box(0, 1, shape=(max_avail_actions,)),
                "observations": self.wrapped.observation_space,
            }
        )
        self._skip_env_checking = True

    def reset(self):
        return {
            "action_mask": self.valid_avail_actions_mask,
            "observations": self.wrapped.reset(),
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
            "action_mask": self.valid_avail_actions_mask,
            "observations": orig_obs,
        }
        return obs, rew, done, info

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_embed, _ = self.action_embed_model({
            "obs": input_dict["obs"]["cart"]
        })

        # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
        # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
        intent_vector = tf.expand_dims(action_embed, 1)

        # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
        action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=2)

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state


class ActionMaskModel(TFModelV2):
    """Model that handles simple discrete action masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, Dict)
                and "action_mask" in orig_space.spaces
                and "observations" in orig_space.spaces
        )

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


if __name__ == "__main__":
    def env_creator(env_config={}):
        return ParametricActionsCartPole(max_avail_actions=6)  # return an env instance


    register_env("my_env", env_creator)

    ModelCatalog.register_custom_model("pa_model", ActionMaskModel)
    tune.run("PPO",
             # algorithm specific configuration
             config={"env": "my_env",  #
                     "framework": "tf2",
                     "num_gpus": 0,
                     "num_workers": 7,
                     "model": {"custom_model": "pa_model", },
                     "evaluation_interval": 2,
                     # "evaluation_num_episodes": 20
                     },
             local_dir="cartpole_v1",  # directory to save results
             checkpoint_freq=2,  # frequency between checkpoints
             keep_checkpoints_num=6,
             )

