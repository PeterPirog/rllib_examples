import argparse
import os

import gym
import numpy as np
import ray
from gym.spaces import Box, Dict, Discrete, Tuple
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
# from export import export_all_models
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

tf1, tf, tfv = try_import_tf()

class ActionMaskingTupleCartPole(gym.Env):
  """Masking tuple action version of CartPole.

  In this env only the first dimension of actions is actually valid, the rest are just for demonstrating action
  masking in tuple action space

  At each step, we emit a dict of:
      - the actual state observation
      - a mask of valid actions, the length of which is the total length of all action dimensions
       (e.g., [0, 1, 1, 0, 0, 0, 1] for Tuple([Discrete(2), Discrete(5)])

  """

  def __init__(self, tuple_actions_shape):


      self.action_space = Tuple([Discrete(i) for i in tuple_actions_shape])
      if self.action_space[0] != Discrete(2):
          raise Exception("The first Action dimension must have 2 possible actions!")

      self.action_length = sum([i.n for i in self.action_space])
      self.wrapped = gym.make("CartPole-v0")
      self.observation_space = Dict({
          "action_mask": Box(0, 1, shape=(self.action_length, )),
          "state": self.wrapped.observation_space,
      })
      # Left move always at 0 index and right always at 1
      self.left_idx = 0
      self.right_idx = 1

  def update_avail_actions(self):
      self.action_mask = np.random.choice([0, 1], size=(self.action_length,))
      # Always have one of the two actions available
      self.action_mask[0] = 1

  def reset(self):
      self.update_avail_actions()
      return {
          "action_mask": self.action_mask,
          "state": self.wrapped.reset(),
      }

  def step(self, action: tuple):
      if action[0] == self.left_idx:
          actual_action = 0
      elif action[0] == self.right_idx:
          actual_action = 1
      else:
          raise ValueError(
              "Chosen action was outside the 'real' space",
              action, self.action_mask,
              self.left_idx, self.right_idx)
      orig_obs, rew, done, info = self.wrapped.step(actual_action)
      self.update_avail_actions()
      obs = {
          "action_mask": self.action_mask,
          "state": orig_obs,
      }
      return obs, rew, done, info


class ActionMaskingModel(TFModelV2):
  """
  Parametric action model that handles action masking.
  """

  def __init__(self,
               obs_space,
               action_space,
               num_outputs,
               model_config,
               name,
               true_obs_shape=(4, ),
               **kw):
      super(ActionMaskingModel, self).__init__(
          obs_space, action_space, num_outputs, model_config, name, **kw)
      self.action_embed_model = FullyConnectedNetwork(
          Box(-1, 1, shape=true_obs_shape), action_space, num_outputs,
          model_config, name + "_action_embed")
      self.register_variables(self.action_embed_model.variables())

  def forward(self, input_dict, state, seq_lens):
      # Extract the action mask tensor from the observation.
      action_mask = input_dict["obs"]["action_mask"]

      # Compute the predicted action embedding
      action_embed, _ = self.action_embed_model({
          "obs": input_dict["obs"]["state"]
      })

      # Mask out invalid actions (use tf.float32.min for stability)
      inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
      return action_embed + inf_mask, state

  def value_function(self):
      return self.action_embed_model.value_function()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    ray.init()

    register_env("am_cartpole", lambda _: ActionMaskingTupleCartPole([2, 5]))
    ModelCatalog.register_custom_model(
      "am_model", ActionMaskingModel)

    cfg = {}

    config = dict(
      {
          "env": "am_cartpole",
          "model": {
              "custom_model": "am_model",
          },
          # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
          "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
          "num_workers": 0,
          "framework": "tf",
      },
      **cfg)

    cls = get_agent_class(args.run)
    trainer_obj = cls(config=config)

    for _ in range(5):
      results = trainer_obj.train()
      print(pretty_print(results))