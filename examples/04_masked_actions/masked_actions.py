# https://towardsdatascience.com/action-masking-with-rllib-5e4bec5e7505
# https://www.datahubbs.com/action-masking-with-rllib/

import numpy as np
import ray
from gym import spaces
from or_gym.utils import create_env
from ray import tune
from ray.rllib import agents
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

env_config = {'N': 5,
              'max_weight': 15,
              'item_weights': np.array([1, 12, 2, 1, 4]),
              'item_values': np.array([2, 4, 2, 1, 10]),
              'mask': True}


class KP0ActionMaskModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs,
                 model_config, name, true_obs_shape=(11,),
                 action_embed_size=5, *args, **kwargs):
        super(KP0ActionMaskModel, self).__init__(obs_space,
                                                 action_space, num_outputs, model_config, name,
                                                 *args, **kwargs)

        self.action_embed_model = FullyConnectedNetwork(
            spaces.Box(0, 1, shape=true_obs_shape),
            action_space, action_embed_size,
            model_config, name + "_action_embedding")
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"]["avail_actions"]

        action_mask = input_dict["obs"]["action_mask"]
        action_embedding, _ = self.action_embed_model({
            "obs": input_dict["obs"]["state"]})
        intent_vector = tf.expand_dims(action_embedding, 1)
        action_logits = tf.reduce_sum(avail_actions * intent_vector,
                                      axis=1)
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()


ModelCatalog.register_custom_model('kp_mask', KP0ActionMaskModel)


def register_env(env_name, env_config={}):
    env = create_env(env_name)
    tune.register_env(env_name, lambda env_name: env(env_name, env_config=env_config))


register_env('Knapsack-v0', env_config=env_config)

ray.init(ignore_reinit_error=True)
trainer_config = {
    "model": {
        "custom_model": "kp_mask"
        },
    "env_config": env_config
     }
trainer = agents.ppo.PPOTrainer(env='Knapsack-v0', config=trainer_config)

env = trainer.env_creator('Knapsack-v0')
state = env.state
print(f'state: {state}')
state['action_mask'][0] = 0

actions = np.array([trainer.compute_action(state) for i in range(10000)])
any(actions==0)

