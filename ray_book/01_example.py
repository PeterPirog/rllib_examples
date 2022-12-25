# base Mastering RL page 217
# https://docs.ray.io/en/latest/ray-air/trainer.html?highlight=evaluation_num_workers#trainer-basics

import pprint
from ray import tune
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo import PPOTrainer as Trainer

if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    config = DEFAULT_CONFIG.copy()

    pp.pprint(config)  # print default PPO configuration
    # set custom configuration
    config['env'] = 'CartPole-v1'
    config['framework'] = 'tf2'
    config['num_workers'] = 6
    config['evaluation_num_workers'] = 1
    config['evaluation_interval'] = 1


    print(config['env'])
    analysis=tune.run(Trainer, config=config)
