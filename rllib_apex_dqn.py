import pprint
from ray import tune
import ray
from ray.rllib.algorithms.apex_dqn import APEX_DEFAULT_CONFIG
from ray.rllib.algorithms.apex_dqn.apex_dqn import ApexDQN as ApexTrainer

if __name__ == '__main__':
    config = APEX_DEFAULT_CONFIG.copy()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    config['env'] = "CartPole-v0"
    config['num_gpus'] = 0
    config['num_workers'] = 5
    config['evaluation_num_workers'] = 10
    config['evaluation_interval'] = 1
    config['learning_starts'] = 5000
    tune.run(ApexTrainer, config=config)