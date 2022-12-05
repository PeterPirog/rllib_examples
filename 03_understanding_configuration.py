
# Common parameters:
# https://docs.ray.io/en/latest/rllib/rllib-training.html?highlight=common%20parameters#common-parameters

import pprint
pp=pprint.PrettyPrinter(indent=4)
from ray import tune
from ray.rllib.agents import ppo

if __name__ == "__main__":
    print(dir(tune))
    config=ppo.DEFAULT_CONFIG
    pp.pprint(config)
   # pp.pprint(tune.TuneConfig.)