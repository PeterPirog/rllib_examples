# https://docs.ray.io/en/latest/rllib/index.html
# This is example from page: https://www.youtube.com/watch?v=HteW2lfwLXM
# Ray RLlib: How to Use Deep RL Algorithms to Solve Reinforcement Learning Problems, Dibya Chakravorty


import ray
from ray import tune

if __name__ == "__main__":
    ray.init()
    analysis=tune.run("PPO",
             # https://docs.ray.io/en/latest/rllib/rllib-training.html#advanced-python-apis
             config={"env":"CartPole-v1",
                     #"evaluation_interval":2, # number of training iterations between evaluation
                     "framework": "tf2",
                     "seed": None,
                     "num_gpus": 0,
                     "num_workers": 4,
                     "evaluation_num_workers": 0,
                     "gamma": tune.grid_search([0.9, 0.95, 0.99]),
                     #"gamma": 0.99,
                     #"lr": tune.grid_search([0.01, 0.001, 0.0001]),
                     #"evaluation_num_episodes": 100,
                     #"log_level": "WARN" #
                     # other configuration parameters
                     })



"""
# Import the RL algorithm (Algorithm) we would like to use.
from ray.rllib.algorithms.ppo import PPO

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "CartPole-v1",
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 2,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "tf",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    },
}

# Create our RLlib Trainer.
algo = PPO(config=config)

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for _ in range(3):
    print(algo.train())

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
algo.evaluate()
"""