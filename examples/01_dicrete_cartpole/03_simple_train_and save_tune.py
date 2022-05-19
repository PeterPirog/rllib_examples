from ray import tune

if __name__ == '__main__':
    #search_alg =  hyperopt_search = HyperOptSearch(metric="episode_reward_mean", mode="max") # metric="episode_reward_mean"
    analysis = tune.run(run_or_experiment="PPO",
             stop={"episode_reward_mean": 500},
             config={"env": "CartPole-v1",
                     "framework": "tf2",
                     "num_gpus": 0,
                     "num_workers": 4,
                     #"gamma": tune.grid_search([0.9, 0.95, 0.99]),
                     "gamma": 0.99,
                     "lr": tune.grid_search([0.01, 0.001, 0.0001]),
                     "evaluation_interval": 2,
                     "evaluation_num_episodes": 100,
                     "log_level": "WARN" #
                     },
             #resources_per_trial={"cpu": 8, "gpu": 0},
             local_dir="cartpole_v1",  # directory to save results
             checkpoint_freq=2,  # frequency between checkpoints
             keep_checkpoints_num=3,
             #search_alg=search_alg,
                        )

# tensorboard --bind_all --port=12301 --logdir /home/ppirog/projects/rllib_examples/examples/01_dicrete_cartpole/cartpole_v1/PPO
# web browser address: http://212.109.128.252:12301/#scalars
"""
 Paremeters of algorithm
https://docs.ray.io/en/latest/rllib/rllib-training.html#common-parameters

"""