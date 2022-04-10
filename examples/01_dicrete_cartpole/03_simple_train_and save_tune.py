from ray import tune

if __name__ == '__main__':
    analysis = tune.run(run_or_experiment="PPO",
             stop={"episode_reward_mean": 200},
             config={"env": "CartPole-v1",
                     "num_gpus": 0,
                     "num_workers": 20,
                     "lr": tune.grid_search([0.01, 0.001, 0.0001]),
                     "evaluation_interval": 2,
                     "evaluation_num_episodes": 20},
             #resources_per_trial={"cpu": 8, "gpu": 0},
             local_dir="cartpole_v1",  # directory to save results
             checkpoint_freq=2,  # frequency between checkpoints
             keep_checkpoints_num=3, )

# tensorboard --bind_all --port=12301 --logdir /home/ppirog/projects/rllib_examples/examples/01_dicrete_cartpole/cartpole_v1/PPO
# web browser address: http://212.109.128.252:12301/#scalars
"""
ray.init()
tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        "num_gpus": 0,
        "num_workers": 1,
        "lr": tune.grid_search([0.01, 0.001, 0.0001]),
    },
)
"""