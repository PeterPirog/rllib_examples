import ray
from ray import tune
from ray.rllib.contrib.alpha_zero.environments.cartpole import CartPole
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
from ray.rllib.models.catalog import ModelCatalog

if __name__ == '__main__':
    ray_num_cpus=64 # number of CPU available for ray
    num_workers=60
    training_iteration=500
    ray.init(num_cpus=ray_num_cpus)

    ModelCatalog.register_custom_model("dense_model", DenseModel)

    tune.run(
        "contrib/AlphaZero",
        stop={"training_iteration": training_iteration},
        max_failures=0,
        local_dir="cartpole_v1",  # directory to save results
        checkpoint_freq=2,  # frequency between checkpoints
        keep_checkpoints_num=3,
        config={
            "env": CartPole,
            "num_workers": num_workers,
            "rollout_fragment_length": 50,
            "train_batch_size": 500,
            "sgd_minibatch_size": 64,
            "lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "num_sgd_iter": 1,
            "mcts_config": {
                "puct_coefficient": 1.5,
                "num_simulations": 100,
                "temperature": 1.0,
                "dirichlet_epsilon": 0.20,
                "dirichlet_noise": 0.03,
                "argmax_tree_policy": False,
                "add_dirichlet_noise": True,
            },
            "ranked_rewards": {
                "enable": True,
            },
            "model": {
                "custom_model": "dense_model",
            },
        },
    )
    """
    tune.run("contrib/AlphaZero",
             config={"env": "CartPole-v1",
                     "evaluation_interval": 2,
                     "evaluation_num_episodes": 20},
             #resources_per_trial={"cpu": 8, "gpu": 0},
             local_dir="cartpole_v1",  # directory to save results
             checkpoint_freq=2,  # frequency between checkpoints
             keep_checkpoints_num=3, )
"""
# tensorboard --bind_all --port=12301 --logdir /home/ppirog/projects/rllib_examples/examples/03_alpha_cartpole/cartpole_v1/contrib
# web browser address: http://212.109.128.252:12301/#scalars