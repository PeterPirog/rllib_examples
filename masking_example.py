import csv
from datetime import datetime

import gym
import numpy as np
import ray
from gym.spaces import Box
from gym.spaces import Dict
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG
from ray.rllib.agents.dqn.apex import ApexTrainer
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_tf
from ray.tune.logger import pretty_print

tf1, tf, tfv = try_import_tf()

DEMO_DATA_DIR = "mcar-out"

class MountainCar(gym.Env):
    def __init__(self, env_config={}):
        self.wrapped = gym.make("MountainCar-v0")
        self.action_space = self.wrapped.action_space
        self.t = 0
        self.reward_fun = env_config.get("reward_fun")
        self.lesson = env_config.get("lesson")
        self.use_action_masking = env_config.get("use_action_masking", False)
        self.action_mask = None
        self.reset()
        if self.use_action_masking:
            self.observation_space = Dict(
                {
                    "action_mask": Box(0, 1, shape=(self.action_space.n,)),
                    "actual_obs": self.wrapped.observation_space,
                }
            )
        else:
            self.observation_space = self.wrapped.observation_space

    def _get_obs(self):
        raw_obs = np.array(self.wrapped.unwrapped.state)
        if self.use_action_masking:
            self.update_avail_actions()
            obs = {
                "action_mask": self.action_mask,
                "actual_obs": raw_obs,
            }
        else:
            obs = raw_obs
        return obs

    def reset(self):
        self.wrapped.reset()
        self.t = 0
        self.wrapped.unwrapped.state = self._get_init_conditions()
        obs = self._get_obs()
        return obs

    def _get_init_conditions(self):
        if self.lesson == 0:
            low = 0.1
            high = 0.4
            velocity = self.wrapped.np_random.uniform(
                low=0, high=self.wrapped.max_speed
            )
        elif self.lesson == 1:
            low = -0.4
            high = 0.1
            velocity = self.wrapped.np_random.uniform(
                low=0, high=self.wrapped.max_speed
            )
        elif self.lesson == 2:
            low = -0.6
            high = -0.4
            velocity = self.wrapped.np_random.uniform(
                low=-self.wrapped.max_speed, high=self.wrapped.max_speed
            )
        elif self.lesson == 3:
            low = -0.6
            high = -0.1
            velocity = self.wrapped.np_random.uniform(
                low=-self.wrapped.max_speed, high=self.wrapped.max_speed
            )
        elif self.lesson == 4 or self.lesson is None:
            low = -0.6
            high = -0.4
            velocity = 0
        else:
            raise ValueError
        obs = (self.wrapped.np_random.uniform(low=low, high=high), velocity)
        return obs

    def set_lesson(self, lesson):
        self.lesson = lesson

    def step(self, action):
        self.t += 1
        state, reward, done, info = self.wrapped.step(action)
        if self.reward_fun == "custom_reward":
            position, velocity = state
            reward += (abs(position + 0.5) ** 2) * (position > -0.5)
        obs = self._get_obs()
        if self.t >= 200:
            done = True
        return obs, reward, done, info

    def update_avail_actions(self):
        self.action_mask = np.array([1.0] * self.action_space.n)
        pos, vel = self.wrapped.unwrapped.state
        # 0: left, 1: no action, 2: right
        if (pos < -0.3) and (pos > -0.8) and (vel < 0) and (vel > -0.05):
            self.action_mask[1] = 0
            self.action_mask[2] = 0




class ParametricActionsModel(DistributionalQTFModel):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        true_obs_shape=(2,),
        **kw
    ):
        super(ParametricActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw
        )
        self.action_value_model = FullyConnectedNetwork(
            Box(-1, 1, shape=true_obs_shape),
            action_space,
            num_outputs,
            model_config,
            name + "_action_values",
        )
        self.register_variables(self.action_value_model.variables())

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        action_values, _ = self.action_value_model(
            {"obs": input_dict["obs"]["actual_obs"]}
        )
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return action_values + inf_mask, state

#from custom_mcar import MountainCar
#from masking_model import ParametricActionsModel
#from mcar_demo import DEMO_DATA_DIR

ALL_STRATEGIES = [
    "default",
    "with_dueling",
    "custom_reward",
    "custom_reward_n_dueling",
    "demonstration",
    "curriculum",
    "curriculum_n_dueling",
    "action_masking",
]



def get_apex_trainer(strategy):
    config = APEX_DEFAULT_CONFIG.copy()
    config["env"] = MountainCar
    config["buffer_size"] = 1000000
    config["learning_starts"] = 10000
    config["target_network_update_freq"] = 50000
    config["rollout_fragment_length"] = 200
    config["timesteps_per_iteration"] = 10000
    config["num_gpus"] = 0  # 1
    config["num_workers"] = 20
    config["evaluation_num_workers"] = 10
    config["evaluation_interval"] = 1
    if strategy not in [
        "with_dueling",
        "custom_reward_n_dueling",
        "curriculum_n_dueling",
    ]:
        config["hiddens"] = []
        config["dueling"] = False

    if strategy == "action_masking":
        ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
        config["env_config"] = {"use_action_masking": True}
        config["model"] = {
            "custom_model": "pa_model",
        }
    elif strategy == "custom_reward" or strategy == "custom_reward_n_dueling":
        config["env_config"] = {"reward_fun": "custom_reward"}
    elif strategy in ["curriculum", "curriculum_n_dueling"]:
        config["env_config"] = {"lesson": 0}
    elif strategy == "demonstration":
        config["input"] = DEMO_DATA_DIR
        # config["input"] = {"sampler": 0.7, DEMO_DATA_DIR: 0.3}
        config["explore"] = False
        config["input_evaluation"] = []
        config["n_step"] = 1

    trainer = ApexTrainer(config=config)
    return trainer, config["env_config"]


def set_trainer_lesson(trainer, lesson):
    trainer.evaluation_workers.foreach_worker(
        lambda ev: ev.foreach_env(lambda env: env.set_lesson(lesson))
    )
    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(lambda env: env.set_lesson(lesson))
    )


def increase_lesson(lesson):
    if lesson < CURRICULUM_MAX_LESSON:
        lesson += 1
    return lesson


def final_evaluation(trainer, n_final_eval, env_config={}):
    env = MountainCar(env_config)
    eps_lengths = []
    for i_episode in range(n_final_eval):
        observation = env.reset()
        done = False
        t = 0
        while not done:
            t += 1
            action = trainer.compute_action(observation)
            observation, reward, done, info = env.step(action)
            if done:
                eps_lengths.append(t)
                print(f"Episode finished after {t} time steps")
    print(
        f"Avg. episode length {np.mean(eps_lengths)} out of {len(eps_lengths)} episodes."
    )
    return np.mean(eps_lengths)

STRATEGY = "action_masking"
CURRICULUM_MAX_LESSON = 4
CURRICULUM_TRANS = 150
MAX_STEPS = 2e6
MAX_STEPS_OFFLINE = 4e5
NUM_TRIALS = 5
NUM_FINAL_EVAL_EPS = 20


if __name__ == "__main__":
    ### START TRAINING ###
    ray.init()
    avg_eps_lens = []
    for i in range(NUM_TRIALS):
        trainer, env_config = get_apex_trainer(STRATEGY)
        if STRATEGY in ["curriculum", "curriculum_n_dueling"]:
            lesson = 0
            set_trainer_lesson(trainer, lesson)
        # Training
        while True:
            results = trainer.train()
            print(pretty_print(results))
            if STRATEGY == "demonstration":
                demo_training_steps = results["timesteps_total"]
                if results["timesteps_total"] >= MAX_STEPS_OFFLINE:
                    trainer, _ = get_apex_trainer("with_dueling")
            if results["timesteps_total"] >= MAX_STEPS:
                if STRATEGY == "demonstration":
                    if results["timesteps_total"] >= MAX_STEPS + demo_training_steps:
                        break
                else:
                    break
            if "evaluation" in results and STRATEGY in ["curriculum", "curriculum_n_dueling"]:
                if results["evaluation"]["episode_len_mean"] < CURRICULUM_TRANS:
                    lesson = increase_lesson(lesson)
                    set_trainer_lesson(trainer, lesson)
                    print(f"Lesson: {lesson}")

        # Final evaluation
        checkpoint = trainer.save()
        if STRATEGY in ["curriculum", "curriculum_n_dueling"]:
            env_config["lesson"] = CURRICULUM_MAX_LESSON
        if STRATEGY == "action_masking":
            # Action masking is running into errors in Ray 1.0.1 during compute action
            # So, we use evaluation episode lengths.
            avg_eps_len = results["evaluation"]["episode_len_mean"]
        else:
            avg_eps_len = final_evaluation(trainer, NUM_FINAL_EVAL_EPS, env_config)
        date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        result = [date_time, STRATEGY, str(i), avg_eps_len, checkpoint]
        avg_eps_lens.append(avg_eps_len)
        with open(r"results.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(result)
    print(f"Average episode length: {np.mean(avg_eps_lens)}")
