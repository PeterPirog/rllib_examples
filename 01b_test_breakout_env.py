"""
This is a simple example how to run open ai breakout environment, this environment is usefull to work with images

Environment details are here: https://www.gymlibrary.dev/environments/atari/breakout/
Basic explanation here: https://www.gymlibrary.dev/content/basic_usage/

If You can't run this code plese do not go to the next examples
"""

import gym
import random
import numpy as np

# env = gym.make("ALE/Breakout-v5",render_mode='human')  # A
env = gym.make("ALE/Breakout-v5")  # A

print(f'Action space:{env.action_space}')
print(f'Observation space: {env.observation_space}')


def basic_policy(obs):  # B
    return random.randint(0, 3)


totals = []
if __name__ == "__main__":
    for episode in range(50):
        episode_rewards = 0
        obs = env.reset()
        for step in range(200):
            action = basic_policy(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                break
        totals.append(episode_rewards)
    env.close()
print(f'Results:')
print(f' mean:{np.mean(totals)}\n'
      f' standard deviation:{np.std(totals)}\n'
      f' the worst score (min):{np.min(totals)}\n'
      f' the best score(max):{np.max(totals)}\n')

"""
A - Breakout is the name of  atari game environment from gym library:
    https://www.gymlibrary.dev/environments/atari/breakout/ , use env = gym.make("ALE/Breakout-v5",render_mode='human')
    to show how the game looks

B - bacic_policy it's the function which decide what action should be taken in specified situation
    in this case the result is random action ( integer from range from 0 to 3)
"""

