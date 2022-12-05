"""
This is a simple example how to run open ai cartpole environment which is reinforcement learning (RL) "Hello world" example
Environment details are here: https://www.gymlibrary.dev/environments/classic_control/cart_pole/
Basic explanation here: https://www.gymlibrary.dev/content/basic_usage/

If You can't run this code plese do not go to the next examples
"""

import gym
import numpy as np

env = gym.make('CartPole-v1')  # A

print(f'Action space:{env.action_space}')  # B
print(f'Observation space: {env.observation_space}')  # B


def basic_policy(obs):  # C
    angle = obs[2]
    return 0 if angle < 0 else 1


totals = []  # D
if __name__ == "__main__":
    for episode in range(500):  # E
        episode_rewards = 0  # F
        obs = env.reset()  # G
        for step in range(200):  # H
            action = basic_policy(obs)  # I
            obs, reward, done, info = env.step(action)  # J
            # env.render() # K
            episode_rewards += reward  # L
            if done:  # M
                break
        totals.append(episode_rewards)  # N
    # env.close()
print(f'Results:')  # O
print(f' mean:{np.mean(totals)}\n'
      f' standard deviation:{np.std(totals)}\n'
      f' the worst score (min):{np.min(totals)}\n'
      f' the best score(max):{np.max(totals)}\n')

"""
A - 'CartPole-v1' is the name of environment from gym library https://www.gymlibrary.dev/api/core/

B - method to check what is the shape and type of the data sent to the environment(action space)
    and what is the shape and type data received from environment (observation space),
    details here: https://www.gymlibrary.dev/content/basic_usage/#spaces

C - bacic_policy it's the function which decide what action should be taken in specified situation
    this function is the solution of the RL algorithm. In this example the function is very simple 
    to check if gym environment works
    
D - totals -is the list to save final results for all games

E - episode is the single game, if you play chess the episode is single match, so
    if you play chess 500 times  episode is equal 500
    
F - episode reward is the result after finish single game, for example if you play chess
    if you win the match episode reward is 1, if you lose episode reward is -1
    
G - if you finish the game you have to reset environment to the initial state,
    observation (obs) is the first observable state of the environment
    
H - this is the number of moves to finish the environment, for many environments (games)
    maximum number of steps is defined, after the last move the game is over or some condition of success or failure
    is met, fo some cases like temperature controller there is no the last step, you can regulate temperature as long
     as you want 
     
I - in this line of the code the policy (function to make decision) decide which action should be taken

J - in this line of the code, chosen action is sent to the environment and we get information what is reaction of the 
    environment after our actions:
    obs - is the new observable state of the environment, like new chess board position, new printscreen of the
    atari game
    reward - this is the reward after our taken action, in some cases reward is equal 0 because whe know if the action
    was good or bad after the last step, the situation when only the last reward is differ than 0 we call 
    "sparse rewards"
    
K - this method is only to print animation, if you have some problem to with it, comment this line and env.close()

L - this is the function to collect all rewards in single game (episode)

M - if the environment is finished (some player wins, success , failure  or step limit reached) done = True 

N - add the result from current game to the results from the other games(episodes) 

O - print results from all episodes

REMARKS:
1. Do not confuse episode with step, episode = whole single game, step = single move in single game
2. Do not confuse reward, episode rewards and totals:
    reward - reward for single move in single game, for example added points after move in atari game
    episode reward - reward after finishing single game
    totals - all rewards after finishing all games
3. Do not confuse observation and state
    - state is real state of the environment, if you play chess and you see board there is no additional data outside you
     see, the board is as you see
    - observation is a data available for you but this data many times  isn't complete
    for example if you play strategy game you know your units, building, etc. your part of the map
    but you don't know units, building and map of the other players
    Many times observation is equal state but not in all cases
"""
