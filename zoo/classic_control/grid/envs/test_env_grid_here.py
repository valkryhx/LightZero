
import gymnasium as gym
import numpy as np
import pygame
#Maze config
from zoo.classic_control.grid.grid.grid import GridEnv
from zoo.classic_control.grid.envs.grid_lightzero_env import MyGridEnv
from gymnasium.envs.registration import register

config = EasyDict(dict(
    replay_path=None,
))
# Test the environment
env = MyGridEnv()
obs = env.reset()
env.render()

done = False
i=0
print(f'\n\n====START TO PLAY====\n')
while i<10:
    #pygame.event.get()
    
    action = env.random_action()
    print(f'idx={i}, action={action}')
    obs, rew, done, info = env.step(action)
    env.render()
    print('Reward:', reward)
    print('Done:', done)
    
    #pygame.time.wait(200)  
    i += 1