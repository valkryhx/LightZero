#import grid_world
import gymnasium as gym
import numpy as np
import pygame
#Maze config
#from mazegame.mazegame import MazeGameEnv
import mazegame

maze = [
    ['S', '', '.', '.'],
    ['.', '#', '.', '#'],
    ['.', '.', '.', '.'],
    ['#', '.', '#', 'G'],
]
# Test the environment
env = gym.make('MyMaze-v1')
obs = env.reset()
env.render()

done = False
i=0
while i<4:
    #pygame.event.get()
    action = env.action_space.sample()  # Random action selection
    obs, reward, done, _,_ = env.step(action)
    env.render()
    print('Reward:', reward)
    print('Done:', done)
    
    #pygame.time.wait(200)  
    i += 1