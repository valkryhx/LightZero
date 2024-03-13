
import gymnasium as gym
import numpy as np
#import pygame
#Maze config
#from zoo.classic_control.grid.grid.grid import GridEnv  这是错误的导入 下面这个才是正确的 注意本文件的路径与grid包是平级 所以直接用grid.grid
from grid.grid import GridEnv

from gymnasium.envs.registration import register

register(
     id="MyGrid-v1",
     #entry_point="zoo.classic_control.grid.grid.grid:GridEnv", 这是错误的导入 下面这个才是正确的 注意本文件的路径与grid包是平级 所以直接用grid.grid
     entry_point="grid.grid:GridEnv",
     max_episode_steps=5,#300,
)

#maze = [
#    ['S', '', '.', '.'],
#    ['.', '#', '.', '#'],
#    ['.', '.', '.', '.'],
#    ['#', '.', '#', 'G'],
#]
# Test the environment
env = gym.make('MyGrid-v1')
obs = env.reset()
env.render()
print(f'h_score={env.h_score}')
done = False
i=0
print(f'\n\nSTART TO PLAY====\n')
while i<10:
    #pygame.event.get()
    
    #action = env.action_space.sample()  # Random action selection
    action = env.random_action()
    print(f'idx={i}, action={action}')
    obs, reward, done, _,_ = env.step(action)
    print(f'h_score={env.h_score}')
    env.render()
    print('Reward:', reward)
    print('Done:', done)
    
    #pygame.time.wait(200)  
    i += 1
    if done:
        
        break
    