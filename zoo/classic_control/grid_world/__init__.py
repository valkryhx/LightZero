#from grid_world.mazegame import MazeGameEnv
from gymnasium.envs.registration import register

register(
     id="MyMaze-v1",
     entry_point="grid_world.mazegame:MazeGameEnv",
     max_episode_steps=300,
)
