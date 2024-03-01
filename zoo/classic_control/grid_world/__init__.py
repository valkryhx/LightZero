from mazegame import MazeGameEnv
from gymnasium.envs.registration import register

register(
     id="MyMaze-v1",
     entry_point="mazegame:MazeGameEnv",
     max_episode_steps=300,
)
