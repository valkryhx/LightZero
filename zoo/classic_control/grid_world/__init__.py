from .grid_world import GridWorldEnv
from gymnasium.envs.registration import register

register(
     id="GridWorld-v0",
     entry_point="grid_world:GridWorldEnv",
     max_episode_steps=300,
)
