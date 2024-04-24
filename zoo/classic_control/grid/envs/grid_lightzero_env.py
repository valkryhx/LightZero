import copy
from datetime import datetime
from typing import Union, Optional, Dict
import logging
import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs import ObsPlusPrevActRewWrapper
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from zoo.classic_control.grid.grid.grid import GridEnv

from gymnasium.envs.registration import register

grid_size = 10
register(
     id="MyGrid-v1",
     entry_point="zoo.classic_control.grid.grid.grid:GridEnv",
     max_episode_steps=grid_size//2,#300,
)



@ENV_REGISTRY.register('mygrid_lightzero')
class MyGridEnv(BaseEnv):
    """
    LightZero version of the classic CartPole environment. This class includes methods for resetting, closing, and
    stepping through the environment, as well as seeding for reproducibility, saving replay videos, and generating random
    actions. It also includes properties for accessing the observation space, action space, and reward space of the
    environment.
    """

    config = dict(
        # env_id (str): The name of the environment.
        env_id="MyGrid-v1",
        # replay_path (str): The path to save the replay video. If None, the replay will not be saved.
        # Only effective when env_manager.type is 'base'.
        replay_path=None,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = {}) -> None:  
        """
        Initialize the environment with a configuration dictionary. Sets up spaces for observations, actions, and rewards.
        """
        self._cfg = cfg
        self._init_flag = False
        self._continuous = False
        self._replay_path = cfg.replay_path
        #self._observation_space = gym.spaces.Box(
        #    low=np.array([-4.8, float("-inf"), -0.42, float("-inf")]),
        #    high=np.array([4.8, float("inf"), 0.42, float("inf")]),
        #    shape=(4,),
        #    dtype=np.float32
        #)
        #self._observation_space=gym.spaces.Box(low=0.0,high=2.0,shape=(4,4,1),dtype=np.float32)
        self._observation_space=gym.spaces.Box(low=-1.0,high=1.0,shape=(3,grid_size,grid_size),dtype=np.float32)
        self._action_space = gym.spaces.Discrete(grid_size*grid_size)
        self._action_space.seed(0)  # default seed
        self._reward_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment. If it hasn't been initialized yet, this method also handles that. It also handles seeding
        if necessary. Returns the first observation.
        """
        if not self._init_flag:
            self._env = gym.make('MyGrid-v1',render_mode="rgb_array")
            if self._replay_path is not None:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                video_name = f'{self._env.spec.id}-video-{timestamp}'
                self._env = gym.wrappers.RecordVideo(
                    self._env,
                    video_folder=self._replay_path,
                    episode_trigger=lambda episode_id: True,
                    name_prefix=video_name
                )
            if hasattr(self._cfg, 'obs_plus_prev_action_reward') and self._cfg.obs_plus_prev_action_reward:
                self._env = ObsPlusPrevActRewWrapper(self._env)
            self._init_flag = True
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._seed = self._seed + np_seed
            self._action_space.seed(self._seed)
            obs, _ = self._env.reset(seed=self._seed)
        elif hasattr(self, '_seed'):
            self._action_space.seed(self._seed)
            obs, _ = self._env.reset(seed=self._seed)
        else:
            obs, _ = self._env.reset()
        self._observation_space = self._env.observation_space
        self._eval_episode_return = 0
        obs = to_ndarray(obs)
        #print(f'obs shape={obs.shape}')
        
        #action_mask = np.ones(self.action_space.n, 'int8')
        # 参考 gomoku_env.py的定义
        action_mask = np.zeros(grid_size*grid_size, 'int8')
        action_mask[self.legal_actions] = 1 # 
        #print(f'self.legal_actions={self.legal_actions}')
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return obs

    def step(self, action: Union[int, np.ndarray]) -> BaseEnvTimestep:
        """
        Overview:
            Perform a step in the environment using the provided action, and return the next state of the environment.
            The next state is encapsulated in a BaseEnvTimestep object, which includes the new observation, reward,
            done flag, and info dictionary.
        Arguments:
            - action (:obj:`Union[int, np.ndarray]`): The action to be performed in the environment. If the action is
              a 1-dimensional numpy array, it is squeezed to a 0-dimension array.
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): An object containing the new observation, reward, done flag,
              and info dictionary.
        .. note::
            - The cumulative reward (`_eval_episode_return`) is updated with the reward obtained in this step.
            - If the episode ends (done is True), the total reward for the episode is stored in the info dictionary
              under the key 'eval_episode_return'.
            - An action mask is created with ones, which represents the availability of each action in the action space.
            - Observations are returned in a dictionary format containing 'observation', 'action_mask', and 'to_play'.
        """
        if action not in self.legal_actions:
            logging.warning(
                f"Illegal action: {action}. Legal actions: {self.legal_actions}. "
                "Choosing a random action from legal actions."
            )
            action = numpy.random.choice(self.legal_actions)
            #print(f'********step_legal_actions={self.legal_actions} and new_step_action={action}')
         
        if isinstance(action, np.ndarray) and action.shape == (1,):
            action = action.squeeze()  # 0-dim array

        obs, rew, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated

        self._eval_episode_return += rew
        if done:
            info['eval_episode_return'] = self._eval_episode_return

        #action_mask = np.ones(self.action_space.n, 'int8')
        # 参考 gomoku_env.py的def _player_step定义 实时更新action_mask 
        action_mask = np.zeros(grid_size*grid_size, 'int8')
        action_mask[self.legal_actions] = 1 # 
        obs = {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

        return BaseEnvTimestep(obs, rew, done, info)

    def close(self) -> None:
        """
        Close the environment, and set the initialization flag to False.
        """
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the seed for the environment's random number generator. Can handle both static and dynamic seeding.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        """
        Enable the saving of replay videos. If no replay path is given, a default is used.
        """
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def random_action(self) -> np.ndarray:
        """
         Generate a random action using the action space's sample method. Returns a numpy array containing the action.
         """
        # 在legal actions中随机选一个 而不是从所有actions中随机选
        # 参考 gomoku_env.py
        action_list = self.legal_actions
        return np.random.choice(action_list)
        #random_action = self.action_space.sample()
        #random_action = to_ndarray([random_action], dtype=np.int64)
        #return random_action


    @property
    def legal_actions(self):
        # 加上@property 可以将legal_actions() 当作属性直接这么用 self.legal_actions
        #return np.arange(self._action_space.n)
        #return np.array(self._env.legal_actions(),dtype=np.int64
        # unwrapped 来源 跟加了@property 注解有关       会报warning
        # """WARN: env.legal_actions to get variables from other wrappers is deprecated and will be removed in v1.0, 
        #to get this variable you can do `env.unwrapped.legal_actions` for environment variables or 
        #`env.get_attr('legal_actions')` that will search the reminding wrappers."""

        return self._env.unwrapped.legal_actions

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        Property to access the observation space of the environment.
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Property to access the action space of the environment.
        """
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        """
        Property to access the reward space of the environment.
        """
        return self._reward_space

    def __repr__(self) -> str:
        """
        String representation of the environment.
        """
        return "LightZero MyGrid-v1 Env"
