

import numpy as np
import pygame
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import random 

class MazeGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array",None], "render_fps": 4}
    def __init__(self):
        super(MazeGameEnv, self).__init__()
            
        self.maze = np.zeros([4,4])  # Maze represented as a 2D numpy array
        self.maze[3,3]=2
        self.start_pos = [0,0]#np.where(self.maze == 'S')  # Starting position
        self.goal_pos = [3,3]#np.where(self.maze == 'G')  # Goal position
        self.current_pos = self.start_pos #starting position is current posiiton of agent
        self.num_rows, self.num_cols = self.maze.shape

        # 4 possible actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)  

        # Observation space is grid of size:rows x columns
        #self.observation_space = spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols)))
        self.observation_space = spaces.Box(low=0.0,high=2.0,shape=(1,4,4))

        # Initialize Pygame
        #pygame.init()
        #self.cell_size = 125

        # setting display size
        #self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.current_pos = self.start_pos
        return self.maze ,{}

    def step(self, action):
        # Move the agent based on the selected action
        new_pos = np.array(self.current_pos)
        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Left
            new_pos[1] -= 1
        elif action == 3:  # Right
            new_pos[1] += 1

        # Check if the new position is valid
        if self._is_valid_position(new_pos):
            self.current_pos = new_pos

        # Reward function
        if np.array_equal(self.current_pos, self.goal_pos):
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False

        return self._get_obs(), reward, done, False, {}

    def _is_valid_position(self, pos):
        row, col = pos
   
        # If agent goes out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False

        # If the agent hits an obstacle
        if self.maze[row, col] == '#':
            return False
        return True

    def render2(self):
        # Clear the screen
        self.screen.fill((255, 255, 255))  

        # Draw env elements one cell at a time
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size
            
                try:
                    print(np.array(self.current_pos)==np.array([row,col]).reshape(-1,1))
                except Exception as e:
                    print('Initial state')

                if self.maze[row, col] == '#':  # Obstacle
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'S':  # Starting position
                    pygame.draw.rect(self.screen, (0, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'G':  # Goal position
                    pygame.draw.rect(self.screen, (255, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))

                if np.array_equal(np.array(self.current_pos), np.array([row, col]).reshape(-1,1)):  # Agent position
                    pygame.draw.rect(self.screen, (0, 0, 255), (cell_left, cell_top, self.cell_size, self.cell_size))

        pygame.display.update()  # Update the display
    def _get_obs(self):
        self.show = deepcopy(self.maze)
        self.show[self.current_pos[0],self.current_pos[1]] = 1
        #self.show
        print(self.show.dtype)
        return [self.show]  
    def render(self):
        print(self._get_obs())        

        
    def close2(self):
        pygame.display.quit()
        pygame.quit()