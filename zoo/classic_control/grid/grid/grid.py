

import numpy #as numpy
import pygame
from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import random 
grid_size = 10
class GridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array",None], "render_fps": 4}
    def __init__(self,render_mode='rgb_array'):
        super(GridEnv, self).__init__()
        #self.max_step=5    
        #self.maze = numpy.zeros([4,4],dtype=numpy.float32)  # Maze represented as a 2D numpy array
        #self.maze[3,3]=2.0
        #self.start_pos = [0,0]#numpy.where(self.maze == 'S')  # Starting position
        #self.goal_pos = [3,3]#numpy.where(self.maze == 'G')  # Goal position
        #self.current_pos = self.start_pos #starting position is current posiiton of agent
        #self.num_rows, self.num_cols = self.maze.shape
        
        self.size = grid_size# size        
        self.MARK_NEGATIVE = -1.0        
        self.agent_get_reward =0
        # 原始的action space为[0,100)
        
        # 每次step都会更新 _used_actions ，使用_actions - _used_actions - _invalid_actions，剩下的才是合法的action space
        
        # position reset
        self.position = None # [0, 0]
        self.pos_history = numpy.zeros([grid_size,grid_size])
        # grid reset
        #a_100 = list(range(1, grid_size*grid_size + 1))
        #random.shuffle(a_100)
        #self.grid = numpy.array(a_100).reshape(grid_size, grid_size) / len(a_100)  # numpy.random.random((10, 10))
        
        #numpy.random.seed(seed)
        self.grid = numpy.random.rand(grid_size,grid_size)#*10
        numpy.fill_diagonal(self.grid, self.MARK_NEGATIVE)
        # marked_position rest
        self.mark = numpy.zeros([grid_size,grid_size])
         
        self._used_actions=set([])
        # invalid actions 比如0 11,22,,,99
        self._invalid_actions = set([i for i in range(grid_size*grid_size) if i//grid_size == i%grid_size])
        # action space reset
        self._actions = set(range(grid_size * grid_size)) -self._invalid_actions
        # 4 possible actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(grid_size*grid_size)  

        # Observation space is grid of size:rows x columns
        #self.observation_space = spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols)))
        #self.observation_space = spaces.Box(low=0.0,high=2.0,shape=(4,4,1),dtype=numpy.float32)
        self.observation_space = spaces.Box(low=-1.0,high=1.0,shape=(3,grid_size,grid_size),dtype=numpy.float32)

        # Initialize Pygame
        #pygame.init()
        #self.cell_size = 125

        # setting display size
        #self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))
    def legal_actions(self):
        legal_actions = self._actions
        if self.position and len(self.position)>1:
            # for example self.position=[2,9]
            #chosen_action = self.position[0]*grid_size + self.position[1]
            marked_row_act_0 = set(range(self.position[0]*grid_size,(self.position[0]+1)*grid_size))
            marked_row_act_1 = set(range(self.position[1] * grid_size, (self.position[1] + 1) * grid_size))
            marked_col_act_0 = set([idx*grid_size + self.position[0] for idx in range(grid_size)])
            marked_col_act_1 = set([idx * grid_size + self.position[1] for idx in range(grid_size)])
            self._used_actions = self._used_actions | marked_row_act_0 | marked_row_act_1 | marked_col_act_0 | marked_col_act_1
            legal_actions = list(legal_actions -self._invalid_actions -  self._used_actions)
        #print(f'legal_actions={legal_actions}')
        return legal_actions #list(self._actions)
        
    def get_observation(self):
        #observation = numpy.zeros((self.size, self.size))
        #observation[self.position[0]][self.position[1]] = 1
        observation = [self.grid ,self.pos_history,self.mark]
        #observation = [self.grid]
        #observation = [self.grid ,self.pos_history,self.mark,self.pos_now,self.invalid_1,self.invalid_2,self.invalid_3,self.invalid_4]
        #observation = self.grid.flatten()
        # flatten 把二维3x3 拉成 单独的1维为9的numpy array
        #return observation.flatten()
        return numpy.array(observation,dtype=numpy.float32)
        #return [[observation]]   

        
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        #self.current_pos = self.start_pos
        #self.max_step =6
        #return self._get_obs() ,{}
        self.position = None # [0, 0]
        self.grid = numpy.random.rand(grid_size,grid_size)#*10
        numpy.fill_diagonal(self.grid, self.MARK_NEGATIVE)

        # marked_position reset
        
        # pos history 
        # marked_position rest
        self.mark = numpy.zeros([grid_size,grid_size])
        self.pos_history = numpy.zeros([grid_size,grid_size])
        #self.pos_now = numpy.zeros([grid_size,grid_size])
        #self.invalid_1 = numpy.zeros([grid_size,grid_size])
        #self.invalid_2 = numpy.zeros([grid_size,grid_size])
        #self.invalid_3 = numpy.zeros([grid_size,grid_size])
        #self.invalid_4 = numpy.zeros([grid_size,grid_size])
        # h score reset 
        #self.h_score = self.heuristic_score()
        self.agent_get_reward =0
        # 每次step都会更新 _used_actions ，使用_actions - _used_actions - _invalid_actions，剩下的才是合法的action space
        self._used_actions=set([])
        # invalid actions 比如0 11,22,,,99
        self._invalid_actions = set([i for i in range(grid_size*grid_size) if i//grid_size == i%grid_size])
        # action space reset
        self._actions = set(range(grid_size * grid_size)) -self._invalid_actions
        return self.get_observation(),{}


        
    def step(self, action):
        #print(f'step legal actions={self.legal_actions()}')
        if action not in self.legal_actions() or len(self.legal_actions())==0 :
            pass
        if not self.position:
            self.position =[-1,-1] # position[-1,-1]表示不在grid上的位置只是为了占位
        self.position[0] = action // grid_size
        self.position[1] = action %  grid_size
        #reward = 1 if self.position == [self.size - 1] * 2 else 0
        #不能写成reward = self.grid[self.position] 因为self.position=[1,1] 会导致grid[1,1]取得是两行
        # 或者写成reward = self.grid[self.position[0],self.position[1]] 
        #reward = self.grid[*self.position] 
        # 这个位置是非法的 也pass
        #if self.grid[self.position[0],self.position[1]]<=self.MARK_NEGATIVE:
        #    done = (numpy.max(self.mark) <= self.MARK_NEGATIVE) or len(self.legal_actions())==0
        #    return self.get_observation(), 0, done#bool(reward)
        reward = self.grid[self.position[0],self.position[1]]  - self.mark[self.position[0],self.position[1]] #- self.h_score / (grid_size/2)
        self.agent_get_reward += reward
        #print(f'123reward={reward}')
        # grid 变化太剧烈? 所以换成mark来记录已经不能下的位置
        #self.grid[self.position, :] = self.MARK_NEGATIVE
        #self.grid[:, self.position] = self.MARK_NEGATIVE
        self.mark[self.position, :] = self.MARK_NEGATIVE
        self.mark[:, self.position] = self.MARK_NEGATIVE
        # pos  history
        self.pos_history[self.position[0],self.position[1]] = 1        
        
        #self.pos_now = numpy.zeros([grid_size,grid_size])
        #self.pos_now[self.position[0],self.position[1]] = 1
        #self.invalid_1 = numpy.zeros([grid_size,grid_size])
        #self.invalid_1[self.position[0], :]=1
        #self.invalid_2 = numpy.zeros([grid_size,grid_size])
        #self.invalid_2[self.position[1], :]=1
        #self.invalid_3 = numpy.zeros([grid_size,grid_size])
        #self.invalid_3[:, self.position[0]]=1
        #self.invalid_4 = numpy.zeros([grid_size,grid_size])
        #self.invalid_4[:, self.position[1]]=1
        
        
        
        #done = (numpy.max(self.grid) <= self.MARK_NEGATIVE) or len(self.legal_actions())==0
        done = (numpy.max(self.mark) <= self.MARK_NEGATIVE) or len(self.legal_actions())==0
        #done =  len(self.legal_actions())==0
        #reward =0
        
        #if done :
        #    if self.agent_get_reward>= self.h_score :
        #        #reward = self.agent_get_reward - self.h_score
        #        reward +=   10
            
        truncated=False# 占位用 无意义
        return self.get_observation(), reward, done, truncated,{}#bool(reward)

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
                    print(numpy.array(self.current_pos)==numpy.array([row,col]).reshape(-1,1))
                except Exception as e:
                    print('Initial state')

                if self.maze[row, col] == '#':  # Obstacle
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'S':  # Starting position
                    pygame.draw.rect(self.screen, (0, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'G':  # Goal position
                    pygame.draw.rect(self.screen, (255, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))

                if numpy.array_equal(numpy.array(self.current_pos), numpy.array([row, col]).reshape(-1,1)):  # Agent position
                    pygame.draw.rect(self.screen, (0, 0, 255), (cell_left, cell_top, self.cell_size, self.cell_size))

        pygame.display.update()  # Update the display
    def _get_obs(self):
        self.show = deepcopy(self.maze)
        self.show[self.current_pos[0],self.current_pos[1]] = 1.0
        #self.show
        #print(self.show.dtype)
        # shape(4,4) to shape(1,4,4)
        return numpy.array([self.show])
        # 
        # self.show = numpy.array([self.show]).transpose(1,2,0) # 1,4,4 - > 4,4,1 貌似要把channel放在最后才行
        #print(self.show.shape)
        #return self.show.flatten() 
        #return self.show        
    
    def render(self):
        print(self.get_observation())        

        
    def close2(self):
        pygame.display.quit()
        pygame.quit()
