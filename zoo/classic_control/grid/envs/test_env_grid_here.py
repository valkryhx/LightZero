from easydict import EasyDict
from zoo.classic_control.grid.envs.grid_lightzero_env import MyGridEnv


config = EasyDict(dict(
    replay_path=None,
))
# Test the environment
env = MyGridEnv(config)
obs = env.reset()
#env.render()
print("====init====")
print(obs)

done = False
i=0
print(f'\n\n====START TO PLAY====\n')
while i<10:
    #pygame.event.get()
    
    action = env.random_action()
    print(f'idx={i}, action={action}')
    obs, rew, done, info = env.step(action)
    print(f'obs={obs}')
    print('Reward:', reward)
    print('Done:', done)
    
    #pygame.time.wait(200)  
    i += 1