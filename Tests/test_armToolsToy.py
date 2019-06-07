import gym
import gym_myGridEnv
import gym_flowers 
import matplotlib.pyplot as plt 
import pickle 
import numpy as np 
import termiplot as tp
from tqdm import tqdm 
from GRBF import sample_random_trajectories
import time 

true_traj, Acs = [], []
count = 0



for i in tqdm(range(1)):
    actions = 2*np.random.random((50,4))-1


    traj = []
    # ~ env = gym.make('ArmToolsToys-v1')
    env = gym.make('MultiTaskFetchArmNLP1-v0')
    
    
    
    # ~ weights = "./"
    # ~ env.unwrapped.init(lambda x: x,0,None,weights)
    
    grip_pos = [[],[]]
    cube_pos = [[],[]]
    obs = env.reset()
    # ~ obs = env.unwrapped.reset(np.(3))
    # ~ traj.append(obs.copy())
    # ~ env.render()
    # ~ time.sleep(0.5)
    for t in range(50):
        env.render()
        time.sleep(1)
        obs, reward, done, info = env.step(actions[t])
        # ~ traj.append(obs.copy())

        
    # ~ if np.linalg.norm( obs[32:]) > 0.3:
        # ~ count += 1 
        # ~ Acs.append(actions)
        # ~ true_traj.append(traj)
        # ~ print(count)
    # ~ if count > 100:
        # ~ break

# ~ print("grasping stick", count)

# ~ with open("/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy_random_MS.pk", 'bw') as f:
    # ~ pickle.dump( [true_traj,Acs], f)

# ~ env.close()
