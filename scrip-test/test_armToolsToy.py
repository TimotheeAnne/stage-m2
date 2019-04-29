import gym
import gym_myGridEnv
import gym_flowers 
import matplotlib.pyplot as plt 
import pickle 
import numpy as np 
import termiplot as tp
from tqdm import tqdm 
import time 

true_traj, Acs = [], []
count = 0
for i in tqdm(range(3)):
    actions = 2*np.random.random((50,4))-1

    traj = []
    env = gym.make('ArmToolsToys-v1')

    grip_pos = [[],[]]
    cube_pos = [[],[]]
    obs = env.reset()
    traj.append(obs.copy())
    print(obs)
    for t in range(50):
        # ~ env.render()
        # ~ time.sleep(0.1)
        obs, reward, done, info = env.step(actions[t])
        traj.append(obs.copy())
        
    if np.linalg.norm( obs[20:24]) > 0:
        count += 1 
    Acs.append(actions)
    true_traj.append(traj)

print("grasping stick", count)

with open("/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy_random_eval.pk", 'bw') as f:
    pickle.dump( [true_traj,Acs], f)

env.close()
