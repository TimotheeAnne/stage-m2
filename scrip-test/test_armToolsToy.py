import gym
import gym_myGridEnv
import gym_flowers 
import matplotlib.pyplot as plt 
import pickle 
import numpy as np 
import termiplot as tp

true_traj, Acs = [], []
for i in range(1):
    actions = 2*np.random.random((50,4))-1
    traj = []
    env = gym.make('ArmToolsToys-v0')

    grip_pos = [[],[]]
    cube_pos = [[],[]]
    obs = env.reset()
    traj.append(obs)

    for t in range(50):
        env.render()
        obs, reward, done, info = env.step(actions[t])
        traj.append(obs)
    Acs.append(actions)
    true_traj.append(traj)
    
env.close()
