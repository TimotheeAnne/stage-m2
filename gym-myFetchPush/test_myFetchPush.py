import gym
import gym_myFetchPush
import matplotlib.pyplot as plt 
import pickle 
import numpy as np 
import termiplot as tp

env = gym.make('myFetchPush-v0')

f = open('/home/tim/Documents/stage-m2/mypets/data/transition_push_1_for_eval.pk', 'rb')
[Acs,Obs] = pickle.load(f)
f.close()
fig = tp.figure()

for _ in range(2):
    grip_pos = [[],[]]
    cube_pos = [[],[]]
    env.reset()
    for t in range(50):
        obs, reward, done, info = env.step(Acs[0][t])
        # ~ obs, reward, done, info = env.step(0)
        grip_x, grip_y = obs['observation'][0],obs['observation'][1]
        cube_x, cube_y = obs['achieved_goal'][0],obs['achieved_goal'][1]
        grip_pos[0].append(grip_x)
        grip_pos[1].append(grip_y)
        cube_pos[0].append(cube_x)
        cube_pos[1].append(cube_y)
    fig.plot(grip_pos[0],grip_pos[1])
fig.show()
