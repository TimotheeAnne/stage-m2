import gym
import gym_myGridEnv
import gym_flowers 
import matplotlib.pyplot as plt 
import pickle 
import numpy as np 
import termiplot as tp

for i in range(100):
    actions = 2*np.random.random((50,2))-1

    env = gym.make('myGridEnv-v0')

    grip_pos = [[],[]]
    cube_pos = [[],[]]
    obs = env.reset()
    grip_x, grip_y = obs[0],obs[1]
    cube_x, cube_y = obs[2],obs[3]
    grip_pos[0].append(grip_x)
    grip_pos[1].append(grip_y)
    cube_pos[0].append(cube_x)
    cube_pos[1].append(cube_y)
    for t in range(50):
        obs, reward, done, info = env.step(actions[t])
        # ~ print(obs,actions[t])
        grip_x, grip_y = obs[0],obs[1]
        cube_x, cube_y = obs[2],obs[3]
        grip_pos[0].append(grip_x)
        grip_pos[1].append(grip_y)
        cube_pos[0].append(cube_x)
        cube_pos[1].append(cube_y)
    plt.plot(grip_pos[0],grip_pos[1], c='red', lw=3)
    plt.plot(cube_pos[0],cube_pos[1], c='green', lw=3)
    
plt.show()

