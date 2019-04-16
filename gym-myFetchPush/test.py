import gym
import gym_myFetchPush
import gym_flowers 
import matplotlib.pyplot as plt 
import pickle 
import numpy as np 
import termiplot as tp

for i in range(4):
    # ~ actions = 2*np.random.random((50,4))-1
    ac = [[-1,0,0,0],[1,0,0,0],[0,1,0,0],[0,-1,0,0]][i]
    actions = [ac for _ in range(50)]

    env = gym.make('myMultiTaskFetchArmNLP-v0')

    grip_pos = [[],[]]
    cube_pos = [[],[]]
    obs = env.reset()
    obs = obs[25:]
    grip_x, grip_y = obs[0],obs[1]
    cube_x, cube_y = obs[3],obs[4]
    grip_pos[0].append(grip_x)
    grip_pos[1].append(grip_y)
    cube_pos[0].append(cube_x)
    cube_pos[1].append(cube_y)
    for t in range(50):
        obs, reward, done, info = env.step(actions[t])
        obs = obs[25:]
        grip_x, grip_y = obs[0],obs[1]
        cube_x, cube_y = obs[3],obs[4]
        grip_pos[0].append(grip_x)
        grip_pos[1].append(grip_y)
        cube_pos[0].append(cube_x)
        cube_pos[1].append(cube_y)
        plt.plot(grip_pos[0],grip_pos[1], c='red', lw=3)
        
    env = gym.make('MultiTaskFetchArmNLP1-v0')
    grip_pos = [[],[]]
    cube_pos = [[],[]]
    obs = env.reset()
    obs = obs[25:]
    grip_x, grip_y = obs[0],obs[1]
    cube_x, cube_y = obs[3],obs[4]
    grip_pos[0].append(grip_x)
    grip_pos[1].append(grip_y)
    cube_pos[0].append(cube_x)
    cube_pos[1].append(cube_y)
    for t in range(50):
        obs, reward, done, info = env.step(actions[t])
        obs = obs[25:]
        # ~ obs, reward, done, info = env.step(0)
        grip_x, grip_y = obs[0],obs[1]
        cube_x, cube_y = obs[3],obs[4]
        grip_pos[0].append(grip_x)
        grip_pos[1].append(grip_y)
        cube_pos[0].append(cube_x)
        cube_pos[1].append(cube_y)
        plt.plot(grip_pos[0],grip_pos[1],c='green',lw=3)
        
plt.show()

