import gym
import gym_myFetchPush
import gym_flowers 
import matplotlib.pyplot as plt 
import pickle 
import numpy as np 
import termiplot as tp

env = gym.make('myMultiTaskFetchArmNLP-v1')
true_env = gym.make('MultiTaskFetchArmNLP1-v0')

fig = tp.figure()

Episodes = []
for _ in range(10):
    obs = true_env.reset()
    episode = [obs]
    for t in range(50):
        obs, reward, done, info = true_env.step(np.random.random(4))
        episode.append(obs)
    Episodes.append(episode)

print(np.shape(Episodes))
env.train(Episodes)

for _ in range(2):
    grip_pos = [[],[]]
    cube_pos = [[],[]]
    env.reset()
    for t in range(50):
        obs, reward, done, info = env.step(np.random.random(4))
        grip_x, grip_y = obs[0],obs[1]
        cube_x, cube_y = obs[3],obs[4]
        grip_pos[0].append(grip_x)
        grip_pos[1].append(grip_y)
        cube_pos[0].append(cube_x)
        cube_pos[1].append(cube_y)
    fig.plot(grip_pos[0],grip_pos[1])
fig.show()

