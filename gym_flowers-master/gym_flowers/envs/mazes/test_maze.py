import gym
import gym_flowers
import numpy as np

# env = gym.make('ModularArm012-v0')
env = gym.make('Maze-v0')
obs = env.reset()
env.render()

for i in range(500):
    act = np.array([1, np.cos(i/3)*0.3])
    _, r, _, _ = env.step(act)
    print(r)
    env.render()