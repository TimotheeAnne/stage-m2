import gym
import gym_flowers
import time
import numpy as np

env = gym.make('flowers-Walker-v2') #4242 gap 42 stump #'flowers-Walker-v2''BipedalWalkerHardcore-v2''OGWalkerHardcore-v2'
env.seed(564564)
for i in range(20):
    env.env.set_environment(roughness=None, stump_height=[0,5], gap_width=None, step_height=None, step_number=None)
    env.reset()
    env.render()
    time.sleep(2)

done = False
for j in range(1):
    start = time.time()
    env.reset()
    for i in range(2000):
        o = env.step(np.random.rand(4))
        env.render()
        #print(o)
    end = time.time()
    print(end-start)


