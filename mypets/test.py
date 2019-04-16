import gym 
import matplotlib.pyplot as plt 
import numpy as np
import gym_flowers 

# ~ env = gym.make('FetchPush-v1')
# ~ obs = np.empty((51,25))
# ~ obs[0] = env.reset()['observation']
# ~ for t in range(50):
    # ~ obs[t] = env.step([0,0,0,0])[0]['observation']


env = gym.make("MultiTaskFetchArmNLP1-v0")
obs = np.empty((51,50))
obs[0] = env.reset()
for t in range(50):
    a  = 0 if t <20 else 1
    # ~ obs[t] = env.step([a,0,0,0])[0]
    obs[t] = env.step([1,1,1,0])[0]
    
print("cube pos", obs[1:,3:6]-obs[:-1,3:6])
print("cube rel pos", obs[:,6:9] + obs[:,0:3] - obs[:,3:6])
print("cube rot", obs[1:,11:14]-obs[:-1,11:14])
print("cube vel", obs[:,14:17]+obs[:,20:23])
print("cube vel rot", obs[1:,17:20]-obs[:-1,17:20])

np.concatenate([obs[:3]+pred[:3], # gripper position
                obs[3:6], # cube position
                obs[3:6]-( obs[:3] + pred[:3]), # relative position of cub/gripper
                pred[9:11]+obs[9:11], # gripper state
                obs[11:14], # cube rot
                -obs[20:23], # relative cube velocitie, equal to -Vgripper if Vcube=0
                obs[17:20], # rotation velocities
                obs[20:25]+pred[20:25], # gripper velocities
                [0] # cube moving
                    ] )

