import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os  
import tensorflow as tf
import numpy as np
import pickle
from dotmap import DotMap
import random 

class MyGridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._observation = np.zeros(9)
        self.half = 5
        self._init_obs = np.zeros(4)
        self._steps = 0
        self._n_timesteps = 50
        self._max_episode_steps = self._n_timesteps
        self._grap_dist = 0.05
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype='float32')
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype='float32')

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def step(self, action):
        """ Compute the next observation """
        self._observation[:2] = self._observation[:2] + action
        
        # first object
        if np.linalg.norm(self._observation[:2]-self._observation[2:4]) < self._grap_dist:
            self._observation[4] = 1
            
        if self._observation[4]:
            self._observation[2:4] = self._observation[:2]
        
        self._observation[self.half:] = self._observation[:4]-self._init_obs 
        """ increment step """
        self._steps += 1
            
        return self._observation.copy(), 0, False, {}
      
    def reset(self, obs=None):
        # ~ self._observation[:4] = 2*np.random.random(4)-1
        
        # ~ while np.linalg.norm( self._observation[2:4] - self._observation[:2] ) < self._grap_dist:
            # ~ self._observation[2:4] = 2*np.random.random(2)-1 
        self._observation[:4] = np.array([0,0,1,1])
        
        self._observation[4] = 0
        
        self._init_obs = self._observation[:4].copy()
        self._observation[self.half:] = self._observation[:4]-self._init_obs 
        
        self._steps = 0
        return  self._observation.copy() 
        
    def render(self, mode='human'):
      pass

    def close(self):
      pass
    
