import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os  
import tensorflow as tf
import numpy as np
import pickle
from dotmap import DotMap
import random 

from dmbrl.modeling.layers import FC
from dmbrl.modeling.models import BNN

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class MyFetchPush(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        
        self._observation = np.zeros(6)
        self.desired_goal = np.zeros(3)
        self.achieved_goal = np.zeros(3)
        self.obj_range = 0.15
        self.reward_type = 'sparse'
        self.distance_threshold = 0.05
        self.target_range = 0.15
        self._done = False
        self._steps = 0
        self._n_timesteps = 50
        self._max_episode_steps = self._n_timesteps
        
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')
        
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=(6,), dtype='float32'),
        ))
        
        """ model Loading """
        self.model_index = 0
        self.model_dir = '/home/tim/Documents/stage-m2/gym-myFetchPush/log/dim6'
        #self.model_dir = '/home/tanne/Experiment/gym-myFetchPush/log/dim6'
        cfg = tf.ConfigProto()
        self.SESS = tf.Session(config=cfg)
        self.model = self.nn_constructor(self.model_dir)
        
    def nn_constructor(self,model_dir):
        """ Load BNN """
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            model = BNN(
                DotMap(
                name="model", num_networks=5, sess=self.SESS, load_model=True,
                model_dir = model_dir
            ))
            model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})

        return model

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        return seed
        
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        d = goal_distance(achieved_goal, desired_goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
            
    def is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return -(d > self.distance_threshold).astype(np.float32)

            
    def sample_goal(self):
        goal = self._observation[:3] + np.random.uniform(-self.target_range, self.target_range, size=3)
        return goal.copy()
        
    def step(self, action):
        """ Compute the next observation """

        inputs = np.array([ [np.concatenate((self._observation,action))] for _ in range(5)])
        pred = self.model.predict(inputs)
        # ~ pred = [ [ [ 0. for _ in range(6)] for _ in range(5)] for _ in range(2)]
        self._observation = self._observation + pred[0][self.model_index][0]
        
        """ increment step """
        self._steps += 1
        if self._steps == self._n_timesteps:
            self._done = True
            
        """ compute achieved goals and reward """
        self.achieved_goal = self._observation[3:6]
        self.reward = self.compute_reward( self.achieved_goal, self.desired_goal)
        
        self.obs = dict(observation=self._observation, desired_goal=self.desired_goal, achieved_goal=self.achieved_goal)
        info = {}
        info['is_success'] = self.is_success(self.achieved_goal, self.desired_goal) == 0
        
        return self.obs, self.reward, self._done, info
      
    def reset(self):
        self._observation = np.array([1.34193113, 0.74890335, 0.41363137, 0,0, 0.42478449])
        # ~ self.achieved_goal[:2] = self._observation[:2] + np.random.uniform(-self.obj_range, self.obj_range, size=2)
        self.achieved_goal[:2] = self._observation[:2] + np.array([0.,0.11])
        
        self.achieved_goal[-1] = 0.42478449
        self._observation[3:] = self.achieved_goal
        self.model_index = np.random.randint(5)
        
        # ~ self.desired_goal = self.sample_goal()
        self.desired_goal = np.array(self.achieved_goal + np.array([0.,0.1,0.])) 
        
        self._done = False
        self._steps = 0
        self.obs = dict(observation=self._observation, desired_goal=self.desired_goal, achieved_goal=self.achieved_goal)

        return  self.obs 
        
    def render(self, mode='human'):
      pass

    def close(self):
      pass
    
