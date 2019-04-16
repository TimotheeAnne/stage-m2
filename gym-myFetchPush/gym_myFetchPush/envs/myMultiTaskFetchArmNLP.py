import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os  
import tensorflow as tf
import numpy as np
import pickle
from dotmap import DotMap
import random 

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class Normalization:
    def __init__(self):
        self.inputs_mean = None
        self.inputs_std = None
        self.outputs_mean = None
        self.outputs_std = None
        
    def init(self,inputs,outputs):
        self.inputs_mean = np.mean(inputs,axis=0)
        self.inputs_std = np.std(inputs,axis=0)
        self.outputs_mean = np.mean(outputs,axis=0)
        self.outputs_std = np.std(outputs,axis=0)
    
    def load(self, model_dir):
        with open(os.path.join(model_dir, "norm.pk"), "br") as f:
            [self.inputs_mean,self.inputs_std,self.outputs_mean, self.outputs_std] = pickle.load(f)
        
    def normalize_inputs(self,x):
        return (x - self.inputs_mean) / self.inputs_std

    def normalize_outputs(self,x):
        return (x - self.outputs_mean) / self.outputs_std
        
    def denormalize_outputs(self,y):
        return (y * self.outputs_std) + self.outputs_mean
        
    def pretty_print(self):
        print( "in mean", self.inputs_mean)
        print( "in std", self.inputs_std)
        print( "out mean", self.outputs_mean)
        print( "out std", self.outputs_std)

class MyMultiTaskFetchArmNLP(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._observation = np.zeros(50)
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
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(50,), dtype='float32')
        
        
        """ model Loading """
        self.model_dir = "/home/tim/Documents/stage-m2/gym-myFetchPush/log/tf25/"
        self.norm = Normalization()
        self.model = self.nn_constructor(self.model_dir)
        
    def nn_constructor(self,model_dir):
        """ Load BNN """
        REG = 0.0005
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation=tf.nn.relu, input_shape=[29], 
                        bias_initializer = tf.constant_initializer(value=0.),
                        kernel_initializer = tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer = tf.keras.regularizers.l2(l=REG)),
            tf.keras.layers.Dense(256, activation=tf.nn.relu,
                        bias_initializer = tf.constant_initializer(value=0.),
                        kernel_initializer = tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer = tf.keras.regularizers.l2(l=REG)),
            tf.keras.layers.Dense(256, activation=tf.nn.relu,
                        bias_initializer = tf.constant_initializer(value=0.),
                        kernel_initializer = tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer = tf.keras.regularizers.l2(l=REG)),
            tf.keras.layers.Dense(25 , activation=None),
        ])
        model.load_weights(model_dir+'model.h5')
        self.norm.load(model_dir)
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

        inputs = np.array([ self.norm.normalize_inputs(np.concatenate((self._observation[:25],action)))])
        
        pred = self.norm.denormalize_outputs(self.model.predict(inputs))

        self._observation[:25] = self._observation[:25] + pred[0]
        self._observation[25:] += pred[0]
        """ increment step """
        self._steps += 1
            
        return self._observation.copy(), 0, False, {}
      
    def reset(self):
        self._observation = np.zeros(50)
        self._observation[:3]= np.array([1.34193113, 0.74890335, 0.484762558])
        self._observation[3:5] = self._observation[:2] + np.random.uniform(-self.obj_range, self.obj_range, size=2)
        self._observation[5] = 0.425990820
        self._observation[6:9] = self._observation[3:6]-self._observation[:3]
        self._observation[9:25] =   [2.71748964e-06, -9.48888964e-08, -0.00000000e+00,
                                      0.00000000e+00, -0.00000000e+00 ,-6.22253242e-05, -2.83505750e-07,
                                      6.65625574e-04,  5.45834835e-10, -4.74639458e-10 ,-1.96138370e-16,
                                      6.22253120e-05 , 2.83491664e-07, -3.98201343e-05 , 4.72693405e-07,
                                      2.35609066e-07 ]

        # ~ self.model_index = np.random.randint(5)
        
        self._steps = 0
        
        return  self._observation.copy() 
        
    def render(self, mode='human'):
      pass

    def close(self):
      pass
    
