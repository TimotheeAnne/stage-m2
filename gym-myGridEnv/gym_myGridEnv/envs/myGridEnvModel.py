import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os  
import tensorflow as tf
import numpy as np
import pickle
from dotmap import DotMap
import random 

class Normalization:
    def __init__(self):
        self.inputs_mean = 0
        self.inputs_std = 1
        self.outputs_mean = 0
        self.outputs_std = 1
        self.samples = 0
        
    def init(self,inputs,outputs):
        if self.samples == 0:
            self.inputs_mean = np.mean(inputs,axis=0)
            self.inputs_std = np.std(inputs,axis=0)
            self.outputs_mean = np.mean(outputs,axis=0)
            self.outputs_std = np.std(outputs,axis=0)
            self.samples = len(inputs)
        else:
            n_old_samples = self.samples
            n_new_samples = len(inputs)
            self.samples = n_old_samples + n_new_samples
            alpha = n_old_samples/self.samples
            self.inputs_mean = alpha * self.inputs_mean + (1-alpha) * np.mean(inputs,axis=0)
            self.inputs_std = alpha * self.inputs_std + (1-alpha) * np.std(inputs,axis=0)
            self.outputs_mean = alpha * self.outputs_mean + (1-alpha) * np.mean(outputs,axis=0)
            self.outputs_std = alpha * self.outputs_std + (1-alpha) * np.std(outputs,axis=0)
        for i in range(len(self.inputs_std)):
            if self.inputs_std[i] == 0.:
                self.inputs_std[i] = 1
        for i in range(len(self.outputs_std)):
            if self.outputs_std[i] == 0.:
                self.outputs_std[i] = 1
        
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

def compute_samples(Episodes,norm):
    Inputs = []
    Targets = []
    moving_cube = 0
    for j in range(len(Episodes['o'])):
        for t in range(50):
            inputs = np.concatenate((Episodes['o'][j][t][:25],Episodes['u'][j][t]))
            targets = Episodes['o'][j][t+1][:25]- Episodes['o'][j][t][:25]
            Inputs.append(inputs)
            Targets.append(targets)
            if np.linalg.norm(targets[3:6]) > 0.001:
                moving_cube += 1
    print("moving cube transition: ", moving_cube, moving_cube/(50*j)) 
    # ~ norm.init(Inputs,Targets)
    # ~ norm.pretty_print()
    return (norm.normalize_inputs(np.array(Inputs)),norm.normalize_outputs(np.array(Targets)))
    
class MyGridEnvModel(gym.Env):
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

        """ model Loading """
        self.model_dir = "/home/tim/Documents/stage-m2/gym-myFetchPush/log/tf25/"
        self.norm = Normalization()
        self.model = self.nn_constructor(self.model_dir)
        self.EPOCH = 100
        self.iteration = 0

    def nn_constructor(self,model_dir):
        """ Load BNN """
        REG = 0.000
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation=tf.nn.relu, input_shape=[7], 
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
            tf.keras.layers.Dense(5 , activation=None),
        ])
        model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error']
              )
        self.weight_init = model.get_weights()
        return model

    def train(self, Episodes, logdir=None):
        # ~ if not logdir is None:
            # ~ with open(os.path.join(logdir, 'train_episodes'+str(self.iteration)+'.pk'), 'ba') as f:
                # ~ pickle.dump(Episodes, f)
        self.iteration += 1
        (x_train, y_train) = compute_samples(Episodes, self.norm)
        self.model.fit(x_train, y_train, epochs=self.EPOCH,shuffle=True, verbose=False)
        
    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def step(self, action):
        """ Compute the next observation """

        inputs = np.array([ self.norm.normalize_inputs(np.concatenate((self._observation[:5],action)))])
        pred = self.norm.denormalize_outputs(self.model.predict(inputs))
        self._observation[:5] = self._observation[:5] + pred[0]
        self._observation[5:] += pred[0]
        
        """ increment step """
        self._steps += 1
            
        return self._observation.copy(), 0, False, {}
      
    def reset(self, obs=None):
        self._observation[:2] = np.array([0,0]) #2*np.random.random(2)-1
        self._observation[2:4] = np.random.random(2)-0.5
        
        while np.linalg.norm( self._observation[2:4] - self._observation[:2] ) < self._grap_dist:
            self._observation[2:4] = 2*np.random.random(2)-1 
        
        # ~ self._observation[:4] = np.array([0,0,1,1])
        
        self._observation[4] = 0
        
        self._init_obs = self._observation[:4].copy()
        self._observation[self.half:] = self._observation[:4]-self._init_obs 
        
        self._steps = 0
        return  self._observation.copy() 
        
    def render(self, mode='human'):
      pass

    def close(self):
      pass
    
