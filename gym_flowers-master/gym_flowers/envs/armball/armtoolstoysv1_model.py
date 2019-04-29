import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os  
import tensorflow as tf
import numpy as np
import pickle
from dotmap import DotMap
import random 

# ~ def goal_distance(goal_a, goal_b):
    # ~ assert goal_a.shape == goal_b.shape
    # ~ return np.linalg.norm(goal_a - goal_b, axis=-1)

OBS_DIM = 18

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
    true_traj, Acs = Episodes['o'], Episodes['u']
    
    Inputs = []
    Targets = []
    moved_objects = 0
    for j in range(len(true_traj)):
        for t in range(50):
            inputs = np.concatenate((true_traj[j][t][:OBS_DIM],Acs[j][t]))
            targets = true_traj[j][t+1][:OBS_DIM] - true_traj[j][t][:OBS_DIM]
            bool_targets = [1 if abs(targets[i])>0 else -1 for i in [6,10,14,16]]
            Inputs.append(inputs)
            Targets.append(np.concatenate((targets,bool_targets)))
            if 1 in bool_targets :
                moved_objects += 1
    print( 'moved_objects transitions', moved_objects, moved_objects/(50*j))
    # ~ norm.init(Inputs,Targets)
    # ~ norm.pretty_print()
    samples = list(zip( Inputs,Targets))
    np.random.shuffle( samples)
    Inputs, Targets = zip( *samples)
    
    return (norm.normalize_inputs(np.array(Inputs)),norm.normalize_outputs(np.array(Targets)))
    
class ArmToolsToysV1_model(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.n_act = 4
        self.half = OBS_DIM
        self.n_obs = self.half*2
        
        self._observation = np.zeros(self.n_obs)

        self.obj_range = 0.15
        self.reward_type = 'sparse'
        self.distance_threshold = 0.05
        self.target_range = 0.15
        self._done = False
        self._steps = 0
        self._n_timesteps = 50
        self._max_episode_steps = self._n_timesteps
        
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.n_obs,), dtype='float32')
        
        
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
            tf.keras.layers.Dense(256, activation=tf.nn.relu, input_shape=[22], 
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
            tf.keras.layers.Dense(22 , activation=None),
        ])
        model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error']
              )
        self.weight_init = model.get_weights()
        return model

    def train(self, Episodes, logdir=None):
        # ~ self.model.set_weights(self.weight_init)
        if not logdir is None:
            with open(os.path.join(logdir, 'train_episodes'+str(self.iteration)+'.pk'), 'ba') as f:
                pickle.dump(Episodes, f)
        self.iteration += 1
        (x_train, y_train) = compute_samples(Episodes, self.norm)
        self.model.fit(x_train, y_train, epochs=self.EPOCH,shuffle=True, verbose=False)

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        return seed
        
    # ~ def compute_reward(self, achieved_goal, desired_goal, info=None):
        # ~ d = goal_distance(achieved_goal, desired_goal)
        # ~ if self.reward_type == 'sparse':
            # ~ return -(d > self.distance_threshold).astype(np.float32)
        # ~ else:
            # ~ return -d
            
    # ~ def is_success(self, achieved_goal, desired_goal):
        # ~ d = goal_distance(achieved_goal, desired_goal)
        # ~ return -(d > self.distance_threshold).astype(np.float32)

            
    # ~ def sample_goal(self):
        # ~ goal = self._observation[:3] + np.random.uniform(-self.target_range, self.target_range, size=3)
        # ~ return goal.copy()
        
    def step(self, action):
        """ Compute the next observation """

        inputs = np.array([ self.norm.normalize_inputs(np.concatenate((self._observation[:self.half],action)))])
        
        output = self.norm.denormalize_outputs(self.model.predict(inputs)[0])
        pred = output[:self.half]
        pred[5] = 1 if pred[5]>0 else -1
        for (i,b) in [(6,18),(7,18),(8,18),(9,18),(10,19),(11,19),(12,19),(13,19),(14,20),(15,20),(16,21),(17,21)]:
            pred[i] = pred[i] if output[b] > 0 else 0
            
        self._observation[:self.half] = self._observation[:self.half] + pred
        self._observation[self.half:] += pred
        
        
        """ increment step """
        self._steps += 1
            
        return self._observation.copy(), 0, False, {}
      
    def reset(self, obs=None):
        if obs is None:
            self._observation = np.zeros(self.n_obs)
            self._observation[:self.half] = [0,0,0,
                                             6.123234e-17,1,
                                             0,
                                             -0.75, 0.25, -1.10355339, 0.603553391,
                                             0.75, 0.25, 1.10355339, 0.603553391,
                                             -0.3, 1.1,
                                             0.3, 1.1
                                             ]
        else:
            self._observation = obs.copy()
            
        self._steps = 0
        
        return  self._observation.copy() 
        
    def render(self, mode='human'):
      pass

    def close(self):
      pass
    
