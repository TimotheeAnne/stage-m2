import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os  
import tensorflow as tf
import numpy as np
import pickle
import random 
import sys


OBS_DIM = 18

def compute_samples(Episodes):
    true_traj, Acs = Episodes['o'], Episodes['u']
    
    Inputs = []
    Targets = []
    moved_objects = 0
    for j in range(len(true_traj)):
        for t in range(50):
            inputs = np.concatenate((true_traj[j][t][:OBS_DIM],Acs[j][t]))
            targets = true_traj[j][t+1][:OBS_DIM] - true_traj[j][t][:OBS_DIM]
            bool_targets = [1 if np.linalg.norm(targets[i:i+2]) > 0 else -1 for i in [8,12,14,16]]
            Inputs.append(inputs)
            Targets.append(np.concatenate((targets,bool_targets)))
            if 1 in bool_targets :
                moved_objects += 1
    print( 'moved_objects transitions', moved_objects, moved_objects/(50*j))
    return (np.array(Inputs),np.array(Targets))
    
class ArmToolsToysV1_model(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.n_act = 4
        self.OBS_DIM = OBS_DIM
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
        self.model_dir = None
        self.eval_data = "../../../../../tf_test/data/ArmToolsToy_1000pertinent.pk"
        self.replay_buffer = ReplayBuffer()
        self.model = self.nn_constructor(self.model_dir)
        self.EPOCH = 50
        self.iteration = 0

    def init(self, oracle, rank, logdir):
        self.oracle = oracle(30)
        self.rank = rank
        self.logdir = logdir

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

    def train(self, Episodes):
        with open(self.logdir+"/final_observations_r"+str(self.rank)+"_"+str(self.iteration)+".pk",'bw') as f:
            pickle.dump(np.array(Episodes['o'])[:,-1,:18],f)
        (inputs, targets) = compute_samples(Episodes)
        self.replay_buffer.add_samples(inputs,targets)
        (x_train, y_train) = self.replay_buffer.sample(random.sample)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2,verbose=0, mode='auto')
        self.model.fit(x_train, y_train, epochs=self.EPOCH, shuffle=True, verbose=False, validation_split=0.1, callbacks=[es])
        self.eval()
        self.iteration += 1
        
    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        return seed
        

    def step(self, action):
        """ Compute the next observation """
        inputs = np.array([ np.concatenate((self._observation[:self.half],action))])
        
        output = self.model.predict(inputs)[0]
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

    def eval(self):
        with open( self.eval_data, 'rb') as f:
            eval_data = pickle.load(f)
        [true_traj, Acs] = eval_data
        true_traj = np.array(true_traj)
        
        traj_pred = self.predict_trajectory(true_traj, Acs)
        trans_pred = self.predict_transition(true_traj, Acs)
        
        true_rewards = self.oracle.eval_all_goals_from_state(true_traj[:,-1])
        predict_rewards = self.oracle.eval_all_goals_from_state( traj_pred[-1,:])
                
        confusion_matrix = self.compute_confusion_matrix(true_rewards, predict_rewards)

        traj_pred = traj_pred[:,:,:18]
        
        with open(self.logdir+"/prediction_r"+str(self.rank)+"_"+str(self.iteration)+".pk",'bw') as f:
            pickle.dump((traj_pred,trans_pred),f)

        with open(self.logdir+"/confusion_matrix_r"+str(self.rank)+"_"+str(self.iteration)+".pk",'bw') as f:
            pickle.dump(confusion_matrix,f)

    def compute_confusion_matrix(self, true_rewards, predict_rewards):
        n_tasks = len(true_rewards[0])
        confusion_matrix = np.zeros((n_tasks,2,2))
        for j in range(len(true_rewards)):
            for r in range(n_tasks):
                truth_value = int(true_rewards[j][r] == 0)
                predict_value = int(predict_rewards[j][r] == 0)
                confusion_matrix[r][truth_value][predict_value] += 1 
        return confusion_matrix

    def filter(self, obs_pred, output, obs):
        # gripper state prediction -1 or 1 
        obs_pred[:,5] =  2*(obs_pred[:,5]>0)-1
        # moving objects prediction
        for (i,b) in [(6,18),(7,18),(8,18),(9,18),(10,19),(11,19),(12,19),(13,19),(14,20),(15,20),(16,21),(17,21)]:
            obs_pred[:,i] = obs_pred[:,i] * (output[:,b] > 0) + obs[:,i] * (output[:,b] <= 0)
        return obs_pred


    def predict_trajectory(self, true_traj, Acs):
        true_traj = np.array(true_traj)
        Acs = np.array(Acs)
        obs = true_traj[:,0,:self.OBS_DIM]
        pred_traj = [np.concatenate((obs,obs-true_traj[:,0,:self.OBS_DIM]),axis=1)]
        for t in range(np.shape(Acs)[1]):
            inputs = np.concatenate((obs[:,:self.OBS_DIM],Acs[:,t]), axis=1)
            output = self.model.predict(inputs)
            obs_pred = output[:,:self.OBS_DIM]+obs
            obs = self.filter(obs_pred, output, obs)
            pred_traj.append(np.concatenate((obs,obs-true_traj[:,0,:self.OBS_DIM]),axis=1))
        return np.array(pred_traj.copy())


    def predict_transition(self, true_traj, Acs):
        true_traj = np.array(true_traj)
        Acs = np.array(Acs)
        obs = true_traj[:,0,:self.OBS_DIM]
        pred_traj = [obs]
        for t in range(np.shape(Acs)[1]):
            inputs = np.concatenate((true_traj[:,t,:self.OBS_DIM],Acs[:,t]), axis=1)
            output = self.model.predict(inputs)
            obs_pred = output[:,:self.OBS_DIM]+true_traj[:,t,:self.OBS_DIM]
            obs = self.filter(obs_pred, output, obs)
            pred_traj.append(obs.copy())
        return np.array(pred_traj.copy())
        
class ReplayBuffer:
    def __init__(self):
        self.buffer = []
        
        self.indexes = [[],[],[],[],[]]
        self.current_size = 0
        self.max_size = 2000000
        self.head = 0

    def add_samples(self, inputs, targets):
        assert( len(inputs)==len(targets))
        n = len(inputs)
        for (inp, target) in zip( inputs,targets):
            self.add(inp,target)


    def add_in_indexes(self, target, idx):
        pertinent = False
        if target[18] == 1:
            self.indexes[1].append(idx)
            pertinent = True
        if target[19] == 1:
            self.indexes[2].append(idx)
            pertinent = True
        if target[20] == 1:
            self.indexes[3].append(idx)
            pertinent = True
        if target[21] == 1:
            self.indexes[4].append(idx)
            pertinent = True
        if not pertinent:
            self.indexes[0].append(idx)

    def remove_from_indexes(self, idx):
        for i in range(5):
            if idx in self.indexes[i]:
                self.indexes[i].remove(idx)

    def add(self, inp, target):
        if self.current_size < self.max_size:
            self.buffer.append({'input': inp, 'target':target})
            self.add_in_indexes( target, self.current_size)
            self.current_size += 1 
        else:
            self.buffer[self.head]={'input': inp, 'target':target}
            self.remove_from_indexes(self.head)
            self.add_in_indexes(target, self.head)
            self.head = (self.head +1) % self.max_size
    
    def sample(self, sampling_function, objects=range(5)):
        sizes = [len(self.indexes[i]) for i in range(5)]
        if np.sum(sizes[1:]) <= 50:
            n_sample = sizes[0]
        elif np.sum(sizes[3:]) <= 50:
            n_sample = np.sum(sizes[1:])
        else:
            n_sample = np.sum(sizes[3:])
        samples_indexes = []
        for i in objects:
            samples_indexes += list(sampling_function( self.indexes[i], min( sizes[i], n_sample)))
        
        count = [0 for _ in range(5)]

        x, y = [], []
        for idx in samples_indexes:
            for i in range(5):
                if idx in self.indexes[i]:
                    count[i] += 1
            sample = self.buffer[idx]
            x.append(sample['input'])
            y.append(sample['target'])
        print("Training transitions: ", count)
        return np.array(x), np.array(y)

    def pretty_print(self):
        sizes = [len(self.indexes[i]) for i in range(5)]
        print("Current size: ", self.current_size)
        print("Head: ", self.head)
        print("Indexes sizes: ", sizes)
        if self.current_size > 0:
            print( " Proportion: ", np.array(sizes)/self.current_size)
