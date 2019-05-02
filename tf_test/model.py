import tensorflow as tf
import numpy as np
import pickle 
from tqdm import tqdm

from replay_buffer import ReplayBuffer
from GRBF import sample_random_trajectories

class Ensemble:
    """ Deterministic ensemble transition model """
    def __init__(self, obs_d, acs_d, out_d, B=5, reg = 0):
        self.ACS_DIM = acs_d
        self.OUTPUT_DIM = out_d
        self.OBS_DIM = obs_d
        self.INPUT_DIM = self.ACS_DIM + self.OBS_DIM
        self.REG = reg
        self.B = B
        self.replay_buffer = ReplayBuffer()
        self.ensemble = []
        self.trained = False
        
        for _ in range(self.B):
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(256, activation=tf.nn.relu, input_shape=[self.INPUT_DIM], 
                            bias_initializer = tf.constant_initializer(value=0.),
                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer = tf.keras.regularizers.l2(l=self.REG)),
                tf.keras.layers.Dense(256, activation=tf.nn.relu,
                            bias_initializer = tf.constant_initializer(value=0.),
                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer = tf.keras.regularizers.l2(l=self.REG)),
                tf.keras.layers.Dense(256, activation=tf.nn.relu,
                            bias_initializer = tf.constant_initializer(value=0.),
                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer = tf.keras.regularizers.l2(l=self.REG)),
                tf.keras.layers.Dense(self.OUTPUT_DIM , activation=None),
            ])
            model.compile(optimizer='adam',
                          loss='mean_squared_error',
                          metrics=['mean_squared_error']
                          )
            self.ensemble.append(model)


    def add_data(self, obs, acs):
        inputs, targets = [], []
        for t in range(50):
            inp = np.concatenate((obs[t][:self.OBS_DIM],acs[t]))
            target = obs[t+1][:self.OBS_DIM] - obs[t][:self.OBS_DIM]
            bool_target = [1 if np.linalg.norm(target[i:i+2]) > 0 else -1 for i in [8,12,14,16]]
            inputs.append(inp)
            targets.append(np.concatenate((target,bool_target)))
        self.replay_buffer.add_samples( inputs, targets)


    def train(self, EPOCH=5):
        for i in range(self.B):
            x, y = self.replay_buffer.sample()
            self.ensemble[i].fit(x,y, epochs=EPOCH, shuffle=True, verbose=False)


    def select_actions(self, init_obs, n_samples, GRBF=True):
        if GRBF:
            actions = sample_random_trajectories(n_samples,4,50)
        else:
            actions = 2*np.random.random((n_samples, 50,self.ACS_DIM))-1
        if self.trained:
            pred_traj = self.predict_trajectory([[init_obs]]*n_samples, actions)
            std = np.std(pred_traj, axis=0)
            epistemic_uncertainty = np.sum( std, axis = (0,2))
            selected = np.argmax(epistemic_uncertainty)
            
            return actions[selected]
        else:
            self.trained = True
            return actions[0]


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
        pred_trajs = []
        for b in range(self.B):
            obs = true_traj[:,0,:self.OBS_DIM]
            pred_traj = [np.concatenate((obs,obs-true_traj[:,0,:self.OBS_DIM]),axis=1)]
            for t in range(50):
                inputs = np.concatenate((obs[:,:self.OBS_DIM],Acs[:,t]), axis=1)
                output = self.ensemble[b].predict(inputs)
                obs_pred = output[:,:self.OBS_DIM]+obs
                obs = self.filter(obs_pred, output, obs)
                pred_traj.append(np.concatenate((obs,obs-true_traj[:,0,:self.OBS_DIM]),axis=1))
            pred_trajs.append(pred_traj.copy())

        return np.array(pred_trajs)

    def predict_transition(self, true_traj, Acs):
        true_traj = np.array(true_traj)
        Acs = np.array(Acs)
        pred_trajs = []
        for b in range(self.B):
            obs = true_traj[:,0,:self.OBS_DIM]
            pred_traj = [obs]
            for t in range(50):
                inputs = np.concatenate((true_traj[:,t,:self.OBS_DIM],Acs[:,t]), axis=1)
                output = self.ensemble[b].predict(inputs)
                obs_pred = output[:,:self.OBS_DIM]+obs
                obs = self.filter(obs_pred, output, obs)
                pred_traj.append(obs.copy())
            pred_trajs.append(pred_traj.copy())
        return np.array(pred_trajs)
        
