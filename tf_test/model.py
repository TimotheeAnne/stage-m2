import tensorflow as tf
import numpy as np
import pickle 
from tqdm import tqdm
import matplotlib.pyplot as plt 
import os 
import random

from replay_buffer import ReplayBuffer
from GRBF import sample_random_trajectories

FIGSIZE = (16,9)

class Ensemble:
    """ Deterministic ensemble transition model """
    def __init__(self, obs_d, acs_d, out_d, logdir, B=5, reg = 0, objects=None):
        self.ACS_DIM = acs_d
        self.OUTPUT_DIM = out_d
        self.OBS_DIM = obs_d
        self.INPUT_DIM = self.ACS_DIM + self.OBS_DIM
        self.REG = reg
        self.B = B
        self.replay_buffer = ReplayBuffer()
        self.ensemble = []
        self.logdir = logdir 
        self.trained = False
        self.iterations = 0
        self.x_eval = []
        self.y_eval = []
        self.history = [ [] for _ in range(self.B)]
        self.objects = range(5) if objects is None else objects  
        
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

    def add_episodes(self, obs, acs):
        assert( len(obs) == len(acs))
        for j in range(len(obs)):
            self.add_episode( obs[j], acs[j])


    def add_episode(self, obs, acs, returns=False):
        inputs, targets = [], []
        for t in range(len(acs)):
            inp = np.concatenate((obs[t][:self.OBS_DIM],acs[t]))
            target = obs[t+1][:self.OBS_DIM] - obs[t][:self.OBS_DIM]
            bool_target = [1 if np.linalg.norm(target[i:i+2]) > 0 else -1 for i in [8,12,14,16]]
            inputs.append(inp)
            targets.append(np.concatenate((target,bool_target)))
        if returns:
            return (inputs,targets)
        else:
            self.replay_buffer.add_samples( inputs, targets)


    def add_validation(self, obs, acs):
        assert( len(obs) == len(acs))
        x ,y = [], []
        for j in range(len(obs)):
            (inputs,targets) = self.add_episode( obs[j], acs[j], returns=True)
            x += inputs
            y += targets
        self.x_eval = np.array(x)
        self.y_eval = np.array(y)


    def train(self, EPOCH=5, verbose=False, validation=False, early_stopping = True, sampling='choice'):
        sampling_function = np.random.choice if sampling=='choice' else random.sample 
        for b in range(self.B):
            x, y = self.replay_buffer.sample(sampling_function, self.objects)
            if validation:
                es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2,
                    verbose=0, mode='auto')

                callbacks = [es] if early_stopping else []
                
                history = self.ensemble[b].fit(x,y, epochs=EPOCH, validation_split = 0.1,
                                    shuffle=True, verbose=verbose, callbacks=callbacks)
                
                self.history[b].append(history.history)
            else:
                self.ensemble[b].fit(x,y, epochs=EPOCH, shuffle=True, verbose=verbose)
        self.trained = True


    def save_ensemble(self):
        for b in range(self.B):
            self.ensemble[b].save_weights(self.logdir+'/model'+str(b)+'.h5')


    def plot_training(self):
        MSE = []
        Val_MSE = []
        for b in range(self.B):
            mse = []
            val_mse = []
            for data in self.history[b]:
                mse += list(data['val_mean_squared_error'])
                val_mse += list(data['mean_squared_error'])
            MSE.append(mse)
            Val_MSE.append(val_mse)
        self.plot_MSE(MSE, Val_MSE)


    def plot_histogram(self, data):
        fig, ax = plt.subplots(figsize=FIGSIZE) 
        plt.hist(data)
        plt.xscale('log')
        plt.savefig(self.logdir+'/uncertitude_'+str(self.iterations)+".svg", format='svg')
        plt.close(fig)


    def select_actions(self, init_obs, n_samples, n_elites, Tmax=50, GRBF=True, exploration=False):
        assert( n_elites <= n_samples)
        if GRBF:
            actions = np.array(sample_random_trajectories(n_samples,4,Tmax))
        else:
            actions = 2*np.random.random((n_samples, Tmax, self.ACS_DIM))-1
        if exploration:
            pred_traj = self.predict_trajectory([[init_obs]]*n_samples, actions)
            std = np.std(pred_traj, axis=0)[:,:,:18]
            norm = np.mean(std, axis=1)
            for t in range(len(norm)):
                for d in range(len(norm[t])):
                    if norm[t][d] == 0:
                        norm[t][d] = 1
            epistemic_uncertainty = np.mean( np.swapaxes(std,0,1)/norm, axis = (1,2))
            sorted_indices = np.argsort(epistemic_uncertainty)
            # ~ self.plot_histogram(epistemic_uncertainty)
            self.iterations += 1
            actions = actions[sorted_indices[-n_elites:]]
            eu = epistemic_uncertainty[sorted_indices[-n_elites:]]
            pred_traj = pred_traj[:,:,sorted_indices[-n_elites:]] 
            return actions, eu, pred_traj
        else: 
            return actions


    def save_exploration(self, actions, uncertainty, observations, pred):
        with open(os.path.join(self.logdir, "exploration.pk"), "ab") as f:
            pickle.dump( [actions, uncertainty, observations, pred], f)


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
            for t in range(np.shape(Acs)[1]):
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
            for t in range(np.shape(Acs)[1]):
                inputs = np.concatenate((true_traj[:,t,:self.OBS_DIM],Acs[:,t]), axis=1)
                output = self.ensemble[b].predict(inputs)
                obs_pred = output[:,:self.OBS_DIM]+true_traj[:,t,:self.OBS_DIM]
                obs = self.filter(obs_pred, output, obs)
                pred_traj.append(obs.copy())
            pred_trajs.append(pred_traj.copy())
        return np.array(pred_trajs)


    def plot_MSE(self, MSE, Val_MSE):
        colors = ['crimson','royalblue','forestgreen','darkorange','orchid']
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for b in range(self.B):
            if b == 0:
                plt.plot( MSE[b], label="validation "+str(b), color = colors[b], ls=':')
            else:
                plt.plot( MSE[b], color = colors[b], ls=':')
            plt.plot( Val_MSE[b], label="training "+str(b), color = colors[b])
        plt.legend()
        plt.yscale('log')
        plt.xlabel('epochs')
        plt.ylabel('MSE')
        with open(os.path.join(self.logdir, "MSE.svg"), "bw") as f:
            fig.savefig(f, format='svg')
        plt.close(fig)
