from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from time import time, localtime, strftime

import tensorflow as tf
import numpy as np
import pickle

from scipy.io import savemat
from dotmap import DotMap

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.misc.Agent import Agent


class MBExperiment:
    def __init__(self, params, myparams):
        """Initializes class instance.

        Argument:
            params (DotMap): A DotMap containing the following:
                .sim_cfg:
                    .env (gym.env): Environment for this experiment
                    .task_hor (int): Task horizon
                    .stochastic (bool): (optional) If True, agent adds noise to its actions.
                        Must provide noise_std (see below). Defaults to False.
                    .noise_std (float): for stochastic agents, noise of the form N(0, noise_std^2I)
                        will be added.

                .exp_cfg:
                    .ntrain_iters (int): Number of training iterations to be performed.
                    .nrollouts_per_iter (int): (optional) Number of rollouts done between training
                        iterations. Defaults to 1.
                    .ninit_rollouts (int): (optional) Number of initial rollouts. Defaults to 1.
                    .policy (controller): Policy that will be trained.

                .log_cfg:
                    .logdir (str): Parent of directory path where experiment data will be saved.
                        Experiment will be saved in logdir/<date+time of experiment start>
                    .nrecord (int): (optional) Number of rollouts to record for every iteration.
                        Defaults to 0.
                    .neval (int): (optional) Number of rollouts for performance evaluation.
                        Defaults to 1.
        """
        self.env = get_required_argument(params.sim_cfg, "env", "Must provide environment.")
        self.task_hor = get_required_argument(params.sim_cfg, "task_hor", "Must provide task horizon.")
        if params.sim_cfg.get("stochastic", False):
            self.agent = Agent(DotMap(
                env=self.env, noisy_actions=True,
                noise_stddev=get_required_argument(
                    params.sim_cfg,
                    "noise_std",
                    "Must provide noise standard deviation in the case of a stochastic environment."
                )
            ))
        else:
            self.agent = Agent(DotMap(env=self.env, noisy_actions=False))

        self.ntrain_iters = get_required_argument(
            params.exp_cfg, "ntrain_iters", "Must provide number of training iterations."
        )
        self.nrollouts_per_iter = params.exp_cfg.get("nrollouts_per_iter", 1)
        self.ninit_rollouts = params.exp_cfg.get("ninit_rollouts", 1)
        self.policy = get_required_argument(params.exp_cfg, "policy", "Must provide a policy.")

        self.logdir = os.path.join(
            get_required_argument(params.log_cfg, "logdir", "Must provide log parent directory."),
            strftime("%Y-%m-%d--%H:%M:%S", localtime())
        )
        self.nrecord = params.log_cfg.get("nrecord", 0)
        self.neval = params.log_cfg.get("neval", 1)
        
        ## Me
        self.pre_data = myparams['pre_data']
        self.pred_eval = myparams['pred_eval']
        self.init_train = myparams['init_train']
        self.train = myparams['train']
        self.eval = myparams['eval']
        
        """ to bound the obs space"""
        self.minimum = None
        self.maximum = None 
 ## Me
    def compute_hand_pos(self,  arm_pos):
            arm_lengths = np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05])
            angles = np.cumsum(arm_pos)
            angles_rads = np.pi * angles
            hand_pos = np.array([np.sum(np.cos(angles_rads) * arm_lengths),
                                np.sum(np.sin(angles_rads) * arm_lengths)
                                ])
            return hand_pos

    def true_env(self, inputs):
        arm_pos = inputs[:7]
        achieved_goal = inputs[7:9]
        action = inputs[9:16]
        hand_pos = self.compute_hand_pos(arm_pos)
        object_handled = False
        if np.linalg.norm(hand_pos - achieved_goal, ord=2) < 0.1 :
            object_handled = True

        arm_pos = np.clip(arm_pos + action / 10,a_min=-np.ones(7),a_max=np.ones(7))
        hand_pos = self.compute_hand_pos( arm_pos)

        if object_handled:
            achieved_goal = hand_pos
        new_obs = np.concatenate( (arm_pos, achieved_goal))
        return new_obs

    def evaluate_model_ArmBall(self, one_start=True):
        neval = 100
        if one_start:
            obs = [np.array([0.,0.,0.,0.,0.,0.,0.,  0.6,0.6]) for _ in range(neval)]
        else:
            arm_pos = [np.random.uniform(-1,1,7) for _ in range(neval)]
            hand_pos = [self.compute_hand_pos( arm_pos[i]) for i in range(neval)]
            obs = [np.concatenate((arm_pos[i],hand_pos[i])) for i in range(neval)]
        acs = [ np.random.uniform(-1, 1, 7) for _ in range(neval)]
        inputs = np.array( [[np.concatenate(( obs[i],acs[i])) for i in range(neval)] for _ in range(5)])

        """
        [ensemble_size, batch_size, obs + acs]
        """
        pred = self.policy.model.predict( inputs)
        error = 0
        for j in range(neval):
            for i in range(5):
                true_obs = self.true_env( inputs[i][j])
                error += np.linalg.norm( true_obs - obs[j] - pred[0][i][j])
        print("error: ","one start"*one_start+"multi start"*(1-one_start), error)
        return error

    def evaluate_model_random(self, model_eval_samples):
        """ Evaluation of the model based on unseen data""" 
        neval = len(model_eval_samples)
        task_horizon = len(model_eval_samples[0]['ac'])
        # construct evaluation samples 

        true_obs = [[] for _ in range(neval)]
        inputs = []
        
        for j in range(neval):
            for k in range(task_horizon ):
                obs = model_eval_samples[j]['obs'][k]
                ac = model_eval_samples[j]['ac'][k]
                inputs.append( np.concatenate(( obs,ac)) ) 
                true_obs[j].append(  model_eval_samples[j]['obs'][k] )
            true_obs[j].append(  model_eval_samples[j]['obs'][task_horizon] )
        
        
        
        inputs = np.array([ inputs for _ in range(5)])

        
        # Evaluation [ensemble_size, batch_size, obs + acs]
        pred = self.policy.model.predict( inputs)
        error = [0 for _ in range(neval)]
        for j in range(neval):
            for k in range(task_horizon):
                for i in range(5):
                    error[j] += np.linalg.norm( true_obs[j][k+1] - true_obs[j][k] - pred[0][i][j*task_horizon+k])
        error = np.array(error) / (5*task_horizon)
        print("model error: ", np.mean(error), np.std(error))
        return np.mean(error), np.std(error)
        
    def evaluate_model_FetchPush_transition_pred(self, eval_file):
        f = open(eval_file, 'rb')
        [true_traj,Acs] = pickle.load(f)
        f.close()
        
        boo = "eval" in eval_file
        
        curr_states = [[ true_traj[j][0] for _ in range(5)] for j in range(len(true_traj))]
        traj_pred = [curr_states]
        error = []
        for t in range(50):
            inputs = [[ np.concatenate((true_traj[j][t],Acs[j][t])) for j in range(len(true_traj))] for i in range(5)]
            inputs = np.array(inputs)
            pred = self.policy.model.predict(inputs)
            curr_states = [[ pred[0][i][j]+true_traj[j][t] for i in range(5)] for j in range(len(true_traj))]
            traj_pred.append(curr_states)
            if boo:
                print(t, "trans input: ", inputs[0][0])
                print(t, 'trans true', true_traj[0][t+1])
                print(t, "trans pred: ", curr_states[0][0])
                
        error = [[[ np.linalg.norm(true_traj[j][t] - traj_pred[t][j][i]) for i in range(5)] for t in range(51)] for j in range(len(true_traj))]
        
        return [traj_pred, true_traj, error]
        
    def bound_obs(self, obs):
        if self.minimum is None:
            f = open(self.init_train, 'rb')
            [true_traj,_] = pickle.load(f)
            f.close()
            self.minimum = np.min(true_traj,axis=(0,1))
            self.maximum = np.max(true_traj,axis=(0,1))
        # ~ return np.min( [ self.maximum, np.max([self.minimum, obs], axis=0)], axis=0)
        return obs
        
    def evaluate_model_FetchPush_traj_pred(self, eval_file, with_moved = False):
        def compute_next_obs( pred, obs):
            moved = pred[-1] >= 0.5
            if moved:
                return np.concatenate( (pred[:-1]+obs[:-1],[1]))
            else:
                return np.concatenate([ obs[:3]+pred[:3], # gripper position
                                        obs[3:6], # cube position
                                        obs[3:6]-( obs[:3] + pred[:3]), # relative position of cub/gripper
                                        pred[9:11]+obs[9:11], # gripper state
                                        obs[11:14], # cube rot
                                        -obs[20:23], # relative cube velocitie, equal to -Vgripper if Vcube=0
                                        obs[17:20], # rotation velocities
                                        obs[20:25]+pred[20:25], # gripper velocities
                                        [0] # cube moving
                                            ] )

        boo = "eval" in eval_file
        
        f = open(eval_file, 'rb')
        [true_traj,Acs] = pickle.load(f)
        f.close()
        curr_states = [[ true_traj[j][0] for _ in range(5)] for j in range(len(true_traj))]
        traj_pred = [curr_states]
        error = []
        for t in range(50):
            inputs = [[ np.concatenate((curr_states[j][i],Acs[j][t])) for j in range(len(true_traj))] for i in range(5)]
            inputs = np.array(inputs)
            pred = self.policy.model.predict(inputs)

            if with_moved:
                curr_states = [[ self.bound_obs(compute_next_obs( pred[0][i][j], curr_states[j][i])) for i in range(5)] for j in range(len(true_traj))]
            else:
                curr_states = [[ self.bound_obs(pred[0][i][j]+curr_states[j][i]) for i in range(5)] for j in range(len(true_traj))]
                #curr_states = [[ pred[0][i][j] for i in range(5)] for j in range(len(true_traj))]
            traj_pred.append(curr_states)
            if boo:
                print(t, "input: ", inputs[0][0])
                print(t, 'true', true_traj[0][t+1])
                print(t, "pred: ", curr_states[0][0])

        error = [[[ np.linalg.norm(true_traj[j][t] - traj_pred[t][j][i]) for i in range(5)] for t in range(51)] for j in range(len(true_traj))]
        
        return [traj_pred, true_traj, error]
    ##


    def run_experiment(self):
        """Perform experiment.
        """
        os.makedirs(self.logdir, exist_ok=True)

        traj_obs, traj_acs, traj_rets, traj_rews, traj_error, traj_goals, traj_rets = [], [], [], [], [], [], []
        learning_obs, learning_acs, learning_rets, learning_rews, learning_error, learning_goals = [], [], [], [], [], []
        model_eval,  info, learning_info = [], [], []
        model_eval_trans, model_train_trans = [], []
        model_eval_samples, model_eval_traj, model_train_traj = [], [], []
        
        # Perform initial rollouts
        samples = []
        
        if self.nrecord:
            self.pre_data = 0
            
        if not self.pre_data:
            for i in range(self.ninit_rollouts):
                samples.append(
                    self.agent.sample(
                        self.task_hor, self.policy
                    )
                )
                traj_obs.append(samples[-1]["obs"])
                traj_acs.append(samples[-1]["ac"])
                traj_rews.append(samples[-1]["rewards"])
                traj_rets.append(samples[-1]["reward_sum"])
                print("rewards: ", samples[-1]["reward_sum"])

        if self.nrecord == 0:
            for i in range(0):
                model_eval_samples.append(
                    self.agent.sample(
                        self.task_hor, self.policy
                    )
                )
            
            # ~ model_eval.append(self.evaluate_model_random(model_eval_samples))
            model_train_traj.append(self.evaluate_model_FetchPush_traj_pred(self.init_train))
            model_eval_traj.append(self.evaluate_model_FetchPush_traj_pred(self.pred_eval))
            model_train_trans.append(self.evaluate_model_FetchPush_transition_pred(self.init_train))
            model_eval_trans.append(self.evaluate_model_FetchPush_transition_pred(self.pred_eval))
            
        if not self.pre_data:
            if self.ninit_rollouts > 0:
                self.policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples]
                )
        else:
            print("Learn with the pre computed samples")
            f = open(self.init_train,'rb')
            [obs,ac] = pickle.load(f)
            f.close()

            self.policy.train( np.array(obs), np.array(ac), [])
        
        print("Random successes :", np.sum( np.array(traj_rets) != -50.) )
        if self.nrecord == 0:
            # ~ model_eval.append(self.evaluate_model_random(model_eval_samples))
            model_train_traj.append(self.evaluate_model_FetchPush_traj_pred(self.init_train))
            model_eval_traj.append(self.evaluate_model_FetchPush_traj_pred(self.pred_eval))
            model_train_trans.append(self.evaluate_model_FetchPush_transition_pred(self.init_train))
            model_eval_trans.append(self.evaluate_model_FetchPush_transition_pred(self.pred_eval))
            
        # Training loop
        for i in range(self.ntrain_iters):
            print("####################################################################")
            print("Starting training iteration %d." % (i + 1))

            iter_dir = os.path.join(self.logdir, "train_iter%d" % (i + 1))
            os.makedirs(iter_dir, exist_ok=True)

            samples = []
            for j in range(self.nrecord):
                samples.append(
                    self.agent.sample(
                        self.task_hor, self.policy, cost_choice=0,
                        record_fname = os.path.join(iter_dir, "rollout%d.mp4" % j)
                    )
                )

            if self.nrecord > 0:
                for item in filter(lambda f: f.endswith(".json"), os.listdir(iter_dir)):
                    os.remove(os.path.join(iter_dir, item))

            ### Me
            neval = self.neval if ((i+1) % 10) == 0 else 0
            for j in range(neval+ self.nrollouts_per_iter - self.nrecord):
                # learning
                if j < self.nrollouts_per_iter:
                    samples.append(
                        self.agent.sample(
                            self.task_hor, self.policy, cost_choice=self.train
                        )
                    )
                #evaluation
                else:
                    samples.append(
                        self.agent.sample(
                            self.task_hor, self.policy, cost_choice=self.eval
                        )
                    )

            print("Rewards obtained:", [sample["reward_sum"] for sample in samples])
            

            if neval >0:
                traj_obs.append([sample["obs"] for sample in samples[self.nrollouts_per_iter:]])
                traj_acs.append([sample["ac"] for sample in samples[self.nrollouts_per_iter:]])
                traj_rets.append([sample["reward_sum"] for sample in samples[self.nrollouts_per_iter:]])
                traj_rews.append([sample["rewards"] for sample in samples[self.nrollouts_per_iter:]])
                traj_goals.append([sample["goal"] for sample in samples[self.nrollouts_per_iter:]])
                info.append([sample["info"] for sample in samples[self.nrollouts_per_iter:]])

            elif self.nrecord >0:
                traj_obs.append([sample["obs"] for sample in samples])
                traj_acs.append([sample["ac"] for sample in samples])
                traj_rets.append([sample["reward_sum"] for sample in samples])
                traj_rews.append([sample["rewards"] for sample in samples])
                traj_goals.append([sample["goal"] for sample in samples])
                info.append([sample["info"] for sample in samples])

            if self.nrollouts_per_iter > 0:
                learning_obs.append([sample["obs"] for sample in samples[:self.nrollouts_per_iter]])
                learning_acs.append([sample["ac"] for sample in samples[:self.nrollouts_per_iter]])
                learning_rets.append([sample["reward_sum"] for sample in samples[:self.nrollouts_per_iter]])
                learning_rews.append([sample["rewards"] for sample in samples[:self.nrollouts_per_iter]])
                learning_goals.append([sample["goal"] for sample in samples[:self.nrollouts_per_iter]])
                learning_info.append([sample["info"] for sample in samples[:self.nrollouts_per_iter]])

            samples = samples[:self.nrollouts_per_iter]

            self.policy.dump_logs(self.logdir, iter_dir)
            savemat(
                os.path.join(self.logdir, "logs.mat"),
                {
                    "observations": traj_obs,
                    "actions": traj_acs,
                    "returns": traj_rets,
                    "rewards": traj_rews,
                    "errors" : model_eval,
                    "transition_error": [model_train_trans,model_eval_trans],
                    "traj_error": [model_train_traj,model_eval_traj],
                    "goals": traj_goals,
                    "info": info,

                    "learning_observations": learning_obs,
                    "learning_actions": learning_acs,
                    "learning_returns": learning_rets,
                    "learning_rewards": learning_rews,
                    "learning_goals": learning_goals,
                    "learning_info": learning_info
                }
            )
            # Delete iteration directory if not used
            if len(os.listdir(iter_dir)) == 0:
                os.rmdir(iter_dir)

            if i < self.ntrain_iters - 1:
                self.policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples]
                )
            
                if self.nrecord == 0:
                    # ~ model_eval.append(self.evaluate_model_random(model_eval_samples))
                    model_train_traj.append(self.evaluate_model_FetchPush_traj_pred(self.init_train))
                    model_eval_traj.append(self.evaluate_model_FetchPush_traj_pred(self.pred_eval))
                    model_train_trans.append(self.evaluate_model_FetchPush_transition_pred(self.init_train))
                    model_eval_trans.append(self.evaluate_model_FetchPush_transition_pred(self.pred_eval))
