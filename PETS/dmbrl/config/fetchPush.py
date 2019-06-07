from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap
import gym
import gym_myFetchPush
import gym_flowers

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC
import dmbrl.env


class ToyEnvConfigModule:
    ENV_NAME = "FetchPush-v1"
    # ~ ENV_NAME = "MultiTaskFetchArmNLP1-v0"
    # ~ ENV_NAME = "myFetchPush-v0"
    TASK_HORIZON = 50 #200
    NTRAIN_ITERS = 2 #50
    NROLLOUTS_PER_ITER = 0
    PLAN_HOR = 5
    # ~ MODEL_IN, MODEL_OUT = 2, 1
    # ~ MODEL_IN, MODEL_OUT = 10, 6
    # ~ MODEL_IN, MODEL_OUT = 12, 8
    # ~ MODEL_IN, MODEL_OUT = 27, 23
    MODEL_IN, MODEL_OUT = 29, 25
    # ~ MODEL_IN, MODEL_OUT = 30, 26
    GP_NINDUCING_POINTS = 200

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000 #2000
            },
            "CEM": {
                "popsize": 500, #400
                "num_elites": 50, #40
                "max_iters": 5, #5
                "alpha": 0.1,
                "goal_dim": 3
            }
        }

    @staticmethod
    def obs_preproc(obs):
        return obs 
        
    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred


    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs
        # ~ return np.concatenate( (next_obs[:,:-1] - obs[:,:-1], next_obs[:,-1:] ), axis=1)

    def obs_cost_fn(self, obs, goal_pos):
        obj_pos = obs[:, 3:6]

        if isinstance(obs, np.ndarray):
            obj_goal_dist = np.linalg.norm(goal_pos - obj_pos, axis=1)
            return  obj_goal_dist
        else:
            obj_goal_dist = tf.linalg.norm(goal_pos - obj_pos, axis=1)
            return  obj_goal_dist
            # ~ return tf.print( obj_goal_dist, [obs], summarize=23)

    @staticmethod
    def ac_cost_fn(acs):
        if isinstance(acs, np.ndarray):
            return 0. * np.sum(np.square(acs), axis=1)
        else:
            return 0. * tf.reduce_sum(tf.square(acs), axis=1)

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))
        if not model_init_cfg.get("load_model", False):
            # ~ activation = 'swish'
            activation = 'ReLU'
            model.add(FC(200, input_dim=self.MODEL_IN, activation=activation, weight_decay=0.00025))
            model.add(FC(200, activation=activation, weight_decay=0.0005))
            model.add(FC(200, activation=activation, weight_decay=0.0005))
            model.add(FC(self.MODEL_OUT, weight_decay=0.00075))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        return model

    def gp_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model",
            kernel_class=get_required_argument(model_init_cfg, "kernel_class", "Must provide kernel class"),
            kernel_args=model_init_cfg.get("kernel_args", {}),
            num_inducing_points=get_required_argument(
                model_init_cfg, "num_inducing_points", "Must provide number of inducing points."
            ),
            sess=self.SESS
        ))
        return model


CONFIG_MODULE = ToyEnvConfigModule
