from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from dotmap import DotMap
import gym


from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC
import dmbrl.env


class ToyEnvConfigModule:
    ENV_NAME = "FetchReach-v1"
    TASK_HORIZON = 50 #200
    NTRAIN_ITERS = 20 #50
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 5
    MODEL_IN, MODEL_OUT = 12, 8
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
                "popsize": 100, #400
                "num_elites": 10, #40
                "max_iters": 5, #5
                "alpha": 0.1,
                "goal_dim": 3
            }
        }

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    def obs_cost_fn(self, obs, goal_pos):
        obj_pos = obs[:, :3]

        if isinstance(obs, np.ndarray):
            obj_goal_dist = np.linalg.norm(goal_pos - obj_pos, axis=1)
            return  obj_goal_dist
        else:
            obj_goal_dist = tf.linalg.norm(goal_pos - obj_pos, axis=1)
            # ~ return  tf.Print( obj_goal_dist, [obs], summarize=9)
            return  obj_goal_dist

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
            model.add(FC(200, input_dim=self.MODEL_IN, activation='swish', weight_decay=0.025)) #500
            model.add(FC(200, activation='swish', weight_decay=0.05))#500
            model.add(FC(200, activation='swish', weight_decay=0.075))#500
            model.add(FC(self.MODEL_OUT, weight_decay=0.01))
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
