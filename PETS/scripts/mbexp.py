from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys

sys.path.insert(0, '/home/tim/Documents/stage-m2/mypets')
#sys.path.insert(0, '/home/tanne/Experiment/stage-m2/mypets')

import os
import argparse
import pprint
import pickle 

from dotmap import DotMap

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config

from scipy.io import loadmat, savemat

import plot

log_dir = "./log"

# ~ env = 'armball'
# ~ env = 'MultiTaskFetchArm'
# ~ env = 'fetchReach'
env = 'fetchPush'

mycfg = {}

# args
parser = argparse.ArgumentParser()
parser.add_argument('-run', type=str, default=0)
parser.add_argument('-Nsamples', type=str, default=0)
parser.add_argument('-window_size', type=str, default=1)
args = parser.parse_args()

head = "/home/tim/Documents/stage-m2/mypets/data/" 

window_size = int(args.window_size)
run = int(args.run)

mycfg['run'] = run
mycfg['window_size'] = window_size

mycfg['Nsamples'] = [1,2,5,10,20,50,100,200,500,900][int(args.Nsamples)]
mycfg['pre_data'] = True

# ~ mycfg['pred_eval'] = head + "transition_multiTask_1_eval.pk"
# ~ mycfg['init_train'] = head + "transition_multiTask_1_train.pk"

# ~ mycfg['pred_eval'] = head + "transition_multitask_8d_nonoise_eval.pk"
# ~ mycfg['init_train'] = head + "transition_multitask_8d_nonoise_train.pk"

mycfg['init_train'] =  head + "transition_multi25_0noise_train.pk"
mycfg['pred_eval'] = head + "transition_multi25_0noise_eval.pk"

mycfg['train'] = 1
mycfg['eval'] = 0
        
cfg = create_config(env, "MPC", DotMap({'model-type':'DE'}), 
                    [('exp_cfg.log_cfg.neval',0),('exp_cfg.exp_cfg.ninit_rollouts',0)], log_dir)

cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg, window_size)
exp = MBExperiment(cfg.exp_cfg, mycfg)

# Create log dir
os.makedirs(exp.logdir)
with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
    f.write(pprint.pformat(cfg.toDict()))
with open(os.path.join(exp.logdir, "myconfig.pk"), "bw") as f:
    pickle.dump( mycfg, f)
f.close()

# Run the experiment
exp.run_experiment()

#Do the ploting !


listdir = os.listdir(log_dir)

listdir.sort()
sub_dir = listdir[-2]


# ~ plot.plot_xp( log_dir, sub_dir)
