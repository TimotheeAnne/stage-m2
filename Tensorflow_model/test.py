import sys
import pickle 
sys.path.append('../multi-task-her-rl/src/reward_function/')
from reward_function_model import OracleRewardFuntion
import numpy as np


with open( "./data/ArmToolsToyR_eval.pk", 'rb') as f:
    eval_data = pickle.load(f)

[true_traj, Acs] = eval_data
true_traj = np.array(true_traj)
oracle = OracleRewardFuntion(30)
true_rewards = oracle.eval_all_goals_from_state(true_traj[:,-1])

with open(" eval_reward.pk",'bw') as f:
    pickle.dump(true_rewards,f)

