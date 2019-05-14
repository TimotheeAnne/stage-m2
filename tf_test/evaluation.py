import sys
import pickle 
sys.path.append('../multi-task-her-rl/src/reward_function/')
from reward_function import OracleRewardFuntion
import numpy as np

class Evaluator:
    def __init__(self, training_data, eval_data, logdir, OBS_DIM):
        self.logdir = logdir
        self.OBS_DIM = OBS_DIM
        self.oracle = OracleRewardFuntion(30)
        self.iteration = 0
        self.training_data = training_data
        with open( eval_data, 'rb') as f:
            self.eval_data = pickle.load(f)

    def eval(self, DE):
        if not self.training_data is None:
            self._eval(DE, "training_data")
        self._eval(DE, "evaluation_data")
        self.iteration += 1

    def _eval(self, DE, data_type):
        [true_traj, Acs] = self.training_data if data_type == "training_data" else self.eval_data
        true_traj = np.array(true_traj)
        
        traj_pred = DE.predict_trajectory(true_traj, Acs)
        trans_pred = DE.predict_transition(true_traj, Acs)
        
        predict_rewards = [[] for _ in range(DE.B)]

        true_rewards = self.oracle.eval_all_goals_from_state(true_traj[:,-1])
        for b in range(DE.B):
            predict_rewards[b] = self.oracle.eval_all_goals_from_state( traj_pred[b,-1,:])
                
        confusion_matrix = self.compute_confusion_matrix(true_rewards, predict_rewards)

        with open(self.logdir+"/eval_"+data_type+"_"+str(self.iteration)+".pk",'bw') as f:
            pickle.dump((traj_pred,trans_pred),f)

        with open(self.logdir+"/confusion_matrix_"+data_type+"_"+str(self.iteration)+".pk",'bw') as f:
            pickle.dump(confusion_matrix,f)


    def compute_confusion_matrix(self, true_rewards, predict_rewards):
        n_tasks = len(true_rewards[0])
        B = len(predict_rewards)
        confusion_matrix = np.zeros((B, n_tasks,2,2))
        for j in range(len(true_rewards)):
            for r in range(n_tasks):
                truth_value = int(true_rewards[j][r] == 0)
                for b in range(B):
                    predict_value = int(predict_rewards[b][j][r] == 0)
                    confusion_matrix[b][r][truth_value][predict_value] += 1 
        return confusion_matrix
