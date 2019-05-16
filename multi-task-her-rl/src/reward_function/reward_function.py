import numpy as np
# ~ from mpi4py import MPI
from task_reward_functions import *


def eucl_dist(pos1, pos2):
    return np.linalg.norm(pos1 - pos2, ord=2)

def oracle_reward_function_all(task_instr, state):
    # first 31 elements of state represent the state at time t
    # last 31 elements of state represent the state à time t minus the initial state

    if state.ndim == 1:
        state = np.expand_dims(state, 0)
    
    obs_dims = len(state[0])
    half = obs_dims // 2
    
    rewards = - np.ones([state.shape[0], len(task_instr)])
    for i in range(state.shape[0]):
        for j in range(len(task_instr)):
            rewards[i, j] = task_instr[j](state[i][:half],state[i][half:] )

    return rewards

def oracle_reward_function(task_instr, state, goal_ids):
    if state.ndim == 1:
        state = np.expand_dims(state, 0)

    obs_dims = len(state[0])
    half = obs_dims // 2
    
    rewards = - np.ones([state.shape[0]])

    for i in range(state.shape[0]):
        rewards[i] = task_instr[goal_ids[i]](state[i][:half],state[i][half:])
    return rewards


class OracleRewardFuntion:
    def __init__(self, nb_goals):
        self.nb_goals = nb_goals
        self.task_instructions = task_instructions[:nb_goals]

    def predict(self, state, goal_ids):
        if goal_ids.ndim == 2:
            goal_ids = goal_ids.squeeze().astype(np.int)
        return oracle_reward_function(self.task_instructions, state, goal_ids)

    def eval_all_goals_from_state(self, state):
        all_predictions = oracle_reward_function_all(self.task_instructions, state)
        return all_predictions.squeeze()

    def eval_all_goals_from_whole_episode(self, episode):
        rollout_batch_size = len(episode['o'][0])
        Successes = []
        for i in range(rollout_batch_size):
            successes = []
            for t in range(len(episode['o'])):
                all_goal_successes = (self.eval_all_goals_from_state(episode['o'][t][i]) == 0).astype(np.float)
                successes.append(all_goal_successes)
            Successes.append(successes)
        return np.array(Successes)

    def eval_all_goals_from_episode(self, episode):
        rollout_batch_size = len(episode['o'][0])
        successes = []
        for i in range(rollout_batch_size):
            all_goal_successes = (self.eval_all_goals_from_state(episode['o'][-1][i]) == 0).astype(np.float)
            successes.append(all_goal_successes)
        return np.array(successes)

    def eval_goal_from_episode(self, episode, goal_id):
        rollout_batch_size = len(episode['o'][0])
        successes = []
        for i in range(rollout_batch_size):
            success = (oracle_reward_function(self.task_instructions, episode['o'][-1][i], np.array([goal_id])) == 0).astype(np.int)[0]
            successes.append(success)
        return np.array(successes)


