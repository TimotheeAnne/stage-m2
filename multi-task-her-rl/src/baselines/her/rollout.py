from collections import deque
import time

import pickle
import numpy as np
from mpi4py import MPI
from mujoco_py import MujocoException

from src.reward_function.reward_function import OracleRewardFuntion
from .util import convert_episode_to_batch_major, store_args
import time

class RolloutWorker:

    @store_args
    def __init__(self, make_env, policy, dims, logger, nb_goals,T, eval, goal_sampling_policy=None,
                 rollout_batch_size=1, exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, save=False,  **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            language_model(object): the language model to decode strings into goals
            goal_sampler (object): tracks list of discovered goals and samples the agent goals
            T (int): number of timesteps in an episode
            eval (bool): whether it is an evaluator rollout worker or not
            nb_instr (int): number of instructions
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]
        self.Q_history = deque(maxlen=history_len)

        self.nb_goals = nb_goals
        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations

        self.oracle_reward_function = OracleRewardFuntion(self.nb_goals)

        # ~ if self.eval:
            # one queue for each goal/instruction.
        self.returns_histories = [deque(maxlen=history_len) for _ in range(self.nb_goals)]

        self.rollout_batch_size = rollout_batch_size

        self.nb_cpu = MPI.COMM_WORLD.Get_size()
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.tested_goal_counter = 0
        self.epoch_start_time = time.time()
        self.start_time = time.time()

        self.reset_all_rollouts()
        self.clear_history()


    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        self.initial_o[i] = self.envs[i].reset()
        # sample all goals from the goal_sampler of rank 0 (the only that contains data)
        goals = []
        if self.rank == 0:
            for _ in range(self.nb_cpu):
                if not self.eval:
                    goal = np.zeros([self.nb_goals])
                    goal[np.random.randint(0, self.nb_goals)] = 1
                    goals.append(goal)
                else:
                    # if eval, test on every possible goal
                    goal = np.zeros([self.nb_goals])
                    goal[self.tested_goal_counter] = 1
                    goals.append(goal)
        goal = MPI.COMM_WORLD.scatter(goals, root=0)
        self.g[i] = goal

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)


    def generate_rollouts(self, index=None):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        if not index is None:
            self.tested_goal_counter = index
        self.reset_all_rollouts()
        
        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        o[:] = self.initial_o

        # generate episodes
        obs, acts, goals = [], [], []
        Qs = []
        timee = time.time()

        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            # compute new states and observations
            
            
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    o_new[i], _, _, _ = self.envs[i].step(u[i])
                    # ~ o_new[i], _, _, _ = self.envs[i].step([0,1,0,0])

                    if self.render:
                        self.envs[i].render()
                except MujocoException as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                #self.logger.warning('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            o[...] = o_new

        obs.append(o.copy())
        self.initial_o[:] = o


        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       )
        

            
        # compute rewards if self.eval
        if self.eval:
            # ~ goals_reached_ids = None
            returns = np.array(self.oracle_reward_function.eval_goal_from_episode(episode, goal_id=self.tested_goal_counter))
            self.returns_histories[self.tested_goal_counter].append(returns.mean())
            # switch to next goal to be tested
            self.tested_goal_counter += 1
            goals_reached_ids = returns

        else:        
            goal_id = np.where( self.g[0] == 1.)[0][0]
            returns = np.array(self.oracle_reward_function.eval_goal_from_episode(episode, goal_id=goal_id))
            self.returns_histories[goal_id].append(returns.mean())
            rewards = self.oracle_reward_function.eval_all_goals_from_episode(episode)
            goals_reached_ids = []
            for i in range(rewards.shape[0]):
                goals_reached_ids.append([])
                for j in range(rewards.shape[1]):
                    if rewards[i][j] == 1:
                        goals_reached_ids[i].append(j)

        # stats
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode), goals_reached_ids


    def current_success_rate(self):
        if self.eval:
            out = []
            for return_hist in self.returns_histories:
                if len(return_hist) == 0:
                    out.append(0)
                else:
                    out.append(np.mean(return_hist))
            return np.mean(out)
        else:
            # this should never happen
            return None

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        if self.eval:
            for return_hist in self.returns_histories:
                return_hist.clear()
            self.tested_goal_counter = 0
            self.epoch_start_time = time.time()
        self.Q_history.clear()

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)


    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        if True or self.eval:
            for i in range(self.nb_goals):
                logs+= [('success_goal_' + str(i), np.mean(self.returns_histories[i]))]
                
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes * self.nb_cpu)]
        if prefix == 'eval':
            logs += [('total_time (s)', time.time() - self.start_time)]
            logs += [('time_per_epoch (s)', time.time() - self.epoch_start_time)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs


    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)
