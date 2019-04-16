from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from dotmap import DotMap

import time
from tqdm import tqdm

### Me
def convert_obs( obs):
    """
    Wrap the Dict observation {desired_goal:Box(2,), achieved_goal:Box(2,), observation:Box(9,)} into a (observation, desired_goal)
    """
    if isinstance(obs, dict):
        return ( obs['observation'],obs['desired_goal'])
    else:
        return obs, None
###

class Agent:
    """An general class for RL agents.
    """
    def __init__(self, params):
        """Initializes an agent.

        Arguments:
            params: (DotMap) A DotMap of agent parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .noisy_actions: (bool) Indicates whether random Gaussian noise will
                    be added to the actions of this agent.
                .noise_stddev: (float) The standard deviation to be used for the
                    action noise if params.noisy_actions is True.
        """
        self.env = params.env
        self.noise_stddev = params.noise_stddev if params.get("noisy_actions", False) else None

        if isinstance(self.env, DotMap):
            raise ValueError("Environment must be provided to the agent at initialization.")
        if (not isinstance(self.noise_stddev, float)) and params.get("noisy_actions", False):
            raise ValueError("Must provide standard deviation for noise for noisy actions.")

        if self.noise_stddev is not None:
            self.dU = self.env.action_space.shape[0]

    def sample(self, horizon, policy, cost_choice=0, record_fname=None):
        """Samples a rollout from the agent.

        Arguments:
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.
            record_fname: (str/None) The name of the file to which a recording of the rollout
                will be saved. If None, the rollout will not be recorded.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """
        video_record = record_fname is not None
        recorder = None if not video_record else VideoRecorder(self.env, record_fname)

        times, rewards = [], []
        O, A, reward_sum, done = [self.env.reset()], [], 0, False
        # ~ self.env.unwrapped.reset_task_goal(np.array([0,0,0]),0)
        
        ### Me
        I = []
        O[0], goal_pos = convert_obs(O[0])
        ###
        # ~ print("0", O[0])
        policy.reset()
        # ~ for t in tqdm(range(horizon)):
        for t in range(horizon):
            if video_record:
                recorder.capture_frame()
            start = time.time()

            action, MPC_info = policy.act(O[t],t, goal_pos, cost_choice)
            A.append(action)

            times.append(time.time() - start)

            if self.noise_stddev is None:
                obs, reward, done, info = self.env.step(A[t])
                ### Me
                obs,goal_pos = convert_obs(obs)
                ###
            else:
                action = A[t] + np.random.normal(loc=0, scale=self.noise_stddev, size=[self.dU])
                action = np.minimum(np.maximum(action, self.env.action_space.low), self.env.action_space.high)
                obs, reward, done, info = self.env.step(action)
                ### Me
                obs,goal_pos = convert_obs(obs)
                ###

            # ~ print(t+1, obs)
            I.append(MPC_info)
            O.append(obs)
            reward_sum += reward
            rewards.append(reward)

            if done:
                break

        if video_record:
            recorder.capture_frame()
            recorder.close()

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
            "goal": goal_pos,
            "info": I
        }


