import random
import numpy as np
import gym
from gym import spaces


class Dummy(gym.Env):
    """The Sokoban environment.
    """

    def __init__(self):
        # We set the space
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=10, shape=(84, 84, 1), dtype=np.uint8)  # joints + 64 * 64 image

        self.viewer = None

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self):

        self.agent_pos = np.zeros((84, 84, 1), dtype=np.uint8)
        return self.agent_pos

    def compute_reward(self):
        if (self.agent_pos == 10).all():
            reward = -1 
            done = True
        else:
            reward = self.agent_pos[0, 0, 0]
            done = False
        return reward, done

    def step(self, action=0):
        """Perform an agent action in the Environment
        """
        self.agent_pos += 1
        reward, done = self.compute_reward()

        info = {}

        return self.agent_pos, reward, done, info
