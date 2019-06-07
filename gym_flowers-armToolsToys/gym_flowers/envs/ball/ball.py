import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import random
import gizeh


class Ball(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, object_initial_pos=np.array([0.6, 0.6]), object_size=0.1, n_timesteps=100,
                 stochastic=False, env_noise=0):
        """Initializes a new ArmBall environment.

                Args:
                    object_initial_pos (np_array): initial pose for the ball
                    object_size (float): ball size, maximum distance to catch the ball
                    n_timesteps (int): maximum number of timesteps in the environment before reset
                    stochastic (bool): if True the ball will start at different initial positions
                    env_noise (float): amount of gaussian noise for rendering
        """
        # We set the parameters
        self._object_initial_pos = object_initial_pos
        self._object_size = object_size
        self._object_pos = None
        self._n_timesteps = n_timesteps

        # We set the space
        self.action_space = spaces.Discrete(5)  # Up, Down, Left, Right, NoOp
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype='uint8')

        self._env_noise = env_noise
        self._stochastic = stochastic

        # Rendering parameters
        self._width = 500
        self._height = 500
        self._rendering = np.zeros([self._height, self._width, 3])
        self._rendering[0] = 1
        self._rgb = True

        self.viewer = None

        # We set to None to rush error if reset not called
        self._observation = None
        self._steps = None
        self._done = None
        self.reward = None

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        """
        if action == 0:
            self._object_pos[0] += 2. / 50
        elif action == 1:
            self._object_pos[0] -= 2. / 50
        elif action == 2:
            self._object_pos[1] += 2. / 50
        elif action == 3:
            self._object_pos[1] -= 2. / 50

        self._calc_rendering(width=84, height=84)

        self._steps += 1
        if self._steps == self._n_timesteps:
            self._done = True

        info = {}

        return self._rendering, self.reward, self._done, info

    def reset(self, goal=None):
        # We reset the simulation
        self.reward = 0
        if self._stochastic:
            self._object_initial_pos = np.random.uniform(-0.9, 0.9, 2)
        self._object_pos = self._object_initial_pos.copy()

        self._steps = 0
        self._done = False

        self._calc_rendering(width=84, height=84)

        return self._rendering

    def _calc_rendering(self, width, height):
        # We retrieve object pose
        object_pos = self._object_pos

        # World parameters
        world_size = 2.

        # Screen parameters
        screen_center_w = np.ceil(width / 2)
        screen_center_h = np.ceil(height / 2)

        # Ratios
        world2screen = min(width / world_size, height / world_size)

        # Instantiating surface
        surface = gizeh.Surface(width=width, height=height)

        # Drawing Background
        background = gizeh.rectangle(lx=width, ly=height,
                                     xy=(screen_center_w, screen_center_h), fill=(1, 1, 1))
        background.draw(surface)

        # Drawing object
        objt = gizeh.circle(r=self._object_size * world2screen,
                            xy=(screen_center_w + object_pos[0] * world2screen,
                                screen_center_h + object_pos[1] * world2screen),
                            fill=(1, 0, 0))
        objt.draw(surface)

        self._rendering = surface.get_npimage().astype(np.float32)
        if self._env_noise > 0:
            self._rendering = np.random.normal(self._rendering, self._env_noise)
        self._rendering = self._rendering.sum(axis=-1, keepdims=True)
        self._rendering -= self._rendering.min()
        self._rendering /= self._rendering.max()

    def render(self, mode='human', close=False):
        """Renders the environment.

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings
        """

        self._calc_rendering(width=self._width, height=self._height)
        if mode == 'rgb_array':
            return self._rendering.squeeze()
        elif mode is 'human':

            if self.viewer is None:
                self._start_viewer()
                # Retrieve image of corresponding to observation
                img = self._rendering.squeeze()
                self._img_artist = self._ax.imshow(img)
            else:
                # Retrieve image of corresponding to observation
                img = self._rendering.squeeze()
                self._img_artist.set_data(img)
                plt.draw()
                plt.pause(0.05)

    def _start_viewer(self):
        plt.ion()
        self.viewer = plt.figure()
        self._ax = self.viewer.add_subplot(111)

    def close(self):
        if self.viewer is not None:
            plt.close(self.viewer)
            self.viewer = None

    @property
    def dim_goal(self):
        return 2
