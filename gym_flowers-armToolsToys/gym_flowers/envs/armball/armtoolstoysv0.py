from __future__ import division
import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt


class ArmToolsToysV0(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, 
                 n_timesteps=50,
                 epsilon=0.1,
                 distractor_noise = 0.01
                 ):
        
        self.epsilon = epsilon
        self.n_timesteps = n_timesteps
        self.n_act = 4
        self.n_obs = 31*2
        
        # GripArm
        self.arm_lengths = [0.5, 0.3, 0.2]
        self.arm_angle_shift = 0.5
        self.arm_rest_state = [0., 0., 0., 0.]

        # Stick1
        self.stick1_length = 0.5
        self.stick1_type = "magnetic"
        self.stick1_handle_tol = 0.03
        self.stick1_handle_tol_sq = self.stick1_handle_tol ** 2.
        self.stick1_rest_state = [-0.75, 0.25, 0.75]
        
        # Stick2
        self.stick2_length = 0.5
        self.stick2_type = "scratch"
        self.stick2_handle_tol = 0.03
        self.stick2_handle_tol_sq = self.stick1_handle_tol ** 2.
        self.stick2_rest_state = [0.75, 0.25, 0.25]
        
        # Magnet1
        self.magnet1_tolsq = 0.03 ** 2.
        self.magnet1_rest_state = [-0.3, 1.1]
        # Magnet2
        self.magnet2_tolsq = 0.
        self.magnet2_rest_state = [-0.5, 1.5]
        # Magnet3
        self.magnet3_tolsq = 0.
        self.magnet3_rest_state = [-0.3, 1.5]
        
        # Scratch1
        self.scratch1_tolsq = 0.03 ** 2.
        self.scratch1_rest_state = [0.3, 1.1]
        # Scratch2
        self.scratch2_tolsq = 0.
        self.scratch2_rest_state = [0.3, 1.5]
        # Scratch3
        self.scratch3_tolsq = 0.
        self.scratch3_rest_state = [0.5, 1.5]
        
        # Cat
        self.cat_noise = distractor_noise
        self.cat_rest_state = [-0.1, 1.1]
        # Dog
        self.dog_noise = distractor_noise
        self.dog_rest_state = [0.1, 1.1]

        # Static objects
        self.static_objects_rest_state = [[-0.7, 1.1],
                                          [-0.5, 1.1],
                                          [0.5, 1.1],
                                          [0.7, 1.1]]
        
        
        # We define the spaces
        self.action_space = spaces.Box(low=-np.ones(self.n_act),
                                       high=np.ones(self.n_act),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.ones(self.n_obs) * 2.,
                                       high=np.ones(self.n_obs) * 2.,
                                       dtype='float32')
                                       


        self.viewer = None
        self.background = None

        self.reset()

        # We set to None to rush error if reset not called
        self.reward = None
        self.observation = None
        self.initial_observation = None
        self.done = None

        self.info = dict(is_success=0)


    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def _sample_goal(self):
        return np.random.uniform(-2.,2.,2)

    def compute_reward(self, achieved_goal, goal, info=None):
        if achieved_goal.ndim > 1:
            d = np.zeros([goal.shape[0]])
            for i in range(goal.shape[0]):
                d[i] = np.linalg.norm(achieved_goal[i, :] - goal[i, :], ord=2)
        else:
            d = np.linalg.norm(achieved_goal - goal, ord=2)
        return -(d > self.epsilon).astype(np.int)

    def reset(self):
        self.arm_angles = self.arm_rest_state[:-1]
        a = self.arm_angle_shift + np.cumsum(np.array(self.arm_angles))
        a_pi = np.pi * a
        self.hand_pos = np.array([np.sum(np.cos(a_pi)*self.arm_lengths),
                                  np.sum(np.sin(a_pi)*self.arm_lengths)])
        
        self.gripper = self.arm_rest_state[3]
        
        self.stick1_held = False
        self.stick1_handle_pos = np.array(self.stick1_rest_state[0:2])
        self.stick1_angle = self.stick1_rest_state[2]
        
        self.stick2_held = False
        self.stick2_handle_pos = np.array(self.stick2_rest_state[0:2])
        self.stick2_angle = self.stick2_rest_state[2]
        
        a = np.pi * self.stick1_angle
        self.stick1_end_pos = [
            self.stick1_handle_pos[0] + np.cos(a) * self.stick1_length, 
            self.stick1_handle_pos[1] + np.sin(a) * self.stick1_length]

        a = np.pi * self.stick2_angle
        self.stick2_end_pos = [
            self.stick2_handle_pos[0] + np.cos(a) * self.stick2_length, 
            self.stick2_handle_pos[1] + np.sin(a) * self.stick2_length]
        
        self.magnet1_move = 0
        self.magnet1_pos = self.magnet1_rest_state
        self.magnet2_move = 0
        self.magnet2_pos = self.magnet2_rest_state
        self.magnet3_move = 0
        self.magnet3_pos = self.magnet3_rest_state
        
        self.scratch1_move = 0
        self.scratch1_pos = self.scratch1_rest_state
        self.scratch2_move = 0
        self.scratch2_pos = self.scratch2_rest_state
        self.scratch3_move = 0
        self.scratch3_pos = self.scratch3_rest_state
        
        self.static_objects_pos = list(self.static_objects_rest_state)
        
        self.cat_pos = np.array(self.cat_rest_state)
        self.dog_pos = np.array(self.dog_rest_state)
        
        # construct vector of observations
        self.observation = np.zeros(self.n_obs)
        self.observation[:31] = self.observe()
        self.initial_observation = np.array(self.observe())
        self.steps = 0
        self.done = False
        return self.observation
        
    def observe(self):
        return [
                self.hand_pos[0],
                self.hand_pos[1],
                self.gripper,
                self.stick1_end_pos[0],
                self.stick1_end_pos[1],
                self.stick2_end_pos[0],
                self.stick2_end_pos[1],
                self.magnet1_pos[0],
                self.magnet1_pos[1],
                self.magnet2_pos[0],
                self.magnet2_pos[1],
                self.magnet3_pos[0],
                self.magnet3_pos[1],
                self.scratch1_pos[0],
                self.scratch1_pos[1],
                self.scratch2_pos[0],
                self.scratch2_pos[1],
                self.scratch3_pos[0],
                self.scratch3_pos[1],
                self.cat_pos[0],
                self.cat_pos[1],
                self.dog_pos[0],
                self.dog_pos[1],
                self.static_objects_rest_state[0][0],
                self.static_objects_rest_state[0][1],
                self.static_objects_rest_state[1][0],
                self.static_objects_rest_state[1][1],
                self.static_objects_rest_state[2][0],
                self.static_objects_rest_state[2][1],
                self.static_objects_rest_state[3][0],
                self.static_objects_rest_state[3][1]
                ]
        
    def step(self, action):
        """Run one timestep of the environment's dynamics.
        """
        action = np.array(action).clip(-1, 1)

        # GripArm
        self.arm_angles = self.arm_angles + action[:-1] / 10.
        # We optimize runtime
        a = [self.arm_angle_shift + self.arm_angles[0]] * 3
        a[1] += self.arm_angles[1]
        a[2] = a[1] + self.arm_angles[2]
        a_pi = [np.pi * a[0],
                np.pi * a[1],
                np.pi * a[2]]
        self.hand_pos[0] = np.cos(a_pi[0])*self.arm_lengths[0]
        self.hand_pos[0] += np.cos(a_pi[1])*self.arm_lengths[1]
        self.hand_pos[0] += np.cos(a_pi[2])*self.arm_lengths[2]
        self.hand_pos[1] = np.sin(a_pi[0])*self.arm_lengths[0]
        self.hand_pos[1] += np.sin(a_pi[1])*self.arm_lengths[1]
        self.hand_pos[1] += np.sin(a_pi[2])*self.arm_lengths[2]
        if action[-1] >= 0.:
            new_gripper = 1. 
        else:
            new_gripper = -1.
        gripper_change = (self.gripper - new_gripper) / 2.
        self.gripper = new_gripper
        hand_angle = np.mod(a[-1] + 1, 2) - 1
        
        # Stick1
        if not self.stick1_held:
            if gripper_change == 1. and ((self.hand_pos[0]
                                         - self.stick1_handle_pos[0]) ** 2.
                                         + (self.hand_pos[1]
                                            - self.stick1_handle_pos[1]) ** 2.
                                         < self.stick1_handle_tol_sq):
                self.stick1_handle_pos = list(self.hand_pos)
                self.stick1_angle = hand_angle
                a = np.pi * self.stick1_angle
                self.stick1_end_pos = [
                    self.stick1_handle_pos[0] + np.cos(a) * self.stick1_length, 
                    self.stick1_handle_pos[1] + np.sin(a) * self.stick1_length]
                self.stick1_held = True
        else:
            if gripper_change == 0:
                self.stick1_handle_pos = list(self.hand_pos)
                self.stick1_angle = hand_angle
                a = np.pi * self.stick1_angle
                self.stick1_end_pos = [
                    self.stick1_handle_pos[0] + np.cos(a) * self.stick1_length, 
                    self.stick1_handle_pos[1] + np.sin(a) * self.stick1_length]
            else:
                self.stick1_held = False
                
        # Stick2
        if not self.stick2_held:
            if gripper_change == 1. and ((self.hand_pos[0]
                                          - self.stick2_handle_pos[0]) ** 2.
                                         + (self.hand_pos[1]
                                            - self.stick2_handle_pos[1]) ** 2.
                                         < self.stick2_handle_tol_sq):
                self.stick2_handle_pos = list(self.hand_pos)
                self.stick2_angle = hand_angle
                a = np.pi * self.stick2_angle
                self.stick2_end_pos = [
                    self.stick2_handle_pos[0] + np.cos(a) * self.stick2_length, 
                    self.stick2_handle_pos[1] + np.sin(a) * self.stick2_length]
                self.stick2_held = True
        else:
            if gripper_change == 0:
                self.stick2_handle_pos = list(self.hand_pos)
                self.stick2_angle = hand_angle
                a = np.pi * self.stick2_angle
                self.stick2_end_pos = [
                    self.stick2_handle_pos[0] + np.cos(a) * self.stick2_length, 
                    self.stick2_handle_pos[1] + np.sin(a) * self.stick2_length]
            else:
                self.stick2_held = False

        # Magnet1
        if self.magnet1_move == 1 or ((self.stick1_end_pos[0]
                                      - self.magnet1_pos[0]) ** 2
                                      + (self.stick1_end_pos[1]
                                         - self.magnet1_pos[1]) ** 2
                                         < self.magnet1_tolsq):
            self.magnet1_pos = self.stick1_end_pos[0:2]
            self.magnet1_move = 1
        # Scratch1
        if self.scratch1_move == 1 or ((self.stick2_end_pos[0]
                                        - self.scratch1_pos[0]) ** 2
                                       + (self.stick2_end_pos[1]
                                          - self.scratch1_pos[1]) ** 2
                                       < self.scratch1_tolsq):
            self.scratch1_pos = self.stick2_end_pos[0:2]
            self.scratch1_move = 1

        # Cat
        rdm = np.random.randn(4)
        self.cat_pos = self.cat_pos + rdm[:2] * self.cat_noise
        # Dog
        self.dog_pos = self.dog_pos + rdm[2:] * self.dog_noise
        
        
        self.observation[:31] = self.observe()
        self.observation[31:] = self.observation[:31]-self.initial_observation
        
        self.steps += 1
        if self.steps == self.n_timesteps:
            self.done = True

        # ~ return self.obs_out, float(self.reward), self.done, self.info
        return self.observation, 0, False, {}

    def render(self, mode='human', close=False):

        if self.viewer is None:
            self.start_viewer()
        
        fig = plt.gcf()
        ax = plt.gca()
        
        fig.canvas.restore_region(self.background)
        
        # Arm
        angles = np.array(self.arm_angles)
        angles[0] += self.arm_angle_shift
        a = np.cumsum(np.pi * angles)
        x = np.hstack((0., np.cumsum(np.cos(a)*self.arm_lengths)))
        y = np.hstack((0., np.cumsum(np.sin(a)*self.arm_lengths)))
        self.lines["l1"][0].set_data(x, y)
        self.lines["l2"][0].set_data(x[0], y[0])
        for i in range(len(self.arm_lengths)-1):
            self.lines[i][0].set_data(x[i+1], y[i+1])
        self.lines["l3"][0].set_data(x[-1], y[-1])

        
        # Gripper
        if self.gripper >= 0.:
            self.lines["g1"][0].set_data(x[-1], y[-1])
            self.lines["g2"][0].set_data(3, 3)
        else:
            self.lines["g1"][0].set_data(3, 3)
            self.lines["g2"][0].set_data(x[-1], y[-1])
            
        # Stick1
        if self.stick1_held or self.steps <= 1:
            self.lines["s11"][0].set_data([self.stick1_handle_pos[0], 
                                           self.stick1_end_pos[0]],
                                          [self.stick1_handle_pos[1], 
                                           self.stick1_end_pos[1]])
            self.lines["s12"][0].set_data(self.stick1_handle_pos[0], 
                                          self.stick1_handle_pos[1])
            self.lines["s13"][0].set_data(self.stick1_end_pos[0],
                                          self.stick1_end_pos[1])
        
        # Magnet1
        self.patches['mag1'].set_xy((self.magnet1_pos[0] - 0.05, 
                                     self.magnet1_pos[1] - 0.05))
            
        # Stick2
        if self.stick2_held or self.steps <= 1:
            self.lines["s21"][0].set_data([self.stick2_handle_pos[0], 
                                           self.stick2_end_pos[0]],
                                          [self.stick2_handle_pos[1], 
                                           self.stick2_end_pos[1]])
            self.lines["s22"][0].set_data(self.stick2_handle_pos[0], 
                                          self.stick2_handle_pos[1])
            self.lines["s23"][0].set_data(self.stick2_end_pos[0],
                                          self.stick2_end_pos[1])
            
        # Scratch1
        self.patches['scr1'].set_xy((self.scratch1_pos[0] - 0.05, 
                                     self.scratch1_pos[1] - 0.05))
    
        

        # Cat
        self.patches['cat'].set_xy((self.cat_pos[0] - 0.05, 
                                    self.cat_pos[1] - 0.05))
        # Dog
        self.patches['dog'].set_xy((self.dog_pos[0] - 0.05, 
                                    self.dog_pos[1] - 0.05))
            
        if mode == 'rgb_array':
            return self.rendering  # return RGB frame suitable for video
        elif mode is 'human':
            fig.canvas.blit(ax.bbox)
            plt.pause(0.001)

    def start_viewer(self):
        self.viewer = plt.figure(figsize=(5, 5), frameon=False)
        fig = plt.gcf()
        ax = plt.gca()
        plt.axis('off')
        fig.show()
        fig.canvas.draw()
        self.lines = {}
        self.patches = {}
        
        # Arm
        angles = np.array(self.arm_angles)
        angles[0] += self.arm_angle_shift
        a = np.cumsum(np.pi * angles)
        x = np.hstack((0., np.cumsum(np.cos(a)*self.arm_lengths)))
        y = np.hstack((0., np.cumsum(np.sin(a)*self.arm_lengths)))
        self.lines["l1"] = ax.plot(x, y, 'grey', lw=4)
        self.lines["l2"] = ax.plot(x[0], y[0], 'ok', ms=8)
        for i in range(len(self.arm_lengths)-1):
            self.lines[i] = ax.plot(x[i+1], y[i+1], 'ok', ms=8)
        self.lines["l3"] = ax.plot(x[-1], y[-1], 'or', ms=4)
        ax.axis([-2, 2., -1.5, 2.])        

        # Gripper
        self.lines["g1"] = ax.plot(3, 3, 'o', markerfacecolor='none', 
                                   markeredgewidth=3, markeredgecolor="r", 
                                   ms=20)
        self.lines["g2"] = ax.plot(3, 3, 'o', color="r", ms=10)
            
        # Stick1
        self.lines["s11"] = ax.plot([self.stick1_handle_pos[0],
                                     self.stick1_end_pos[0]],
                                    [self.stick1_handle_pos[1],
                                     self.stick1_end_pos[1]], '-',
                                    color='grey', lw=4)
        self.lines["s12"] = ax.plot(self.stick1_handle_pos[0],
                                    self.stick1_handle_pos[1], 'o',
                                    color = "g", ms=6)
        self.lines["s13"] = ax.plot(self.stick1_end_pos[0],
                                    self.stick1_end_pos[1], 'o',
                                    color = "b", ms=6)
                    
        # Stick2
        self.lines["s21"] = ax.plot([self.stick2_handle_pos[0],
                                     self.stick2_end_pos[0]],
                                    [self.stick2_handle_pos[1],
                                     self.stick2_end_pos[1]], '-',
                                    color='grey', lw=4)
        self.lines["s22"] = ax.plot(self.stick2_handle_pos[0],
                                    self.stick2_handle_pos[1], 'o',
                                    color = "g", ms=6)
        self.lines["s23"] = ax.plot(self.stick2_end_pos[0],
                                    self.stick2_end_pos[1], 'o',
                                    color = "c", ms=6)
                
        # Magnet1
        p = plt.Rectangle((self.magnet1_pos[0] - 0.05,
                           self.magnet1_pos[1] - 0.05),
                          0.1, 0.1, fc='b')
        self.patches['mag1'] = p
        ax.add_patch(p)
        # Magnet2
        ax.add_patch(plt.Rectangle((self.magnet2_pos[0] - 0.05,
                                    self.magnet2_pos[1] - 0.05),
                                   0.1, 0.1, fc='b'))
        # Magnet3
        ax.add_patch(plt.Rectangle((self.magnet3_pos[0] - 0.05,
                                    self.magnet3_pos[1] - 0.05),
                                   0.1, 0.1, fc='b'))
        
        # Scratch1
        p = plt.Rectangle((self.scratch1_pos[0] - 0.05,
                           self.scratch1_pos[1] - 0.05),
                          0.1, 0.1, fc="c")
        ax.add_patch(p)
        self.patches['scr1'] = p
        # Scratch2
        ax.add_patch(plt.Rectangle((self.scratch2_pos[0] - 0.05,
                                    self.scratch2_pos[1] - 0.05),
                                   0.1, 0.1, fc="c"))
        # Scratch3
        ax.add_patch(plt.Rectangle((self.scratch3_pos[0] - 0.05,
                                    self.scratch3_pos[1] - 0.05),
                                   0.1, 0.1, fc="c"))

        # Cat
        p = plt.Rectangle((self.cat_pos[0] - 0.05,
                           self.cat_pos[1] - 0.05),
                          0.1, 0.1, fc="m")
        ax.add_patch(p)
        self.patches['cat'] = p
        # Dog
        p = plt.Rectangle((self.dog_pos[0] - 0.05,
                           self.dog_pos[1] - 0.05),
                          0.1, 0.1, fc="y")
        ax.add_patch(p)
        self.patches['dog'] = p
        
        # Static
        for pos in self.static_objects_pos:
            ax.add_patch(plt.Rectangle((pos[0] - 0.05,
                                        pos[1] - 0.05),
                                       0.1, 0.1, fc='k'))
        
        self.background = fig.canvas.copy_from_bbox(ax.bbox)

    def close(self):
        if self.viewer is not None:
            plt.close(self.viewer)
            self.viewer = None

    @property
    def dim_goal(self):
        return 2



