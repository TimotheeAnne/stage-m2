import matplotlib.pyplot as plt
import numpy as np

class GRBFTrajectory(object):
    def __init__(self, n_dims, sigma, steps_per_basis, max_basis):
        self.n_dims = n_dims
        self.sigma = sigma
        self.alpha = - 1. / (2. * self.sigma ** 2.)
        self.steps_per_basis = steps_per_basis
        self.max_basis = max_basis
        self.precomputed_gaussian = np.zeros(2 * self.max_basis * self.steps_per_basis)
        for i in range(2 * self.max_basis * self.steps_per_basis):
            self.precomputed_gaussian[i] = self.gaussian(self.max_basis * self.steps_per_basis, i)
        
    def gaussian(self, center, t):
        return np.exp(self.alpha * (center - t) ** 2.)
    
    def trajectory(self, weights):
        n_basis = len(weights)//self.n_dims
        weights = np.reshape(weights, (n_basis, self.n_dims)).T
        steps = self.steps_per_basis * n_basis
        traj = np.zeros((steps, self.n_dims))
        for step in range(steps):
            g = self.precomputed_gaussian[self.max_basis * self.steps_per_basis + self.steps_per_basis - 1 - step::self.steps_per_basis][:n_basis]
            traj[step] = np.dot(weights, g)
        return np.clip(traj, -1., 1.)
    
    def plot(self, traj):
        plt.plot(traj)
        plt.ylim([-1.05, 1.05])


def sample_random_trajectories( n_samples, n_dims, time_horizon):
    samples = []
    sigma = 5
    steps_per_basis = 5
    max_basis = time_horizon//5
    trajectory_generator = GRBFTrajectory(n_dims, sigma, steps_per_basis, max_basis)
    for _ in range(n_samples):
        m = 2. * np.random.random(n_dims*max_basis) - 1.
        traj = trajectory_generator.trajectory(m)
        samples.append(traj)
    return samples

