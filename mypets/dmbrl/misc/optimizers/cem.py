from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import scipy.stats as stats
import random 

from .optimizer import Optimizer
from ..GRBF import GRBFTrajectory

class CEMOptimizer(Optimizer):
    """A Tensorflow-compatible CEM optimizer.
    """
    def __init__(self, sol_dim, max_iters, popsize, num_elites, tf_session=None,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25, goal_dim=2):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            tf_session (tf.Session): (optional) Session to be used for this optimizer. Defaults to None,
                in which case any functions passed in cannot be tf.Tensor-valued.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
            goal_dim (int): dimension of the goal space
        """
        super().__init__()
        self.goal_dim = goal_dim
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites
        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha
        self.tf_sess = tf_session

        self.init_goal_pos = tf.zeros([1,self.goal_dim], tf.float32)
        self.init_cost_choice = tf.constant(1)

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        if self.tf_sess is not None:
            with self.tf_sess.graph.as_default():
                with tf.variable_scope("CEMSolver") as scope:
                    self.init_mean = tf.placeholder(dtype=tf.float32, shape=[sol_dim])
                    self.init_var = tf.placeholder(dtype=tf.float32, shape=[sol_dim])

        self.num_opt_iters, self.mean, self.var = None, None, None
        self.tf_compatible, self.cost_function = None, None

    def setup(self, cost_function1,cost_function2,  tf_compatible):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            tf_compatible (bool): True if the cost function provided is tf.Tensor-valued.

        Returns: None
        """
        if tf_compatible and self.tf_sess is None:
            raise RuntimeError("Cannot pass in a tf.Tensor-valued cost function without passing in a TensorFlow "
                               "session into the constructor")

        self.tf_compatible = tf_compatible

        if not tf_compatible:
            self.cost_function = cost_function1
        else:
            def continue_optimization(t, mean, var, best_val, best_sol, goal_pos, cost_choice):
                return tf.cond(
                    tf.equal(cost_choice, 0),
                    lambda: tf.logical_and(tf.less(t, self.max_iters), tf.reduce_max(var) > self.epsilon),
                    lambda: tf.logical_and(tf.less(t, 2), tf.reduce_max(var) > self.epsilon)
                )

            def sample_random_trajectories( n_samples, n_dims, time_horizon):
                samples = []
                sigma = 3
                steps_per_basis = 5
                max_basis = time_horizon//5
                trajectory_generator = GRBFTrajectory(n_dims, sigma, steps_per_basis, max_basis)
                for _ in range(n_samples):
                    m = 2. * np.random.random(n_dims*max_basis) - 1.
                    traj = trajectory_generator.trajectory(m)
                    sample = []
                    for ac in traj:
                        for x in ac:
                            sample.append(x)
                    samples.append(sample)
                return tf.print( tf.convert_to_tensor( samples), ["GBRF"])
                # ~ return tf.convert_to_tensor([[ 2*random.random()-1 for _ in range(7*time_horizon)] for _ in range(n_samples)], dtype=tf.float32)
                
            def iteration(t, mean, var, best_val, best_sol, goal_pos, cost_choice):

                lb_dist, ub_dist =  mean - self.lb, self.ub - mean
                constrained_var = tf.minimum(tf.minimum(tf.square(lb_dist / 2), tf.square(ub_dist / 2)), var)
                
                
                samples = tf.cond(
                    tf.equal(cost_choice, 0),
                    lambda: tf.truncated_normal([self.popsize, self.sol_dim], mean, tf.sqrt(constrained_var)),
                    lambda: tf.truncated_normal([500, self.sol_dim], mean, tf.sqrt(constrained_var))
                )
                
                # ~ samples = tf.cond(
                    # ~ tf.equal(mean[0],0.),
                    # ~ # to replace with a generation coming from a DMP
                    # ~ lambda: sample_random_trajectories(self.popsize,7,self.sol_dim//7),
                    # ~ lambda: tf.truncated_normal([self.popsize, self.sol_dim], mean, tf.sqrt(constrained_var))
                # ~ )

                costs = tf.cond(
                    tf.equal(cost_choice, 0),
                    lambda: cost_function1(samples, goal_pos),
                    lambda: cost_function2(samples, goal_pos)
                )


                values, indices =  tf.nn.top_k(-costs, k=self.num_elites, sorted=True)

                best_val, best_sol = tf.cond(
                    tf.less(-values[0], best_val),
                    lambda: (-values[0], samples[indices[0]]),
                    lambda: (best_val, best_sol)
                )

                elites =  tf.gather(samples, indices)

                new_mean =  tf.reduce_mean(elites, axis=0)
                new_var = tf.reduce_mean(tf.square(elites - new_mean), axis=0)

                mean = self.alpha * mean + (1 - self.alpha) * new_mean
                var = self.alpha * var + (1 - self.alpha) * new_var

                return t + 1, mean, var, best_val, best_sol, goal_pos, cost_choice

            with self.tf_sess.graph.as_default():
                self.num_opt_iters, self.mean, self.var, self.best_val, self.best_sol, _, _ = tf.while_loop(
                    cond=continue_optimization, body=iteration,
                    loop_vars=[0, self.init_mean, self.init_var, float("inf"),
                               self.init_mean, self.init_goal_pos, self.init_cost_choice]
                )

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var, goal_pos, cost_choice):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        if self.tf_compatible:
            sol, solvar, best_val = self.tf_sess.run(
                [self.mean, self.var, self.best_val],
                feed_dict={self.init_mean: init_mean, self.init_var: init_var,
                           self.init_goal_pos: [goal_pos], self.init_cost_choice: cost_choice}
            )
        else:
            mean, var, t = init_mean, init_var, 0
            X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))

            while (t < self.max_iters) and np.max(var) > self.epsilon:
                lb_dist, ub_dist = mean - self.lb, self.ub - mean
                constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

                samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
                costs = self.cost_function(samples)
                elites = samples[np.argsort(costs)][:self.num_elites]

                new_mean = np.mean(elites, axis=0)
                new_var = np.var(elites, axis=0)

                mean = self.alpha * mean + (1 - self.alpha) * new_mean
                var = self.alpha * var + (1 - self.alpha) * new_var

                t += 1
            sol, solvar = mean, var

        return sol, best_val
