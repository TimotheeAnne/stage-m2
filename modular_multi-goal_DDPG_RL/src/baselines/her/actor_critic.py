import tensorflow as tf
from src.baselines.her.util import store_args, nn


class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers, normalize_obs,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.layers = layers
        self.dimo = dimo
        self.dimg = dimg
        self.dimu = dimu
        self.max_u = max_u
        self.o_stats = o_stats
        self.g_stats = g_stats
        self.hidden = hidden
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        if normalize_obs:
            o = self.o_stats.normalize(self.o_tf)
            g = self.g_stats.normalize(self.g_tf)
        else:
            o = self.o_tf
            g = self.g_tf


        # Networks.
        with tf.variable_scope('pi'):
            input_pi = tf.concat(axis=1, values=[o, g])  # for actor
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)



class ActorCriticTD3:
    def __init__(self, inputs, dimu, max_u,  hidden, layers, normalize_obs, o_stats=None, g_stats=None, **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """

        self.o = inputs['o']
        self.g = inputs['g']
        self.u = inputs['u']
        # Prepare inputs for actor and critic.
        if normalize_obs:
            o = o_stats.normalize(self.o)
            g = g_stats.normalize(self.g)
        else:
            o = self.o = inputs['o']
            g = self.g
        u = self.u

        # Networks.
        with tf.variable_scope('pi'):
            input_pi = tf.concat(axis=1, values=[o, g])  # for actor
            self.pi = max_u * tf.tanh(nn(input_pi, [hidden] * layers + [dimu]))
        with tf.variable_scope('q1'):
            # for critic training
            input_Q1 = tf.concat(axis=1, values=[o, g, u / max_u])
            self.q1 = nn(input_Q1, [hidden] * layers + [1])
        with tf.variable_scope('q2'):
            # for critic training
            input_Q2 = tf.concat(axis=1, values=[o, g, u / max_u])
            self.q2 = nn(input_Q2, [hidden] * layers + [1])
        with tf.variable_scope('q1', reuse=True):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi / max_u])
            self.q_pi = nn(input_Q, [hidden] * layers + [1], reuse=True)


