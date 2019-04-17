import os
import subprocess
import sys
import importlib
import inspect
import functools
import time

import tensorflow as tf
import numpy as np
from mpi4py import MPI

from src.baselines.common import tf_util as U
from src.baselines.common.mpi_moments import mpi_moments
from src.baselines import logger


def log(epoch, evaluator, rollout_worker, policy, best_success_rate, save_policies, best_policy_path, latest_policy_path,
        policy_save_interval, rank, periodic_policy_path, first_time, last_time):

    new_time = time.time()
    # record logs
    logger.record_tabular('epoch', epoch)
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, mpi_average(val))
    for key, val in rollout_worker.logs('train'):
        print('_________________________________________________________', key,val)
        logger.record_tabular(key, mpi_average(val))
    for key, val in policy.logs():
        logger.record_tabular(key, mpi_average(val))
    logger.record_tabular('pos_rew_ratio', mpi_average(policy.get_positive_reward_stat()))
    logger.record_tabular('total_duration (s)', new_time - first_time)
    logger.record_tabular('epoch_duration (s)', new_time - last_time)
    

    if rank == 0:
        logger.dump_tabular()

    # save the policy if it's better than the previous ones
    success_rate = mpi_average(evaluator.current_success_rate())
    if rank == 0 and success_rate >= best_success_rate and save_policies:
        best_success_rate = success_rate
        logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
        evaluator.save_policy(best_policy_path)
        evaluator.save_policy(latest_policy_path)
    if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
        policy_path = periodic_policy_path.format(epoch)
        logger.info('Saving periodic policy to {} ...'.format(policy_path))
        evaluator.save_policy(policy_path)

    # make sure that different threads have different seeds
    local_uniform = np.random.uniform(size=(1,))
    root_uniform = local_uniform.copy()
    MPI.COMM_WORLD.Bcast(root_uniform, root=0)
    if rank != 0:
        assert local_uniform[0] != root_uniform[0]

    return best_success_rate, new_time


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper

def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]

def find_save_path(dir, trial_id):
    """
    Create a directory to save results and arguments. Adds 100 to the trial id if a directory already exists.

    Params
    ------
    - dir (str)
        Main saving directory
    - trial_id (int)
        Trial identifier
    """
    i=0
    while True:
        save_dir = dir+str(trial_id+i*100)+'/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            break
        i+=1
    return save_dir

def import_function(spec):
    """Import a function identified by a string like "pkg.module:fn_name".
    """
    mod_name, fn_name = spec.split(':')
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name)
    return fn


def flatten_grads(var_list, grads):
    """Flattens a variables and their gradients.
    """
    return tf.concat([tf.reshape(grad, [U.numel(v)])
                      for (v, grad) in zip(var_list, grads)], 0)


def nn(input, layers_sizes, reuse=None, flatten=False, name=""):
    """Creates a simple neural network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                reuse=reuse,
                                name=name + '_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input


def install_mpi_excepthook():
    import sys
    from mpi4py import MPI
    old_hook = sys.excepthook

    def new_hook(a, b, c):
        old_hook(a, b, c)
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort()
    sys.excepthook = new_hook


def mpi_fork(n, extra_mpi_args=[]):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    """
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        # "-bind-to core" is crucial for good performance
        args = ["mpirun", "-np", str(n)] + \
            extra_mpi_args + \
            [sys.executable]

        args += sys.argv
        subprocess.check_call(args, env=env)
        return "parent"
    else:
        install_mpi_excepthook()
        return "child"


def convert_episode_to_batch_major(episode):
    """Converts an episode to have the batch dimension in the major (first)
    dimension.
    """
    episode_batch = {}
    for key in episode.keys():
        val = np.array(episode[key]).copy()
        # make inputs batch-major instead of time-major
        episode_batch[key] = val.swapaxes(0, 1)

    return episode_batch


def transitions_in_episode_batch(episode_batch):
    """Number of transitions in a given episode batch.
    """
    shape = episode_batch['u'].shape
    return shape[0] * shape[1]


def reshape_for_broadcasting(source, target):
    """Reshapes a tensor (source) to have the correct shape and dtype of the target
    before broadcasting it with MPI.
    """
    dim = len(target.get_shape())
    shape = ([1] * (dim - 1)) + [-1]
    return tf.reshape(tf.cast(source, target.dtype), shape)
