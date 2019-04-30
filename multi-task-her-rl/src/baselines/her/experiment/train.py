import os
import sys
import time
os.environ['LD_LIBRARY_PATH']=os.environ['HOME']+'/.mujoco/mjpro150/bin:'
import numpy as np
import json
import argparse
import pickle
from mpi4py import MPI
import datetime
from tqdm import tqdm 

sys.path.append('../../../')
sys.path.append('../../../../')
from src.baselines import logger
from src.baselines.common import set_global_seeds
from src.baselines.her.experiment import config
from src.baselines.her.rollout import RolloutWorker
from src.baselines.her.util import mpi_fork, find_save_path, mpi_average, log
from src.reward_function.reward_function import OracleRewardFuntion
from subprocess import CalledProcessError
import src.baselines.common.tf_util as U


# Me
import gym_myFetchPush
import gym_myGridEnv
from plot_eval_episode import plot_eval_episodes
from src.baselines.her.replay_buffer import ReplayBuffer
from src.baselines.her.ddpg import dims_to_shapes
os.environ["CUDA_VISIBLE_DEVICES"] = ''
#

NUM_CPU = 1
NB_EPOCHS = 100
NB_GOALS = 24


def train(policy, env_worker, model_worker, evaluator, reward_function, model_buffer, n_collect,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
        best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
        periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')
        logger.info("Training...")
    else:
        latest_policy_path = None
        best_policy_path = None
        periodic_policy_path = None
    first_time = last_time = time.time()
    best_success_rate = -1

    my_tqdm = (lambda x: x) if rank >0 else tqdm
    for epoch in range(n_epochs):

        # ~ # train
        # ~ """ Collecting Data for training the model """
        # ~ env_worker.clear_history()

        # ~ for i_c in my_tqdm(range(n_collect)):
            # ~ # interact with the environment
            # ~ episode, goals_reached_ids = env_worker.generate_rollouts()
            # ~ # save experience in memory
            # ~ model_buffer.store_episode(episode, goals_reached_ids)

        # ~ """ Training the model"""
        # ~ samples = model_buffer.sample_transition_for_model(n_collect*(epoch+1))
        # ~ model_worker.envs[0].unwrapped.train(samples, logger.get_dir())
        # ~ """ Training DDPG on the model """
        # ~ model_worker.clear_history()

        for i_c in my_tqdm(range(n_cycles)):
            # interact with the environment
            episode, goals_reached_ids = model_worker.generate_rollouts()
            # save experience in memory
            policy.store_episode(episode, goals_reached_ids)
            # train the reward function and the actor-critic algorithm
            for _ in range(n_batches):
                policy.train()
            policy.update_target_net()

        # test
        evaluator.clear_history()
        episodes = []
        for _ in range(n_test_rollouts):
            episode, goals_reached_ids = evaluator.generate_rollouts()
            episodes.append( (episode, goals_reached_ids ))
        if rank == 0:
            with open(os.path.join(logger.get_dir(), 'eval_episodes.pk'), 'ba') as f:
                pickle.dump(episodes, f)

        best_success_rate, last_time = log(epoch, evaluator, model_worker, policy, best_success_rate, save_policies, best_policy_path, latest_policy_path,
            policy_save_interval, rank, periodic_policy_path, first_time, last_time)
    
    # ~ if rank == 0:
        # ~ plot_eval_episodes(logger.get_dir())
        


def launch(env, trial_id, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return, normalize_obs, nb_goals,
           override_params={}, save_policies=True):
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        # ~ import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        logdir = find_save_path('../../../data/' + env + "/", trial_id)
        logger.configure(dir=logdir)
    else:
        logdir = None

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['date_time'] = str(datetime.datetime.now())
    params['env_name'] = env
    params['normalize_obs'] = normalize_obs
    params['seed'] = seed
    params['logdir'] = logdir
    params['num_cpu'] = num_cpu
    params['replay_strategy'] = replay_strategy
    params['n_test_rollouts'] = nb_goals
    params['nb_goals'] = nb_goals
    if env in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter

    if rank == 0:
        with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
            json.dump(params, f)
    
    params_for_eval = params.copy()
    params_for_model = params.copy()
    params = config.prepare_params(params)
    
    
    if rank == 0:
        config.log_params(params, logger=logger)

    """ for evaluation environment """
    # ~ params_for_eval['env_name'] = 'MultiTaskFetchArmNLP1-v0'
    params_for_eval['env_name'] = 'ArmToolsToys-v1'
    params_for_eval = config.prepare_params(params_for_eval)
    
    """ for model environment """
    # ~ params_for_model['env_name'] = 'myMultiTaskFetchArmNLP-v1'
    # ~ params_for_model['env_name'] = 'MultiTaskFetchArmNLP1-v0'
    params_for_model['env_name'] = 'ArmToolsToys-v1'
    # ~ params_for_model['env_name'] = 'ArmToolsToysModel-v1'

    params_for_model = config.prepare_params(params_for_model)

    reward_function = OracleRewardFuntion(nb_goals)
    dims = config.configure_dims(params)


    policy = config.configure_learning_algo(reward_function=reward_function, normalize_obs=normalize_obs,
                                   dims=dims, params=params, clip_return=clip_return)
    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'eval': False,
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'eval': True
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]
    eval_params['rollout_batch_size'] = 2
    
    """ Replay Buffer for training the model"""
    input_shapes = dims_to_shapes(dims.copy())
    buffer_shapes = {key: (50 if key != 'o' else 51, *input_shapes[key]) for key, val in input_shapes.items()}
    buffer_shapes['g'] = (buffer_shapes['g'][0], dims['g'])
    buffer_size = (params_for_model['_buffer_size'] // params_for_model['rollout_batch_size']) * params_for_model['rollout_batch_size']
    model_buffer = ReplayBuffer(buffer_shapes, buffer_size, 50, None)
    
    """ Rollout workers """
    env_worker = RolloutWorker(params['make_env'], policy, dims, logger,  nb_goals, **rollout_params)
    env_worker.seed(rank_seed)

    model_worker = RolloutWorker(params_for_model['make_env'], policy, dims, logger,  nb_goals, **rollout_params)
    model_worker.seed(rank_seed*100)

    evaluator = RolloutWorker(params_for_eval['make_env'], policy, dims, logger, nb_goals, eval_env=model_worker.envs[0] , **eval_params)
    evaluator.seed(rank_seed * 10)

    train(logdir=logdir, policy=policy, env_worker=env_worker, model_worker=model_worker,  evaluator=evaluator,
          n_epochs=n_epochs, model_buffer = model_buffer, n_collect=params['n_collect'],
          n_test_rollouts=params['n_test_rollouts'], n_cycles=params['n_cycles'], n_batches=params['n_batches'],
          policy_save_interval=policy_save_interval, save_policies=save_policies, reward_function=reward_function)



# ~ env = "myMultiTaskFetchArmNLP-v0"
# ~ env = "MultiTaskFetchArmNLP1-v0"
env = "ArmToolsToys-v1"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', type=str, default=env, help='the name of the OpenAI Gym environment that you want to train on')
    parser.add_argument('--trial_id', type=int, default='0', help='trial identifier, name of the saving folder')
    parser.add_argument('--n_epochs', type=int, default=NB_EPOCHS, help='the number of training epochs to run')
    parser.add_argument('--num_cpu', type=int, default=NUM_CPU, help='the number of CPU cores to use (using MPI)')
    parser.add_argument('--seed', type=int, default=np.random.randint(0, 1e6), help='the random seed used to seed both the environment and the training code')
    parser.add_argument('--policy_save_interval', type=int, default=50, help='the interval with which policy pickles are saved.')
    parser.add_argument('--replay_strategy', type=str, default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
    parser.add_argument('--normalize_obs', type=bool, default=False, help='normalize observations and goals')
    parser.add_argument('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
    parser.add_argument('--nb_goals', type=int, default=NB_GOALS, help='number of instructions')

    kwargs = vars(parser.parse_args())
    launch(**kwargs)
