import click
import numpy as np
import pickle
import os
os.environ['LD_LIBRARY_PATH'] = os.environ['HOME']+'/.mujoco/mjpro150/bin:'
import json
import sys
sys.path.append('../../../')
sys.path.append('../../../../')
from src.baselines import logger
from src.baselines.common import set_global_seeds
import src.baselines.her.experiment.config as config
from src.baselines.her.rollout import RolloutWorker
from src.reward_function.reward_function import OracleRewardFuntion


PATH = '/home/tim/Documents/stage-m2/multi-task-her-rl/src/data/MultiTaskFetchArmNLP1-v0/81%' + '/'

POLICY_FILE = PATH + 'policy_best.pkl'
PARAMS_FILE = PATH + 'params.json'

@click.command()
@click.argument('policy_file', type=str, default=POLICY_FILE)
@click.option('--seed', type=int, default=int(np.random.randint(1e6)))
@click.option('--n_test_rollouts', type=int, default=12)
@click.option('--render', type=int, default=1)

def main(policy_file, seed, n_test_rollouts, render):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)

    # Load params
    with open(PARAMS_FILE) as json_file:
        params = json.load(json_file)
        
    # # Prepare params.
    
    nb_goals = params['nb_goals']
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_Q': True,
        'rollout_batch_size': 1,
        'render': bool(render),
        'eval': True
    }

    for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]
    
    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, nb_goals, **eval_params)
    evaluator.seed(seed)

    oracle = OracleRewardFuntion(nb_goals)
    # Run evaluation.
    evaluator.clear_history()
    
    all_obs = []
    all_acs = []
    all_out = []
    
    task_ind = list(range(nb_goals))  
    n_per_task = 100
    
    for ind in range(nb_goals):
        print("goal: ", ind)
        Obs = []
        Acs = []
        Out = []
        for _ in range(n_per_task):
            ep = evaluator.generate_rollouts(ind)[0]
            out = oracle.eval_goal_from_episode(ep, ind)
            if out[-1]:
                print('SUCCESS')
            else:
                print('failed')

            obs = ep['o'][0, :, :25]
            Obs.append(obs)
            ac = ep['u'][0,:,:]
            Acs.append(ac)
            Out.append(out[-1])
            # ~ print(obs[:,14:17])
            # ~ print(obs[1:,3:6]-obs[:-1,3:6])
        all_obs.append(Obs)
        all_acs.append(Acs)
        all_out.append(Out)
        

    with open('/home/tim/Documents/stage-m2/mypets/data/multiTask_80%_1200.pk', 'wb') as f:
        pickle.dump(np.array([all_obs, all_acs, all_out]), f)
    

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    main()
