import numpy as np
from mpi4py import MPI
from baselines import logger

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, nb_goals):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        goal_sampler (object): contains the list of discovered goals
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0


    def _sample_her_transitions(episode_batch, goal_ids, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # select which goals to learn from
        valid_buffers = []
        for i in range(len(goal_ids)):
            if len(goal_ids[i]) > 10:
                valid_buffers.append(i)

        if len(valid_buffers) > 0:
            buffer_ids = np.random.choice(valid_buffers, size=batch_size)
            episode_idxs = []
            for i in buffer_ids:
                episode_idxs.append(np.random.choice(goal_ids[i]))
        else:
            # Select which episodes and time steps to use.
            episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}


        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.argwhere(np.random.uniform(size=batch_size) < future_p).squeeze()
        # future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        # future_offset = future_offset.astype(int)
        # future_t = (t_samples + 1 + future_offset)[her_indexes]


        # if some goals have been discovered, shuffle the list,
        # and try each after the other until one gives a reward.
        # replay using that goal.
        # if none gives a reward, replay using a random goal


        proba = np.ones([nb_goals]) / nb_goals

        for ind in her_indexes:
            obs = transitions['o'][ind, :]
            # implement a bias in replay, replay proportionally to 1/nb_positive_feedbacks
            replay_goals_ids = np.arange(nb_goals)
            np.random.shuffle(replay_goals_ids)
            # np.random.shuffle(discovered_goals_ids)
            replay = False
            for goal_id in replay_goals_ids:
                if reward_fun(state=obs, goal=np.array([goal_id]), info={})[0] == 0:
                    # replay, update the goal and goal_id
                    goal = np.zeros([nb_goals])
                    goal[goal_id] = 1
                    transitions['g'][ind, :] = goal
                    replay = True
                    break
            if not replay:
                goal_id = np.random.randint(nb_goals)
                goal = np.zeros([nb_goals])
                goal[goal_id] = 1
                transitions['g'][ind, :] = goal

        # # Replace goal with achieved goal but only for the previously-selected
        # # HER transitions (as defined by her_indexes). For the other transitions,
        # # keep the original goal.
        # new_goals = transitions['g'][her_indexes]
        # transitions['g'][her_indexes] = new_goals


        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        goals = transitions['g']
        rew_goal_ids = np.zeros([goals.shape[0], 1])
        for i in range(goals.shape[0]):
            rew_goal_ids[i, 0] = int(np.argwhere(goals[i] == 1))
        reward_params = dict(state=transitions['o'],
                             goal=rew_goal_ids)
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        ratio_positive_rewards = (transitions['r']==0).mean()

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions, ratio_positive_rewards, proba

    return _sample_her_transitions
