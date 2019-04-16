import threading
from collections import deque
import numpy as np



class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        self.goals_indices = []
        self.discovered_goals_ids = []

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.pointer = 0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]

        transitions, ratio_positive_rewards, replay_proba = self.sample_transitions(buffers, self.goals_indices, batch_size)

        for key in (['r', 'o_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions, ratio_positive_rewards, replay_proba

    def sample_transition_for_model(self, batch_size):
        ind = np.random.randint(self.current_size, size=batch_size)
        transitions = dict()
        for key in ['o', 'u']:
            transitions[key] = self.buffers[key][ind] 
        transitions['o_2'] = self.buffers['o'][ind, 1:, :]
        
        return transitions
    
    def sample_transition_for_normalization(self, batch_size):
        
        ind = np.random.randint(self.n_transitions_stored, size=batch_size)
        
        transitions = dict()
        
        for key in ['o', 'g']:
            transitions[key] = self.buffers[key][ind]
        transitions['o_2'] = self.buffers['o'][ind, 1:, :]
        
        return transitions
        
    def store_episode(self, episode_batch, goals_reached_ids):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            for i in range(len(idxs)):
                for reached_id in goals_reached_ids[i]:
                    if reached_id not in self.discovered_goals_ids:
                        self.discovered_goals_ids.append(reached_id)
                        self.goals_indices.append(deque())
                # remove old indices
                if self.current_size == self.size:
                    for goal_buffer_ids in self.goals_indices:
                        if len(goal_buffer_ids) > 0:
                            if idxs[i] == goal_buffer_ids[0]:
                                goal_buffer_ids.popleft()
                # append new goal indices
                for reached_id in goals_reached_ids[i]:
                    ind_list = self.discovered_goals_ids.index(reached_id)
                    self.goals_indices[ind_list].append(idxs[i])

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1  # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.pointer + inc <= self.size:
            idx = np.arange(self.pointer, self.pointer + inc)
            self.pointer = self.pointer + inc
        else:
            overflow = inc - (self.size - self.pointer)
            idx_a = np.arange(self.pointer, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.pointer = overflow

        # update replay size
        self.current_size = min(self.size, self.current_size + inc)

        # ~ if inc == 1:
            # ~ idx = idx[0]
        return idx

    # def _get_storage_idx(self, inc=None):
    #     inc = inc or 1   # size increment
    #     assert inc <= self.size, "Batch committed to replay is too large!"
    #     # go consecutively until you hit the end, and then go randomly.
    #     if self.current_size+inc <= self.size:
    #         idx = np.arange(self.current_size, self.current_size+inc)
    #     elif self.current_size < self.size:
    #         overflow = inc - (self.size - self.current_size)
    #         idx_a = np.arange(self.current_size, self.size)
    #         idx_b = np.random.randint(0, self.current_size, overflow)
    #         idx = np.concatenate([idx_a, idx_b])
    #     else:
    #         idx = np.random.randint(0, self.size, inc)
    #
    #     # update replay size
    #     self.current_size = min(self.size, self.current_size+inc)
    #
    #     if inc == 1:
    #         idx = idx[0]
    #     return idx
