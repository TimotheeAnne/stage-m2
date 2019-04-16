import numpy as np
from mpi4py import MPI
from src.reward_function.reward_function import instructions
from collections import deque

class GoalSampler:
    def __init__(self, language_model, goal_dim, nb_instr):
        self.goal_dim = goal_dim
        self.nb_feedbacks = []
        self.nb_positive_feedbacks = []
        self.nb_negative_feedbacks = []
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.nb_cpu = MPI.COMM_WORLD.Get_size()
        self.goal_discovered_encodings = []
        self.goal_discovered_str = []
        self.nb_goals_discovered = 0
        self.discovered_goal_oracle_ids = [] # true id of discovered goals
        self.feedback_goal_to_goal = []
        self.window_feedback_goal_to_goal = 100
        self.target_goal_counters = []

        self.eps_goal_selection = 0.2
        self.oracle_goal_counters = [0] * nb_instr
        self.oracle_goal_encodings = []
        for goal_str in instructions[:nb_instr]:
            self.oracle_goal_encodings.append(language_model.encode(goal_str).squeeze())


    def update(self,id_goals_attempted, goals_reached, goals_not_reached, goals_reached_str, goals_not_reached_str):

        id_goals_attempted = MPI.COMM_WORLD.gather(id_goals_attempted.copy(), root=0)
        goals_reached = MPI.COMM_WORLD.gather(goals_reached.copy(), root=0)
        goals_reached_str  = MPI.COMM_WORLD.gather(goals_reached_str.copy(), root=0)
        goals_not_reached = MPI.COMM_WORLD.gather(goals_not_reached.copy(), root=0)
        goals_not_reached_str = MPI.COMM_WORLD.gather(goals_not_reached_str.copy(), root=0)
        goals_reached_ids = []
        if self.rank == 0:
            assert len(goals_reached) == len(goals_not_reached_str) == len(goals_not_reached) == len(goals_reached_str)
            rollout_batch_size = len(goals_reached[0])
            for k in range(self.nb_cpu):
                goals_reached_ids.append([])
                for i in range(rollout_batch_size):
                    goals_reached_ids[-1].append([])
                    if id_goals_attempted[k][i] != -1:
                        ind_attempted = self.discovered_goal_oracle_ids.index(id_goals_attempted[k][i])
                        self.target_goal_counters[ind_attempted] += 1
                    else:
                        ind_attempted = -1
                    for j in range(len(goals_reached[k][i])):
                        # track discovered goals (oracle tracking for evaluation) and get univ id for current goal
                        ind = self.arreq_in_list(goals_reached[k][i][j], self.oracle_goal_encodings)[1]
                        self.oracle_goal_counters[ind] += 1
                        goals_reached_ids[-1][-1].append(ind)

                        # new goal discovered !
                        if not ind in self.discovered_goal_oracle_ids:
                            self.discovered_goal_oracle_ids.append(ind) # track the true id of the goal
                            
                            # start tracking feedback for a new goal
                            self.feedback_goal_to_goal.append([])
                            for el in range(self.nb_goals_discovered):
                                self.feedback_goal_to_goal[-1].append(deque([0], maxlen=self.window_feedback_goal_to_goal))
                            for el in self.feedback_goal_to_goal:
                                el.append(deque([0], maxlen=self.window_feedback_goal_to_goal))
                            if ind_attempted != -1:
                                self.feedback_goal_to_goal[ind_attempted][-1].append(1)
                            self.target_goal_counters.append(1)
                            # track discovered goal (agent's tracking)
                            self.goal_discovered_encodings.append(goals_reached[k][i][j])
                            self.goal_discovered_str.append(goals_reached_str[k][i][j])
                            self.nb_goals_discovered += 1
                            self.nb_feedbacks.append(1)
                            self.nb_positive_feedbacks.append(1)
                            self.nb_negative_feedbacks.append(0)
                        # old goal
                        else:
                            # ind_1 = self.arreq_in_list(goals_reached[k][i][j], self.goal_discovered_encodings)[1]
                            ind_reached = self.discovered_goal_oracle_ids.index(ind)
                            self.nb_feedbacks[ind_reached] += 1
                            self.nb_positive_feedbacks[ind_reached] += 1

                            # save feedbacks
                            if ind_attempted != -1:
                                self.feedback_goal_to_goal[ind_attempted][ind_reached].append(1)

                    for j in range(len(goals_not_reached[k][i])):
                        ind = self.arreq_in_list(goals_not_reached[k][i][j], self.oracle_goal_encodings)[1]
                        # this should not happen, cause new goals do not receive negative feedbacks
                        # if not self.arreq_in_list(goals_not_reached[k][i][j], self.goal_discovered_encodings)[0]:
                        if not ind in self.discovered_goal_oracle_ids:
                            print('ERRORORORORORRO')
                            self.goal_discovered_encodings.append(goals_not_reached[k][i][j])
                            self.goal_discovered_str.append(goals_not_reached_str[k][i][j])
                            self.nb_feedbacks.append(0)
                            self.nb_goals_discovered += 1
                            self.nb_negative_feedbacks.append(1)
                            self.nb_positive_feedbacks.append(0)
                        # old goal
                        else:
                            # ind_1 = self.arreq_in_list(goals_not_reached[k][i][j], self.goal_discovered_encodings)[1]
                            ind_not_reached = self.discovered_goal_oracle_ids.index(ind)
                            self.nb_feedbacks[ind_not_reached] += 1
                            self.nb_negative_feedbacks[ind_not_reached] += 1

                            # save feedbacks
                            if ind_attempted != -1:
                                self.feedback_goal_to_goal[ind_attempted][ind_not_reached].append(0)
            self.compute_scores()
        goals_reached_ids = MPI.COMM_WORLD.scatter(goals_reached_ids, root=0)
        return goals_reached_ids

    def compute_scores(self):
        self.perceived_learning_progress = np.zeros([self.nb_goals_discovered])
        self.perceived_competence = np.zeros([self.nb_goals_discovered])
        self.feedback_stats = np.zeros([self.nb_goals_discovered, self.nb_goals_discovered])
        for i in range(self.nb_goals_discovered):
            for j in range(self.nb_goals_discovered):
                self.feedback_stats[i, j] = np.mean(self.feedback_goal_to_goal[i][j])
                # compute perceived competence and competence progress
                if i == j:
                    self.perceived_competence[i] = self.feedback_stats[i, j]
                    len_history = len(self.feedback_goal_to_goal[i][j])
                    if len_history > 20:
                        history = np.array(self.feedback_goal_to_goal[i][j])
                        self.perceived_learning_progress[i] = np.abs(history[:len_history//2].mean() - history[len_history//2:].mean())
        self.score_target_goals = np.zeros([self.nb_goals_discovered])
        for i in range(self.nb_goals_discovered):
            for j in range(self.nb_goals_discovered):
                oracle_ind = self.discovered_goal_oracle_ids[j]
                self.score_target_goals[i] += 1 / (self.oracle_goal_counters[oracle_ind] + 1) * self.feedback_stats[i, j]
        if self.score_target_goals.sum() == 0:
            self.proba_goals = np.ones([self.nb_goals_discovered]) / self.nb_goals_discovered
        else:
            self.proba_goals = self.eps_goal_selection / self.nb_goals_discovered + (1 - self.eps_goal_selection) * self.score_target_goals / self.score_target_goals.sum()

    def arreq_in_list(self, myarr, list_arrays):
        for i, elem in enumerate(list_arrays):
            if np.array_equal(elem, myarr):
                return True, i
        return False, None

    def sample(self, strategy):
        """

        :param strategy: (str) strategy to sample goal. Implemented:
            'random_from_memory': sample a random goal among the one said by the human partner.
        :return: a goal embedding
        """
        if self.rank == 0:
            if self.nb_goals_discovered == 0:
                # when there is no goal in memory, sample random goal from standard normal distribution
                return np.random.normal(size=self.goal_dim), -1
            else:
                if strategy == 'random_from_memory':
                    ind = np.random.randint(0, self.nb_goals_discovered)
                    return self.goal_discovered_encodings[ind], self.discovered_goal_oracle_ids[ind]
                elif strategy == 'bias_towards_rare_feedbacks':
                    # draw random goals with probability 0.2
                    # with proba 0.8, draw goals in proportion to their score: score_j = sum_i (1 / #feedback_positif_gi) * p(gi reached | gj targeted)
                    ind = np.random.choice(range(0, self.nb_goals_discovered), p=self.proba_goals)
                    return self.goal_discovered_encodings[ind], self.discovered_goal_oracle_ids[ind]
                else:
                    raise NotImplementedError
        else:
            raise ValueError('Rank should be 0')


class EvalGoalSampler:
    def __init__(self, language_model, nb_instr, replay=False):
        self.instructions = instructions[:nb_instr]
        self.nb_instructions = nb_instr
        self.goal_encodings = []
        for goal_str in self.instructions:
            self.goal_encodings.append(language_model.encode(goal_str).squeeze())
        self.count = -1
        self.replay = replay

    def update(self, goals_reached, goals_not_reached, goals_reached_str, goals_not_reached_str):
        pass

    def sample(self, goal_id):
        if self.replay:
            self.count += 1
            print(instructions[self.count])
            return self.goal_encodings[self.count]
        else:
            return self.goal_encodings[goal_id]



