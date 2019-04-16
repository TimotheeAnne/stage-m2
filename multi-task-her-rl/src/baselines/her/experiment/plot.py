import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

# font = {'size'   : 75}
# matplotlib.rc('font', **font)
instructions = ['Move the gripper to the left',
                'Move the gripper to the right', 
                'Move the gripper further', 
                'Move the gripper closer',
                'Move the gripper higher',
                'Move the gripper lower',
                'Move the gripper above the yellow object',
                'Touch the yellow object from above', 
                'Throw the yellow object on the floor',
                'Move the yellow object to the left',
                'Move the yellow object to the right',
                'Move the yellow object away',
                'Move the yellow object closer',
                'Lift the yellow object',
                'Lift the yellow object higher',
                'Lift the yellow object and put it on the left', 
                'Lift the yellow object and put it on the right',
                'Lift the yellow object and place it further', 'Lift the yellow object and place it closer']
                
matlab_colors2 = [[0,0.447,0.7410],[0.85,0.325,0.098],[0.466,0.674,0.188],[0.929,0.694,0.125],[0.494,0.1844,0.556],[0,0.447,0.7410],[0.3010,0.745,0.933],[0.85,0.325,0.098],
                  [0.466,0.674,0.188],[0.929,0.694,0.125],
                  [0.3010,0.745,0.933],[0.635,0.078,0.184]]
colors = matlab_colors2



# ~ folder_path = '/home/tim/Documents/stage-m2/multi-task-her-rl/src/data/myMultiTaskFetchArmNLP-v0/'
folder_path = '/home/tim/Documents/stage-m2/multi-task-her-rl/src/data/MultiTaskFetchArmNLP1-v0/'
folder_path = '/home/tim/Documents/stage-m2/multi-task-her-rl/src/data/from_remote/'
trials = ['13_onmyEnv_85%'] #list(range(30,40))#

track_time = False
live_plot = False
use_groups = True

# ~ groups = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 10, 12, 13, 14, 15], [20, 22, 24, 25, 26, 27], [32, 34], [9, 11, 16, 17, 18, 19], [21, 23, 28, 29, 30, 31], [33, 35]]
groups = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12]]

groups_colors = [[0.92, 0.28, 0.28], [0.98, 0.83, 0.37], [0.92, 0.71, 0.039], [0.90, 0.56, 0.05], [0.59, 0.74, 0.93], [0.28, 0.58, 0.92], [0.054, 0.30, 0.68]]
group_legend = ['Gripper', 'Move yellow cube', 'Lift yellow cube', 'Stack yellow cube', 'Move blue cube',  'Lift blue cube','Stack blue cube']

for trial in trials:
    print('Ploting trial', trial)
    path = folder_path + str(trial) + '/'

    # extract params from json
    with open(path + 'params.json') as json_file:
        params = json.load(json_file)

    nb_instr = params['nb_goals']
    n_cycles = params['n_cycles']
    rollout_batch_size = params['rollout_batch_size']
    n_cpu = params['num_cpu']

    # extract results
    data = pd.read_csv(path+'progress.csv')

    n_points = data['test/success_goal_0'].shape[0]
    episodes = data['train/episode']

    n_epochs = len(episodes)
    n_eps = n_cpu * rollout_batch_size * n_cycles
    episodes = np.arange(n_eps, n_epochs * n_eps + 1, n_eps)
    episodes = episodes / 1000

    task_success_rates = np.zeros([n_points, nb_instr])
    learning_progress = np.zeros([n_points, nb_instr])
    competence = np.zeros([n_points, nb_instr])
    probas_goal_selection = np.zeros([n_points, nb_instr])
    for i in range(nb_instr):
        task_success_rates[:, i] = data['test/success_goal_' + str(i)]
        # ~ learning_progress[:, i] = data['perceived_LP_' + str(i)]
        # ~ competence[:, i] = data['perceived_C_' + str(i)]
        # ~ probas_goal_selection[:, i] = data['proba_select_' + str(i)]


    zero_success_rates = task_success_rates.copy()
    for i in range(zero_success_rates.shape[0]):
        for j in range(zero_success_rates.shape[1]):
            if np.isnan(zero_success_rates[i, j]):
                zero_success_rates[i, j] = 0

    if use_groups:
        nb_groups = len(groups)
        group_task_success_rates = np.zeros([n_points, nb_groups])
        group_learning_progress = np.zeros([n_points, nb_groups])
        group_competence = np.zeros([n_points, nb_groups])
        group_probas_goal_selection = np.zeros([n_points, nb_groups])
        for i in range(nb_groups):
            group_task_success_rates[:, i] = zero_success_rates[:, groups[i]].mean(axis=1)
            group_learning_progress[:, i] = learning_progress[:, groups[i]].mean(axis=1)
            group_competence[:, i] = competence[:, groups[i]].mean(axis=1)
            group_probas_goal_selection[:, i] = probas_goal_selection[:, groups[i]].mean(axis=1)



    if live_plot:
        fig, ax = plt.subplots()
        annot = ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        lines = plt.plot(episodes, competence, linewidth=5)

        def update_annot(ind, i_l):
            x,y = lines[i_l].get_data()
            annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
            text = instructions[i_l]
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            if event.inaxes == ax:
                for i_l, l in enumerate(lines):
                    cont, ind = l.contains(event)
                    if cont:
                        update_annot(ind, i_l)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.show()


        fig, ax = plt.subplots()
        annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        lines = plt.plot(episodes, learning_progress, linewidth=5)
        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.show()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # plot success_rate

    fig = plt.figure(figsize=(22, 15), frameon=False)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.tick_params(width=4, direction='in', length=10, labelsize='small')
    p = plt.plot(episodes, zero_success_rates.mean(axis=1), linewidth=10)#, c=colors[i])
    # leg = plt.legend(['task ' + str(i) for i in range(nb_instr)], frameon=False)
    lab = plt.xlabel('Episodes (x$10^3$)')
    plt.ylim([-0.01, 1.01])
    plt.yticks([0.25, 0.50, 0.75, 1])
    lab2 = plt.ylabel('Average success rate')
    plt.savefig(path + 'plot_av_success_rate.png', bbox_extra_artists=(lab,lab2), bbox_inches='tight', dpi=300) # add leg

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # plot evolution of competence
    fig = plt.figure(figsize=(22, 15), frameon=False)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.tick_params(width=4, direction='in', length=10, labelsize='small')
    if not use_groups:
        for i in range(nb_instr):
            p = plt.plot(episodes, task_success_rates[:, i], linewidth=3)#, c=colors[i])
    else:
        for i in range(nb_groups):
            p = plt.plot(episodes, group_task_success_rates[:, i], linewidth=10, c=groups_colors[i])

    lab = plt.xlabel('Episodes (x$10^3$)')
    plt.ylim([-0.01, 1.01])
    plt.yticks([0.25, 0.50, 0.75, 1])
    lab2 = plt.ylabel('Competence')
    if use_groups:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
        leg = ax.legend(group_legend, loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2,
                        borderaxespad=0, frameon=False, prop={'size': 30})

    plt.savefig(path + 'plot_success_rates.png', bbox_extra_artists=(lab, lab2, leg), bbox_inches='tight', dpi=300) # add leg

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # plot evolution of subjective competence
    fig = plt.figure(figsize=(22, 15), frameon=False)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.tick_params(width=4, direction='in', length=10, labelsize='small')
    for i in range(nb_instr):
        p = plt.plot(episodes, competence[:, i], linewidth=10)#, c=colors[i])
    # leg = plt.legend(['task ' + str(i) for i in range(nb_instr)], frameon=False)
    lab = plt.xlabel('Episodes (x$10^3$)')
    plt.ylim([-0.01, 1.01])
    plt.yticks([0.25, 0.50, 0.75, 1])
    lab2 = plt.ylabel('Perceived competence')
    plt.savefig(path + 'plot_c.png', bbox_extra_artists=(lab,lab2), bbox_inches='tight', dpi=300) # add leg

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # plot evolution of subjective learning progress
    fig = plt.figure(figsize=(22, 15), frameon=False)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.tick_params(width=4, direction='in', length=10, labelsize='small')
    for i in range(nb_instr):
        p = plt.plot(episodes, learning_progress[:, i], linewidth=10)  # , c=colors[i])
    # leg = plt.legend(['task ' + str(i) for i in range(nb_instr)], frameon=False)
    lab = plt.xlabel('Episodes (x$10^3$)')
    plt.ylim([-0.01, 1.01])
    plt.yticks([0.25, 0.50, 0.75, 1])
    lab2 = plt.ylabel('Perceived LP')
    plt.savefig(path + 'plot_lp.png', bbox_extra_artists=(lab, lab2), bbox_inches='tight', dpi=300)  # add leg

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # plot evolution of proba goal selection

    fig = plt.figure(figsize=(22, 15), frameon=False)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.tick_params(width=4, direction='in', length=10, labelsize='small')
    for i in range(nb_instr):
        p = plt.plot(episodes, probas_goal_selection[:, i], linewidth=10)  # , c=colors[i])
    # leg = plt.legend(['task ' + str(i) for i in range(nb_instr)], frameon=False)
    lab = plt.xlabel('Episodes (x$10^3$)')
    plt.ylim([-0.01, 1.01])
    plt.yticks([0.25, 0.50, 0.75, 1])
    lab2 = plt.ylabel('Perceived competence')
    plt.savefig(path + 'plot_probas.png', bbox_extra_artists=(lab, lab2), bbox_inches='tight', dpi=300)  # add leg


    if track_time:
        computation_time = data['epoch_duration (s)']
        time_batch = data['time_batch']
        time_epoch = data['time_epoch']
        time_eval = data['time_eval']
        time_info = data['time_info']
        time_rollout = data['time_rollout']
        time_store = data['time_store']
        time_train = data['time_train']
        time_update = data['time_update']
        time_reset = data['time_reset']
        time_env = data['time_env']
        time_goal_sampler = data['time_goal_sampler']
        time_social_peer = data['time_social_peer']
        time_encoding = data['time_encoding']
        time_get_classif_samples = data['time_get_classif_samples']
        time_stuff = [time_update, time_batch, time_train, time_info, time_rollout, time_store, time_eval, time_epoch,
                      time_reset, time_env, time_social_peer, time_goal_sampler, time_get_classif_samples, time_encoding]
        legends = ['time_update', 'time_batch', 'time_train', 'time_info', 'time_rollout', 'time_store', 'time_eval', 'time_epoch',
                   'time_reset', 'time_env', 'time_social_peer', 'time_goal_sampler', 'time_get_classif_samples', 'time_encoding']
        discoveries = np.zeros([nb_instr])
        discoveries.fill(np.nan)
        for i in range(nb_instr):
            ind_nan = np.argwhere(np.isnan(task_success_rates[:, i]))
            if ind_nan.size == 0:
                discoveries[i] = 0
            else:
                discoveries[i] = ind_nan[-1][0]


        # plot computation time per epoch
        plt.figure()
        plt.plot(episodes, computation_time)
        for d in discoveries:
            plt.axvline(x=episodes[int(d)])

        plt.figure()
        for i in range(len(time_stuff)):
            plt.plot(episodes, time_stuff[i])
        plt.legend(legends)
        for d in discoveries:
            plt.axvline(x=episodes[int(d)])
        plt.show()

