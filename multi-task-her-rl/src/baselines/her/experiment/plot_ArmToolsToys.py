import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib


# font = {'size'   : 75}
# matplotlib.rc('font', **font)
instructions = ['Move the Hand to the left', #0 44% 42%
                'Move the Hand to the right',  #1 44% 42%
                'Move the Hand further', #2 0% 0%
                'Move the Hand closer',  #3 94% 99%
                
                'Grasp Stick1', #4 9.7% 0.8%
                'Grasp Stick2', #5 9.7% 0.8%
                
                'Move the Stick1 to the left', #6 3.5% 
                'Move the Stick1 to the right',  #7 2.6%
                'Move the Stick1 further', #8 1.6%
                'Move the Stick1 closer', #9 6.8%

                'Move the Stick2 to the left', #10 2.7%
                'Move the Stick2 to the right', #11 3.4%
                'Move the Stick2 further', #12 1.6%
                'Move the Stick2 closer', #13 6.9%
                
                'Move the Stick1 50% closer to the Magnet', #14
                'Move the Stick2 50% closer to the Scratch', #15
                
                'Grasp the Magnet1', #16
                'Grasp the Scratch1', #17
                
                'Move the Magnet1 to the left', #18 6 27
                'Move the Magnet1 to the right', #19 7 10
                'Move the Magnet1 further', #20 1 4
                'Move the Magnet1 closer', #21 6 31

                'Move the Scratch1 to the left', #22 6 8
                'Move the Scratch1 to the right', #23 3 28
                'Move the Scratch1 further', #24 1 1
                'Move the Scratch1 closer', #25 5 33
                
                """ To help more"""
                'Move the Stick1 25% closer to the Magnet', #26
                'Move the Stick2 25% closer to the Scratch', #27
            
                'Move the Stick1 75% closer to the Magnet', #28
                'Move the Stick2 75% closer to the Scratch', #29
]
                
matlab_colors2 = [[0,0.447,0.7410],[0.85,0.325,0.098],[0.466,0.674,0.188],[0.929,0.694,0.125],
                  [0.494,0.1844,0.556],[0,0.447,0.7410],[0.3010,0.745,0.933],[0.85,0.325,0.098],
                  [0.466,0.674,0.188],[0.929,0.694,0.125], [0.3010,0.745,0.933],[0.635,0.078,0.184]]
                  

                  
colors = matlab_colors2

my_colors = ['crimson','royalblue','forestgreen','darkorange','orchid']

# ~ folder_path = '/home/tim/Documents/stage-m2/multi-task-her-rl/src/data/myMultiTaskFetchArmNLP-v0/'
# ~ folder_path = '/home/tim/Documents/stage-m2/multi-task-her-rl/src/data/MultiTaskFetchArmNLP1-v0/'
folder_path = "/home/tim/Documents/stage-m2/multi-task-her-rl/src/data/ArmToolsToys-v1/"
# ~ folder_path = '/home/tim/Documents/stage-m2/multi-task-her-rl/src/data/from_remote/'
trials = ['5000'] #list(range(30,40))#

track_time = False
live_plot = False
use_groups = True


# ~ groups = [[0,1,2,3],[4,6,7,8,9],[5,10,11,12,13],[14,16,18,19,20,21], [17,22,23,24,25]]
groups = [[0,1,2,3],[4],[6,7,8,9],[26,14,27] , [16] , [18,19,20,21], [5],[10,11,12,13], [28,15,29], [17], [22, 23,24,25]]

groups_colors = [[0.92, 0.28, 0.28], [0.98, 0.83, 0.37], [0.92, 0.71, 0.039], [0.90, 0.56, 0.05],
                 [0.59, 0.74, 0.93], [0.28, 0.58, 0.92], [0.054, 0.30, 0.68]]
                 
groups_colors = ['grey', 
            [0.98, 0.83, 0.37], [0.92, 0.71, 0.039], [0.90, 0.56, 0.05],[0.92, 0.28, 0.28], 'red',
            [0.59, 0.74, 0.93] ,[0.28, 0.58, 0.92]  , [0.054, 0.30, 0.68], 'blue',  'navy' ]

group_legend = ['Hand', 
                'Grasp Stick1','Move Stick1', 'Bring Stick1 Closer to Magnet', 'Grasp Magnet', 'Move Magnet',  
                'Grasp Stick2','Move Stick2', 'Bring Stick2 Closer to Scratch', 'Grasp Scratch','Move Scratch']

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

    # ~ # plot Model evaluation
    # ~ fig = plt.figure(figsize=(22, 15), frameon=False)
    # ~ ax = fig.add_subplot(111)
    # ~ ax.spines['top'].set_linewidth(6)
    # ~ ax.spines['right'].set_linewidth(6)
    # ~ ax.spines['bottom'].set_linewidth(6)
    # ~ ax.spines['left'].set_linewidth(6)
    # ~ ax.tick_params(width=4, direction='in', length=10, labelsize='small')
    
    # ~ legend = ["TPR","TNR","ACC","F1 score"]
        
    # ~ for i in range(4):
        # ~ p = plt.plot(episodes, data['test/rate_'+legend[i]], linewidth=5, c=my_colors[i])
    
    # ~ lab = plt.xlabel('Episodes (x$10^3$)')
    # ~ plt.ylim([-0.01, 1.01])
    # ~ plt.yticks([0.25, 0.50, 0.75, 1])
    # ~ lab2 = plt.ylabel('Competence')

    # ~ box = ax.get_position()
    # ~ ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    # ~ leg = ax.legend(legend, loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2,
                    # ~ borderaxespad=0, frameon=False, prop={'size': 30})

    # ~ plt.savefig(path + 'plot_model_evaluation.png', bbox_extra_artists=(lab, lab2, leg), bbox_inches='tight', dpi=300) # add leg

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # ~ # plot Matrice confusion
    # ~ fig = plt.figure(figsize=(22, 15), frameon=False)
    # ~ ax = fig.add_subplot(111)
    # ~ ax.spines['top'].set_linewidth(6)
    # ~ ax.spines['right'].set_linewidth(6)
    # ~ ax.spines['bottom'].set_linewidth(6)
    # ~ ax.spines['left'].set_linewidth(6)
    # ~ ax.tick_params(width=4, direction='in', length=10, labelsize='small')
    
    # ~ legend = ["FP","FN"]
    
    # ~ for i in range(2):
        # ~ p = plt.plot(episodes, data['test/count_'+legend[i]], linewidth=5, c=my_colors[i])
    
    # ~ lab = plt.xlabel('Episodes (x$10^3$)')

    # ~ lab2 = plt.ylabel('Competence')

    # ~ box = ax.get_position()
    # ~ ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
    # ~ leg = ax.legend(legend, loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2,
                    # ~ borderaxespad=0, frameon=False, prop={'size': 30})

    # ~ plt.savefig(path + 'plot_matrice_confusion.png', bbox_extra_artists=(lab, lab2, leg), bbox_inches='tight', dpi=300) # add leg


