import os
import csv

import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = [15, 10]

log_dir = "./log"

"""
Function plot the courbes of means [ [y0,y1,y2,...], [y0,y1,y2,...], ... ] with stds (same format)

"""
def plot_performance(X,means,stds,names, minimum = -np.inf, maximum=np.inf, save=True, path=None):
    fig, ax = plt.subplots(figsize=(16,9))
    colors = ['red','blue','green','orange']
    handles = []
    for i in range(len(means)):
        color = colors[i]
        mean, std = np.array(means[i]), np.array(stds[i])
        mean_p_std = np.minimum(mean+std, maximum)
        mean_m_std = np.maximum(mean-std, minimum)
        X = X[:len(mean)]
        plt.plot(X,mean, color = color, linewidth=3)
        ax.fill_between(X,mean_p_std , mean_m_std, color= color, alpha=0.15)
        plt.plot(X, mean_p_std, color=color, alpha=0.2)
        plt.plot(X, mean_m_std, color=color, alpha=0.2)
        handles.append( mlines.Line2D([], [], color = color, linewidth=3, label= names[i]))
    plt.title(names[0])
    plt.xlabel(names[1])
    plt.ylabel(names[2])
    if save:
        fig.savefig(path+"/"+names[0])
        plt.close(fig)
    else:
        plt.legend(handles=handles, numpoints = 2, loc="center left",ncol=2,bbox_to_anchor=(0.3, 1.07))
        plt.show()

def compute_hand_pos( arm_pos):
    arm_lengths = np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05])
    angles = np.cumsum(arm_pos[:7])
    angles_rads = np.pi * angles
    hand_pos = np.array([np.sum(np.cos(angles_rads) * arm_lengths),
                        np.sum(np.sin(angles_rads) * arm_lengths)
                        ])
    return hand_pos

def post_proc_rewards_solo(subdirs):
    means = []
    stds = []
    for subdir in subdirs:
        mean = []
        std = []
        rewards = loadmat(os.path.join(log_dir,subdir , "logs.mat"))['rewards'][0]
        init = len(rewards)
        while len(rewards[init-1])>1:
            init = init - 1
        for i in range(init,len(rewards)):
            mean.append(1+np.mean(rewards[i][:,-1:]))
            std.append(np.std(rewards[i][:,-1:]))
        means.append(mean)
        stds.append(std)
    iterations = [(i+1)*10 for i in range(len(means[0]))]
    return means,stds, iterations

def post_proc_rewards_multi(subdirs):
    means = []
    stds = []
    for subdir_param in subdirs:
        mean = []
        for subdir in subdir_param:
            rewards = loadmat(os.path.join(log_dir,subdir , "logs.mat"))['rewards'][0]
            for i in range(1,len(rewards)):
                if len(mean) < i:
                    mean.append([1+np.mean(rewards[i][:,-1:])])
                else:
                    mean[i-1].append(1+np.mean(rewards[i][:,-1:]))
        means.append(list(map(lambda x:np.mean(x),mean)))
        stds.append(list(map(lambda x:np.std(x),mean)))

    iterations = [(i+1)*10 for i in range(len(means[0]))]
    return means,stds, iterations

def compare_errors(subdirs, names=None):
    names = []
    means = []
    stds = []

    means2 = []
    stds2 = []

    for subdir_param in subdirs:
        mean = []
        mean2 = []
        for subdir in subdir_param:
            error = loadmat(os.path.join(log_dir,subdir , "logs.mat"))['errors']
            mean.append(error[0])
            mean2.append(error[1])
        means.append(np.mean(mean,axis=0))
        stds.append(np.std(mean,axis=0))
        means2.append(np.mean(mean2,axis=0))
        stds2.append(np.std(mean2,axis=0))

    iterations = [(i+1)*10 for i in range(len(means[0]))]
    sub_dir = subdirs[0][0]
    plot_performance(iterations,means,stds,["Prediction_Error_mono","learning episodes","Prediction error from one start"],  0, np.inf, True, log_dir+"/"+sub_dir)
    plot_performance(iterations,means2,stds2,["Prediction_Error_multi","learning episodes","Prediction error from random start"], 0, np.inf, True, log_dir+"/"+sub_dir)

def plot_initial_episode(sub_dir,path):
    names = ["Initial episode", "x","y"]
    obs = loadmat(os.path.join(log_dir,sub_dir , "logs.mat"))['observations'][0]
    fig, ax = plt.subplots(figsize=(16,9))
    handles = []

    init = len(obs)
    while len(obs[init-1])<50:
        init = init - 1

    for i in range(init):
        x = []
        y = []
        for pos in list(map(compute_hand_pos, obs[i])):
            x.append(pos[0])
            y.append(pos[1])
        plt.plot(x[:50],y[:50],c='green', linewidth=3)
    handles.append( mlines.Line2D([], [], color = 'green', linewidth=3, label= "hand movement"))
    for i in range(init):
        x = []
        y = []
        for pos in list(obs[i]):
            x.append(pos[7])
            y.append(pos[8])

        plt.plot(x[0],y[0],'o', c= 'red')
        plt.plot(x[:50],y[:50], 'red', linewidth=3, linestyle=":")
    handles.append( mlines.Line2D([], [], color = 'red', linestyle=":", linewidth=3, label= "ball movement"))

    plt.legend(handles=handles, numpoints = 2, loc="center left",ncol=2,bbox_to_anchor=(0, 1.02))
    plt.title(names[0])
    plt.xlabel(names[1])
    plt.ylabel(names[2])
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    fig.savefig(path+"/initial_episodes.svg")
    plt.close(fig)

def plot_episodes(sub_dir, path):
    names = ["Episode_", "x","y"]
    obs = loadmat(os.path.join(log_dir,sub_dir , "logs.mat"))['observations'][0]
    goals = loadmat(os.path.join(log_dir,sub_dir , "logs.mat"))['goals']
    init = len(obs)
    while len(obs[init-1])<50:
        init = init - 1

    for iteration in range(init,len(obs)):
        name = names[0]+str(init+(iteration-init+1)*10)
        fig, ax = plt.subplots(figsize=(16,9))

        handles = []
        for i in range(len(obs[iteration])):
            x = []
            y = []
            for pos in list(map(compute_hand_pos, obs[iteration][i])):
                x.append(pos[0])
                y.append(pos[1])
            plt.plot(x[0:50],y[0:50], linewidth=3)

        handles.append( mlines.Line2D([], [], color = 'orange', linewidth=3, label= "hand movement"))

        for i in range(len(obs[iteration])):
            x = []
            y = []
            for pos in list(obs[iteration][i]):
                x.append(pos[7])
                y.append(pos[8])
            plt.scatter(x[49:50],y[49:50],c='red',zorder=3)
            plt.scatter(x[1:2],y[1:2],c='red',zorder=3)
            g_x = goals[iteration-init][i][0]
            g_y = goals[iteration-init][i][1]
            plt.scatter(g_x,g_y, color='g', marker='x', s=200, zorder=4)
        handles.append( mlines.Line2D([], [], color = 'red', linestyle=":", linewidth=3, label= "ball movement"))

        plt.legend(handles=handles, numpoints = 2, loc="center left",ncol=2,bbox_to_anchor=(0, 1.02))

        plt.title(name)
        plt.xlabel(names[1])
        plt.ylabel(names[2])
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        fig.savefig(path+"/"+name+".svg")
        plt.close(fig)



def plot_xp( log_dir, sub_dir) :
    path = log_dir+"/"+sub_dir
    means, stds, iterations = post_proc_rewards_solo([sub_dir])
    plot_performance(iterations,means,stds, ["performance", "learning episodes", "success to perform the task"], 0,1, True, path)
    compare_errors([[sub_dir]], names=None)
    plot_initial_episode(sub_dir, path)
    plot_episodes(sub_dir, path)
