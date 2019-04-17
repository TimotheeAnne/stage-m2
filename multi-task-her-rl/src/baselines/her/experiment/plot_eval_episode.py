import pickle
import numpy as np
import matplotlib.pyplot as plt 
import os 

FIGSIZE = (16,9)

tasks = ['Move the gripper to the left', #0
                'Move the gripper to the right', #1
                'Move the gripper further', #2
                'Move the gripper closer', #3
                'Move the gripper higher', #4 
                'Move the gripper lower', #5
                'Move the gripper above the yellow object', #6
                'Touch the yellow object from above', #7
                'Throw the yellow object on the floor', #8
                'Move the yellow object to the left', #9
                'Move the yellow object to the right', #10
                'Move the yellow object away', #11
                'Move the yellow object closer', #12
                'Lift the yellow object', #13
                'Lift the yellow object higher', #14
                'Lift the yellow object and put it on the left', #15 
                'Lift the yellow object and put it on the right', #16
                'Lift the yellow object and place it further', #17
                 'Lift the yellow object and place it closer'] #18
                
colors = ['firebrick','forestgreen','steelblue','darkorchid',
          'red','limegreen','aqua','magenta',
          'orangered','springgreen','blue','crimson',
          'black','dimgrey','silver','yellow','chartreuse',
          'orchid']
LW = 3

def plot_episode(data,epoch, logdir):
    n_goals = len(data[epoch])
    fig,axarr = plt.subplots(1,1,figsize=FIGSIZE)
    for j in range(n_goals):
        [episode,rew] = data[epoch][j]
        for i in range(len(episode)):
            obs = episode['o'][i]
            plt.scatter( obs[0,0],obs[0,1], c='b')
            if i ==0:
                plt.plot(obs[:,1],obs[:,0], label=tasks[j]+[' failure',' success'][rew[0]], ls=['--','-'][rew[0]], lw=LW, c=colors[j])
            else:
                plt.plot(obs[:,1],obs[:,0], ls=['--','-'][rew[0]], lw=LW, c=colors[j])
            plt.plot(obs[:,4],obs[:,3], ls=[':','-.'][rew[0]], lw=LW, c=colors[j])

    plt.legend()
    plt.xlim((1.5,0))
    plt.ylim((1,1.6))
    plt.title("Epoch "+str(epoch))
    # ~ plt.show()
    with open(os.path.join(logdir, "Epoch_"+str(epoch)+".png"), "bw") as f:
        fig.savefig(f)
    plt.close(fig)

def plot_eval_episodes(logdir):
    filename = logdir+ 'eval_episodes.pk'
    # ~ filename = "/home/tim/Documents/stage-m2/multi-task-her-rl/src/data/MultiTaskFetchArmNLP1-v0/1300/eval_episodes.pk"
    # ~ logdir = "/home/tim/Documents/stage-m2/multi-task-her-rl/src/data/MultiTaskFetchArmNLP1-v0/1300/"
    data = []
    with open(filename, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    
    for epoch in range(len(data)):
        plot_episode(data,epoch, logdir)

# ~ plot_eval_episodes(None)
