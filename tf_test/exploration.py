from model import *
from evaluation import Evaluator
import gym
import gym_flowers 
from tqdm import tqdm
import datetime
import os 
import numpy as np

OBS_DIM = 18
ACS_DIM = 4
OUTPUT_DIM = 22
REG = 0.000
objects = [0,1,2,3,4]

EPOCH = 50
STEP = 5
N_EXPLORATIONS = 100000
N_POPULATION = 1000
N_ELITES = N_POPULATION//10
N_SAMPLES = None
N_ITERATIONS = 10
B = 5
GRBF = False

training_data = None
eval_data = None
episodic_exploration = False

# ~ training_data = "/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy-v1_4000_train.pk"
eval_data = "/home/tim/Documents/stage-m2/tf_test/data/ArmToolsToy_1000pertinent.pk"

timestamp = datetime.datetime.now()
logdir = './log_exploration/'+str(timestamp)
os.makedirs(logdir)

config = "" 
config += "EPOCH: "+str(EPOCH) +"\n"
config += "N_EXPLORATIONS: "+str(N_EXPLORATIONS)+"\n"
config += "N_POPULATION: "+str(N_POPULATION)+"\n"
config += "N_ELITES: "+str(N_ELITES)+"\n"
config += "N_SAMPLES: "+str(N_SAMPLES)+"\n"
config += "N_ITERATIONS: "+str(N_ITERATIONS)+"\n"
config += "STEP: "+str(STEP)+"\n"
config += "REG: "+str(REG)+"\n"
config += "B : "+str(B )+"\n"
config += "training_data: "+str(training_data)+"\n"
config += "eval_data: "+str(eval_data )+"\n"
config += "objects: "+ str(objects) + "\n"
config += "episodic_exploration: "+ str(episodic_exploration) + "\n"
config += "GRBF: "+ str(GRBF) + "\n"

with open(logdir+"/config.txt", 'w') as f:
    f.write(config)

DE = Ensemble(OBS_DIM, ACS_DIM, OUTPUT_DIM, reg=REG, B=B, logdir=logdir, objects=objects)
evaluator = Evaluator( None, eval_data, logdir, OBS_DIM )
env = gym.make('ArmToolsToys-v1')

Observations = []
Actions = []

""" Episodic Exploration """
for iteration in tqdm(range(N_ITERATIONS)):
    for _ in range(N_EXPLORATIONS//(N_ELITES*N_ITERATIONS)):
        """ Select an action sequence to perform """
        Tmax = 50
        obs = env.reset()
        actions, uncertainty, pred = DE.select_actions(obs, N_POPULATION, N_ELITES, exploration = True, Tmax=Tmax, GRBF=False)
        observations = [[obs.copy()] for _ in range(N_ELITES)]
        
        """ Perform the action sequence """
        for j in range(N_ELITES):
            obs = env.reset()
            for t in range(Tmax):
                # ~ #env.render()
                obs, _, _, _ = env.step(actions[j][t])
                observations[j] = np.concatenate( (observations[j], [obs]))

        DE.save_exploration(actions, uncertainty, observations, pred)
        
        """ Add the collected data to the replay buffer """
        DE.add_episodes(observations,  actions)
        DE.replay_buffer.pretty_print()

        """ Train the ensemble """ 
        for i in range(EPOCH//STEP):
            DE.train(STEP, verbose=True, validation=True, sampling='choice')

        DE.plot_training()

    """ Evaluate """
    with open(logdir+"/final_observations_"+str(iteration)+".pk", 'bw') as f:
        pickle.dump(np.array(observations), f)
    evaluator.eval(DE)

DE.save_ensemble()
