from model import *
from evaluation import Evaluator
import gym
import gym_flowers 
from tqdm import tqdm
import datetime
import os 
import numpy as np
import argparse

# args
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=str, default=False)
parser.add_argument('-type', type=str, default="Random")
args = parser.parse_args()


if not int(args.gpu) :
    os.environ["CUDA_VISIBLE_DEVICES"] = ''


OBS_DIM = 18
ACS_DIM = 4
OUTPUT_DIM = 22
REG = 0.000
objects = [0,1,2,3,4]

EPOCH = 50
STEP = 10
N_EXPLORATIONS = 500
N_POPULATION = 1000
N_ELITES = N_POPULATION//20
N_SAMPLES = None
N_ITERATIONS = 10
B = 20
GRBF = False

training_data = None
eval_data = None
episodic_exploration = False

assert(args.type in ["Random","GRBF"])

if args.type == "Random":
    print("Random")
    GRBF = False

elif args.type == "GRBF":
    print("GRBF")
    GRBF = True
    

eval_data = "/home/tim/Documents/stage-m2/Tensorflow_model/data/ArmToolsToyR_eval.pk"

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

Observations = [[] for _ in range(DE.B)]

""" Episodic Exploration """
for iteration in tqdm(range(N_ITERATIONS)):
    for b in tqdm(range(DE.B)):
        for _ in range(N_EXPLORATIONS//N_ELITES):
            """ Select an action sequence to perform """
            Tmax = 50
            obs = env.reset()
            actions, uncertainty, pred = DE.select_actions(obs, N_POPULATION, N_ELITES, exploration = iteration, Tmax=Tmax, GRBF=False)
            observations = [[obs.copy()] for _ in range(N_ELITES)]
            
            """ Perform the action sequence """
            for j in range(N_ELITES):
                env.unwrapped.reset(obs)
                for t in range(Tmax):
                    # ~ #env.render()
                    obs, _, _, _ = env.step(actions[j][t])
                    observations[j] = np.concatenate( (observations[j], [obs]))
                Observations[b].append(obs[:18])

            # ~ if iteration > 0:
                # ~ DE.save_exploration(actions, uncertainty, observations, pred)
            
            """ Add the collected data to the replay buffer """
            DE.add_episodes(observations,  actions, b)
        # ~ DE.replay_buffers[b].pretty_print()

    """ Train the ensemble """ 
    for i in range(EPOCH//STEP):
        DE.train(STEP, verbose=False, validation=True, sampling='sample')

    DE.plot_training()

    for b in range(DE.B):
        with open(logdir+"/final_observations_r"+str(b)+"_"+str(iteration)+".pk", 'bw') as f:
            pickle.dump(np.array(Observations[b]), f)
    evaluator.eval(DE)

DE.save_ensemble()